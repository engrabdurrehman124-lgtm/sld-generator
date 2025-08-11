from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session
import os
import pandas as pd
import tempfile
import utm
import re
from werkzeug.utils import secure_filename
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-to-random')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True  # Use only over HTTPS in production
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # Session expires after 30 minutes of inactivity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Passcode for access control
PASSCODE = os.environ.get('PASSCODE', 'SLD2025')

ALLOWED_EXTENSIONS = {'xlsx'}  # Remove MDB for Vercel deployment

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_passcode():
    return session.get('authenticated', False)

# Grid spacing for diagram generation
grid_spacing = 10

def sort_key(row):
    """Custom sort key function for DataFrame rows"""
    feeder_id = str(row['FeederId']).strip()
    from_node_id = str(row['FromNodeId_Logical'] if 'FromNodeId_Logical' in row else row['FromNodeId']).strip()

    if from_node_id == feeder_id:
        return (0, feeder_id, 0, 0)
    else:
        parts = from_node_id.split('-')
        prefix = parts[0]
        if len(parts) > 1:
            second_part = parts[1]
            t_flag = 1 if 'T' in second_part else 0
            num = int(''.join(filter(str.isdigit, second_part)) or 0)
        else:
            t_flag = 0
            num = 0
        return (1, prefix, t_flag, num)

def add_logical_numbering(df):
    """Add logical numbering to FromNodeId and ToNodeId columns"""
    node_to_number = {}
    current_number = 1

    from_col = 'FromNodeId'
    to_col = 'ToNodeId'
    
    for node in pd.concat([df[from_col], df[to_col]]).unique():
        node_to_number[str(node)] = f"{current_number:02d}"
        current_number += 1

    df['FromNodeId_Logical'] = df[from_col].astype(str).map(node_to_number)
    df['ToNodeId_Logical'] = df[to_col].astype(str).map(node_to_number)

    from_idx = df.columns.get_loc(from_col) + 1
    df.insert(from_idx, 'FromNodeId_Logical', df.pop('FromNodeId_Logical'))

    to_idx = df.columns.get_loc(to_col) + 1
    df.insert(to_idx, 'ToNodeId_Logical', df.pop('ToNodeId_Logical'))

    return df

def drop_latlon_columns(df):
    """Drop coordinate columns and keep only logical numbering"""
    columns_to_drop = ['From_X', 'From_Y', 'To_X', 'To_Y', 'FromNodeId', 'ToNodeId']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df

def shift_description_chain(df):
    """Shift descriptions backward up the chain"""
    to_node_map = df.set_index('ToNodeId_Logical').to_dict()['Description']

    for idx, row in df.iterrows():
        from_node = row['FromNodeId_Logical']
        desc = row['Description']

        if pd.notna(desc) and desc.strip() != '':
            if from_node in to_node_map:
                parent_idx = df[df['ToNodeId_Logical'] == from_node].index
                if not parent_idx.empty:
                    df.at[parent_idx[0], 'Description'] = desc
                    df.at[idx, 'Description'] = ''
    return df

def remove_lnode_rows(df):
    """Remove rows with ToNodeId matching FeederID-LnodeXX pattern"""
    pattern = re.compile(r"^[\w\s]+-Lnode\d+$")
    filtered_df = df[~df['ToNodeId_Logical'].apply(lambda x: bool(pattern.fullmatch(str(x))))].reset_index(drop=True)
    return filtered_df

def line_intersection(p1, p2, p3, p4):
    """Check if line segments p1-p2 and p3-p4 intersect."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return False

    o1 = ccw(p1, p3, p4)
    o2 = ccw(p2, p3, p4)
    o3 = ccw(p1, p2, p3)
    o4 = ccw(p1, p2, p4)

    if o1 != o2 and o3 != o4:
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
    return False

def find_crossing_edges(edges, node_positions):
    """Identify pairs of edges that intersect."""
    crossings = []
    for i, (from1, to1) in enumerate(edges):
        if from1 not in node_positions or to1 not in node_positions:
            continue
        p1 = node_positions[from1]
        p2 = node_positions[to1]
        for j, (from2, to2) in enumerate(edges[i+1:], start=i+1):
            if from2 not in node_positions or to2 not in node_positions:
                continue
            p3 = node_positions[from2]
            p4 = node_positions[to2]
            if from1 in (from2, to2) or to1 in (from2, to2):
                continue
            if line_intersection(p1, p2, p3, p4):
                crossings.append(((from1, to1), (from2, to2)))
    return crossings

def check_node_overlap(node_positions, min_distance=grid_spacing/2):
    """Check for nodes closer than min_distance."""
    overlaps = []
    nodes = list(node_positions.items())
    for i, (node1, pos1) in enumerate(nodes):
        for node2, pos2 in nodes[i+1:]:
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            if dist < min_distance:
                overlaps.append((node1, node2))
    return overlaps

def build_adjacency_graph(edges):
    """Build an adjacency list representation of the graph"""
    graph = {}
    for from_node, to_node in edges:
        if from_node not in graph:
            graph[from_node] = []
        if to_node not in graph:
            graph[to_node] = []
        graph[from_node].append(to_node)
        graph[to_node].append(from_node)
    return graph

def get_connected_subgraph(graph, start_node, exclude_edge=None):
    """Get all nodes connected to start_node, optionally excluding a specific edge"""
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        for neighbor in graph.get(node, []):
            if exclude_edge and ((node, neighbor) == exclude_edge or (neighbor, node) == exclude_edge):
                continue
            if neighbor not in visited:
                stack.append(neighbor)

    return visited

def shift_nodes_recursively(node_positions, graph, moved_node, shift_x, shift_y, fixed_node):
    """Recursively shift all nodes connected to moved_node, except those connected to fixed_node"""
    exclude_edge = (fixed_node, moved_node)
    nodes_to_move = get_connected_subgraph(graph, moved_node, exclude_edge)
    nodes_to_move.discard(moved_node)

    for node in nodes_to_move:
        if node != fixed_node:
            old_x, old_y = node_positions[node]
            node_positions[node] = (old_x + shift_x, old_y + shift_y)

def generate_diagram_plots(df):
    """Generate diagrams with crossing resolution and return as base64 encoded images"""
    plots = {}
    
    # Build edges and node positions
    node_positions = {}
    used_positions = set()
    edges = []

    direction_preferences = [
        (grid_spacing, 0), (0, grid_spacing), (0, -grid_spacing),
        (-grid_spacing, 0), (grid_spacing, grid_spacing),
        (-grid_spacing, grid_spacing), (grid_spacing, -grid_spacing),
        (-grid_spacing, -grid_spacing)
    ]

    def find_available_position(ref_pos):
        ref_x, ref_y = ref_pos
        for dx, dy in direction_preferences:
            new_pos = (ref_x + dx, ref_y + dy)
            if new_pos not in used_positions:
                return new_pos
        return (ref_x + grid_spacing, ref_y + grid_spacing)

    start_x, start_y = 0, 0

    for row_index, (_, row) in enumerate(df.iterrows()):
        from_node = str(row['FromNodeId_Logical'])
        to_node = str(row['ToNodeId_Logical'])
        if pd.isna(from_node) or pd.isna(to_node):
            continue

        if from_node not in node_positions:
            if row_index == 0:
                node_positions[from_node] = (start_x, start_y)
            else:
                ref_pos = list(node_positions.values())[-1]
                node_positions[from_node] = find_available_position(ref_pos)
            used_positions.add(node_positions[from_node])

        from_pos = node_positions[from_node]

        if to_node in node_positions:
            to_node = f"{to_node}_dup{row_index}"

        to_pos = find_available_position(from_pos)
        node_positions[to_node] = to_pos
        used_positions.add(to_pos)
        edges.append((from_node, to_node))

    # Generate base diagram
    fig, ax = plt.subplots(figsize=(30, 24))
    for from_node, to_node in edges:
        if from_node in node_positions and to_node in node_positions:
            x1, y1 = node_positions[from_node]
            x2, y2 = node_positions[to_node]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', color='lightblue', markersize=8, markeredgecolor='black')
        ax.text(x, y - 2, node.split('_')[0], fontsize=10, ha='center', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title("Base Logical Single-Line Diagram", fontsize=16, weight='bold')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plots['base_diagram'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots

def process_xlsx_file(xlsx_filepath):
    """Process the uploaded XLSX file and return processed DataFrame"""
    try:
        df_section = pd.read_excel(xlsx_filepath)
        
        required_columns = ['FeederId', 'FromNodeId', 'ToNodeId', 'Description', 'PhaseConductorId', 'SectionLength_MUL']
        missing_columns = [col for col in required_columns if col not in df_section.columns]
        
        # Remove rows where PhaseConductorId is ANT, ant, WASP, or wasp (case-insensitive)
        initial_count = len(df_section)
        df_section = df_section[~df_section['PhaseConductorId'].str.lower().isin(['ant', 'wasp'])].reset_index(drop=True)
        removed_count = initial_count - len(df_section)
        logger.info(f"Removed {removed_count} rows with PhaseConductorId 'ANT' or 'WASP'.")
        
        if missing_columns:
            df_section = add_logical_numbering(df_section)
            df_section = drop_latlon_columns(df_section)
        
        df_section = shift_description_chain(df_section)
        df_section = remove_lnode_rows(df_section)

        return df_section

    except Exception as e:
        logger.error(f"Error processing XLSX file: {str(e)}")
        raise

@app.route('/')
def index():
    if not check_passcode():
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        entered_passcode = request.form.get('passcode')
        if entered_passcode == PASSCODE:
            session['authenticated'] = True
            session.permanent = True
            return redirect(url_for('index'))
        else:
            flash('Invalid passcode. Please try again.')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if not check_passcode():
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            
            # Use in-memory processing for Vercel
            file_content = BytesIO(file.read())
            
            df_processed = process_xlsx_file(file_content)
            plots = generate_diagram_plots(df_processed)
            
            return render_template('results.html', plots=plots, filename=filename)
        
        except Exception as e:
            return f"Error processing file: {str(e)}"
    else:
        return "Invalid file type. Please upload a .xlsx file."

# For Vercel
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)

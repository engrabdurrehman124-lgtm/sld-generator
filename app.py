from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session
import os
import subprocess
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
import zipfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-to-random'  # Change this to a random secret key
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'Uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Passcode for access control
PASSCODE = "SLD2025"  # Change this to your desired passcode

ALLOWED_EXTENSIONS = {'mdb', 'xlsx'}

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

def shift_transformers_to_parent(df):
    """Shift transformer descriptions to parent rows"""
    for idx, row in df.iterrows():
        desc = str(row['Description']).strip()
        from_node = row['FromNodeId_Logical']
        phase_conductor = str(row['PhaseConductorId']).strip()
        section_length = float(row['SectionLength_MUL'])

        if desc and phase_conductor.upper() == "GOPHER" and abs(section_length - 1.2192) < 0.001:
            parent_rows = df[df['ToNodeId_Logical'] == from_node]
            if not parent_rows.empty:
                parent_idx = parent_rows.index[0]
                df.at[parent_idx, 'Description'] = desc
                df.at[idx, 'Description'] = ''
    return df

def remove_redundant_gopher_rows(df):
    """Remove redundant GOPHER rows"""
    initial_count = len(df)
    condition = (
        (df['PhaseConductorId'].str.upper() == 'GOPHER') &
        (abs(df['SectionLength_MUL'] - 1.2192) < 0.001) &
        (df['Description'].astype(str).str.strip() == '')
    )
    df_cleaned = df[~condition].reset_index(drop=True)
    removed_count = initial_count - len(df_cleaned)
    logger.info(f"Removed {removed_count} redundant GOPHER rows.")
    return df_cleaned

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
    fig, ax = plt.subplots(figsize=(20, 16))
    for from_node, to_node in edges:
        if from_node in node_positions and to_node in node_positions:
            x1, y1 = node_positions[from_node]
            x2, y2 = node_positions[to_node]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', color='lightblue', markersize=6, markeredgecolor='black')
        ax.text(x, y - 2, node.split('_')[0], fontsize=8, ha='center', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title("Base Logical Single-Line Diagram", fontsize=14, weight='bold')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plots['base_diagram'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Resolve crossings and overlaps
    graph = build_adjacency_graph(edges)
    direction_vectors = {
        'up': (0, grid_spacing),
        'down': (0, -grid_spacing),
        'left': (-grid_spacing, 0),
        'right': (grid_spacing, 0),
        'upright': (grid_spacing, grid_spacing),
        'upleft': (-grid_spacing, grid_spacing),
        'downright': (grid_spacing, -grid_spacing),
        'downleft': (-grid_spacing, -grid_spacing)
    }
    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        crossings = find_crossing_edges(edges, node_positions)
        overlaps = check_node_overlap(node_positions)

        if not crossings and not overlaps:
            logger.info("No crossings or overlaps detected.")
            break

        logger.info(f"Iteration {iteration + 1}: Found {len(crossings)} crossings and {len(overlaps)} overlaps")

        for (from1, to1), (from2, to2) in crossings:
            if from1 in node_positions and to1 in node_positions:
                x1, y1 = node_positions[from1]
                old_x2, old_y2 = node_positions[to1]
                if abs(old_x2 - x1) == grid_spacing and abs(old_y2 - y1) == grid_spacing:
                    new_x2 = x1 + 4 * grid_spacing
                    new_y2 = y1
                    shift_x = new_x2 - old_x2
                    shift_y = new_y2 - old_y2
                    node_positions[to1] = (new_x2, new_y2)
                    shift_nodes_recursively(node_positions, graph, to1, shift_x, shift_y, from1)
                    logger.info(f"Stretched edge {from1} → {to1} to resolve crossing")
                    continue
                for direction, (dir_x, dir_y) in direction_vectors.items():
                    new_x2 = x1 + dir_x
                    new_y2 = y1 + dir_y
                    temp_positions = node_positions.copy()
                    temp_positions[to1] = (new_x2, new_y2)
                    shift_x = new_x2 - old_x2
                    shift_y = new_y2 - old_y2
                    shift_nodes_recursively(temp_positions, graph, to1, shift_x, shift_y, from1)
                    new_crossings = find_crossing_edges(edges, temp_positions)
                    if len(new_crossings) < len(crossings):
                        node_positions[to1] = (new_x2, new_y2)
                        shift_nodes_recursively(node_positions, graph, to1, shift_x, shift_y, from1)
                        logger.info(f"Changed {from1} → {to1} to {direction} to reduce crossings")
                        break

        for node1, node2 in overlaps:
            if node2 in node_positions:
                x1, y1 = node_positions[node1]
                old_x2, old_y2 = node_positions[node2]
                new_pos = None
                for dx, dy in direction_vectors.values():
                    temp_pos = (old_x2 + dx, old_y2 + dy)
                    if temp_pos not in node_positions.values():
                        new_pos = temp_pos
                        break
                if new_pos:
                    node_positions[node2] = new_pos
                    shift_x = new_pos[0] - old_x2
                    shift_y = new_pos[1] - old_y2
                    shift_nodes_recursively(node_positions, graph, node2, shift_x, shift_y, node1)
                    logger.info(f"Moved node {node2} to resolve overlap with {node1}")

        iteration += 1

    if iteration == max_iterations:
        logger.warning("Max iterations reached. Some crossings or overlaps may remain.")

    # Generate final extended diagram
    fig, ax = plt.subplots(figsize=(20, 16))
    for from_node, to_node in edges:
        if from_node in node_positions and to_node in node_positions:
            x1, y1 = node_positions[from_node]
            x2, y2 = node_positions[to_node]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', color='lightblue', markersize=6, markeredgecolor='black')
        ax.text(x, y - 2, node.split('_')[0], fontsize=8, ha='center', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title("Final Extended Diagram", fontsize=14, weight='bold')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plots['extended_diagram'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Generate labeled diagram
    fig, ax = plt.subplots(figsize=(20, 16))
    for _, row in df.iterrows():
        from_node = str(row['FromNodeId_Logical'])
        to_node = str(row['ToNodeId_Logical'])
        conductor = row.get('PhaseConductorId', '')
        length = row.get('SectionLength_MUL', 0)
        description = row.get('Description', '')
        capacitor = row.get('Capacitor_kVAR', 0)

        actual_to_node = to_node
        if to_node not in node_positions:
            matching_nodes = [n for n in node_positions if n.startswith(to_node)]
            if matching_nodes:
                actual_to_node = matching_nodes[0]

        if from_node not in node_positions or actual_to_node not in node_positions:
            continue

        x1, y1 = node_positions[from_node]
        x2, y2 = node_positions[actual_to_node]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

        label_x, label_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = abs(x2 - x1), abs(y2 - y1)

        if dy > dx:
            ax.text(label_x + 1.5, label_y, str(conductor), fontsize=6, color='blue', ha='center', weight='bold', rotation=90)
            ax.text(label_x - 1.5, label_y, f"{length:.1f}m", fontsize=6, color='green', ha='center', rotation=90)
        else:
            ax.text(label_x, label_y + 1.5, str(conductor), fontsize=6, color='blue', ha='center', weight='bold')
            ax.text(label_x, label_y - 1.5, f"{length:.1f}m", fontsize=6, color='green', ha='center')

        if isinstance(description, str) and description.strip():
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            ax.plot(x2, y2, marker=(3, 0, angle + 90), markersize=12,
                    color='red', markerfacecolor='orange', markeredgewidth=2)
            ax.text(x2 + 1, y2 + 1, description.strip(), fontsize=6, color='red', weight='bold')

        if capacitor > 0:
            ax.plot(x1, y1, marker='s', markersize=8, color='purple', markerfacecolor='yellow', markeredgewidth=2)
            ax.text(x1 + 1, y1 - 2, f"{capacitor:.0f}kVAR", fontsize=6, color='purple', weight='bold')

    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', color='lightblue', markersize=6, markeredgecolor='black')
        ax.text(x, y - 2, node.split('_')[0], fontsize=8, ha='center', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title("Complete Single-Line Diagram with Labels & Components", fontsize=14, weight='bold')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png',knockout, dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plots['final_diagram'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots

def process_mdb_file(mdb_filepath):
    """Process the uploaded MDB file and return processed DataFrame and Excel file path"""
    try:
        sai_control_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        result = subprocess.run(['mdb-export', mdb_filepath, 'SAI_Control'], 
                              stdout open(sai_control_csv, 'w', encoding='utf-8'), 
                              stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to export SAI_CONTROL table: {result.stderr}")
        
        df_sai = pd.read_csv(sai_control_csv)
        length_unit = df_sai.iloc[0, 2].strip() if not df_sai.empty else ''

        sheet_names = ['InstSection', 'InstPrimaryTransformers', 'Node']
        sheet_csv_paths = {}

        for sheet in sheet_names:
            temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
            result = subprocess.run(['mdb-export', mdb_filepath, sheet], 
                                  stdout=open(temp_csv, 'w', encoding='utf-8'),
                                  stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to export {sheet} table: {result.stderr}")
            
            sheet_csv_paths[sheet] = temp_csv

        df_section = pd.read_csv(sheet_csv_paths['InstSection'])
        df_transformers = pd.read_csv(sheet_csv_paths['InstPrimaryTransformers'])
        df_node = pd.read_csv(sheet_csv_paths['Node'])

        columns_to_keep = ['FeederId', 'FromNodeId', 'ToNodeId', 'Description', 'PhaseConductorId', 'SectionLength_MUL']
        df_section = df_section[columns_to_keep].copy()
        df_section['SectionLength_MUL'] = df_section['SectionLength_MUL'].apply(lambda x: float(f"{x:.6f}"))

        if length_unit == 'English2':
            df_section['SectionLength_MUL'] = df_section['SectionLength_MUL'] / 3.2808

        df_section['SortKey'] = df_section.apply(sort_key, axis=1)
        df_section = df_section.sort_values(by='SortKey').drop(columns='SortKey').reset_index(drop=True)

        node_dict = df_node.set_index(df_node.columns[0])[[df_node.columns[1], df_node.columns[2]]].to_dict('index')

        def utm_to_latlon(x, y):
            try:
                lat, lon = utm.to_latlon(float(x), float(y), 43, 'N')
                return round(lat, 6), round(lon, 6)
            except:
                return '', ''

        def get_latlon(node_id):
            if node_id in node_dict:
                x, y = node_dict[node_id][df_node.columns[1]], node_dict[node_id][df_node.columns[2]]
                if length_unit == 'English2':
                    x = float(x) / 3.2808
                    y = float(y) / 3.2808
                return utm_to_latlon(x, y)
            else:
                return '', ''

        from_x, from_y = zip(*df_section['FromNodeId'].map(get_latlon))
        to_x, to_y = zip(*df_section['ToNodeId'].map(get_latlon))

        from_idx = df_section.columns.get_loc('FromNodeId') + 1
        to_idx = df_section.columns.get_loc('ToNodeId') + 3

        df_section.insert(from_idx, 'From_X', from_x)
        df_section.insert(from_idx + 1, 'From_Y', from_y)
        df_section.insert(to_idx, 'To_X', to_x)
        df_section.insert(to_idx + 1, 'To_Y', to_y)

        transformer_map = dict(zip(df_transformers['SectionId'], df_transformers['TransformerType']))
        df_section['Description'] = df_section['Description'].apply(lambda val: transformer_map.get(val, val))
        df_section['Description'] = df_section['Description'].apply(
            lambda x: x if isinstance(x, str) and 'KVA' in x.upper() else ''
        )

        try:
            capacitor_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
            result = subprocess.run(['mdb-export', mdb_filepath, 'InstCapacitors'], 
                                  stdout=open(capacitor_csv, 'w', encoding='utf-8'),
                                  stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                df_capacitors = pd.read_csv(capacitor_csv)
                inst_section_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
                subprocess.run(['mdb-export', mdb_filepath, 'InstSection'], stdout=open(inst_section_csv, 'w', encoding='utf-8'))
                df_inst_section = pd.read_csv(inst_section_csv)

                df_capacitors = df_capacitors[df_capacitors.iloc[:, 8].notna()].copy()
                df_capacitors['CapacitorValue'] = df_capacitors.iloc[:, 8].astype(float) * 3

                capacitor_node_map = {}
                for _, row in df_capacitors.iterrows():
                    capacitor_section_id = row.iloc[0]
                    capacitor_value = row['CapacitorValue']
                    match_row = df_inst_section[df_inst_section.iloc[:, 0] == capacitor_section_id]
                    if not match_row.empty:
                        from_node_id = match_row.iloc[0, 2]
                        capacitor_node_map[from_node_id] = capacitor_value

                df_section['Capacitor_kVAR'] = 0.0
                placed_nodes = set()
                for node_id, cap_value in capacitor_node_map.items():
                    matching_indices = df_section[df_section['FromNodeId'] == node_id].index
                    if not matching_indices.empty:
                        first_idx = matching_indices[0]
                        if node_id not in placed_nodes:
                            df_section.at[first_idx, 'Capacitor_kVAR'] = cap_value
                            placed_nodes.add(node_id)
            else:
                df_section['Capacitor_kVAR'] = 0.0
        except:
            df_section['Capacitor_kVAR'] = 0.0

        df_section = add_logical_numbering(df_section)
        df_section = drop_latlon_columns(df_section)
        df_section = shift_transformers_to_parent(df_section)
        df_section = remove_redundant_gopher_rows(df_section)
        df_section = shift_description_chain(df_section)
        df_section = remove_lnode_rows(df_section)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
        df_section.to_excel(output_path, index=False, engine='openpyxl')

        for temp_file in [sai_control_csv] + list(sheet_csv_paths.values()):
            try:
                os.unlink(temp_file)
            except:
                pass

        return df_section, output_path

    except Exception as e:
        logger.error(f"Error processing MDB file: {str(e)}")
        raise

def process_xlsx_file(xlsx_filepath):
    """Process the uploaded XLSX file and return processed DataFrame"""
    try:
        df_section = pd.read_excel(xlsx_filepath)
        
        required_columns = ['FeederId', 'FromNodeId', 'ToNodeId', 'Description', 'PhaseConductorId', 'SectionLength_MUL']
        missing_columns = [col for col in required_columns if col not in df_section.columns]
        
        if missing_columns:
            df_section = add_logical_numbering(df_section)
            df_section = drop_latlon_columns(df_section)
        
        df_section = shift_transformers_to_parent(df_section)
        df_section = remove_redundant_gopher_rows(df_section)
        df_section = shift_description_chain(df_section)
        df_section = remove_lnode_rows(df_section)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
        df_section.to_excel(output_path, index=False, engine='openpyxl')

        return df_section, output_path

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
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext == 'mdb':
                df_processed, excel_output_path = process_mdb_file(filepath)
            elif file_ext == 'xlsx':
                df_processed, excel_output_path = process_xlsx_file(filepath)
            
            plots = generate_diagram_plots(df_processed)
            plot_paths = {}
            for plot_name, plot_data in plots.items():
                plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{plot_name}_{filename.rsplit('.', 1)[0]}.png")
                with open(plot_path, 'wb') as f:
                    f.write(base64.b64decode(plot_data))
                plot_paths[plot_name] = plot_path
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.write(excel_output_path, f"processed_{filename.rsplit('.', 1)[0]}.xlsx")
                zip_file.write(plot_paths['final_diagram'], "final_diagram.png")
            
            zip_buffer.seek(0)
            
            os.unlink(filepath)
            os.unlink(excel_output_path)
            for plot_path in plot_paths.values():
                try:
                    os.unlink(plot_path)
                except:
                    pass
            
            zip_path = tempfile.NamedTemporaryFile(delete=False, suffix='.zip').name
            with open(zip_path, 'wb') as f:
                f.write(zip_buffer.getvalue())
            
            return send_file(zip_path, as_attachment=True, 
                           download_name=f"SLD_Results_{filename.rsplit('.', 1)[0]}.zip",
                           mimetype='application/zip')
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            logger.error(f"Error processing file: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a .mdb or .xlsx file.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

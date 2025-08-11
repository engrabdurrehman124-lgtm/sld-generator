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
app.config['UPLOAD_FOLDER'] = 'uploads'

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

    # Use appropriate column names based on whether they exist
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

def generate_diagram_plots(df):
    """Generate all the diagram plots and return them as base64 encoded images"""
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
    
    # Draw edges
    for from_node, to_node in edges:
        if from_node in node_positions and to_node in node_positions:
            x1, y1 = node_positions[from_node]
            x2, y2 = node_positions[to_node]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Draw nodes
    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', color='lightblue', markersize=6, markeredgecolor='black')
        ax.text(x, y - 2, node.split('_')[0], fontsize=8, ha='center', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title("Base Logical Single-Line Diagram", fontsize=14, weight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plots['base_diagram'] = base64.b64encode(buffer.getvalue()).decode()
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

        # Handle duplicated node names
        actual_to_node = to_node
        if to_node not in node_positions:
            matching_nodes = [n for n in node_positions if n.startswith(to_node)]
            if matching_nodes:
                actual_to_node = matching_nodes[0]

        if from_node not in node_positions or actual_to_node not in node_positions:
            continue

        x1, y1 = node_positions[from_node]
        x2, y2 = node_positions[actual_to_node]

        # Draw line
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

        # Add labels
        label_x, label_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = abs(x2 - x1), abs(y2 - y1)

        if dy > dx:
            # Vertical line
            ax.text(label_x + 1.5, label_y, str(conductor), fontsize=6, color='blue', ha='center', weight='bold', rotation=90)
            ax.text(label_x - 1.5, label_y, f"{length:.1f}m", fontsize=6, color='green', ha='center', rotation=90)
        else:
            # Horizontal or diagonal
            ax.text(label_x, label_y + 1.5, str(conductor), fontsize=6, color='blue', ha='center', weight='bold')
            ax.text(label_x, label_y - 1.5, f"{length:.1f}m", fontsize=6, color='green', ha='center')

        # Add transformer symbol
        if isinstance(description, str) and description.strip():
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            ax.plot(x2, y2, marker=(3, 0, angle + 90), markersize=12,
                    color='red', markerfacecolor='orange', markeredgewidth=2)
            ax.text(x2 + 1, y2 + 1, description.strip(), fontsize=6, color='red', weight='bold')

        # Add capacitor symbol
        if capacitor > 0:
            ax.plot(x1, y1, marker='s', markersize=8, color='purple', markerfacecolor='yellow', markeredgewidth=2)
            ax.text(x1 + 1, y1 - 2, f"{capacitor:.0f}kVAR", fontsize=6, color='purple', weight='bold')

    # Draw nodes
    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', color='lightblue', markersize=6, markeredgecolor='black')
        ax.text(x, y - 2, node.split('_')[0], fontsize=8, ha='center', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title("Complete Single-Line Diagram with Labels & Components", fontsize=14, weight='bold')
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plots['final_diagram'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots

def process_mdb_file(mdb_filepath):
    """Process the uploaded MDB file and return processed DataFrame and Excel file path"""
    try:
        # Read SAI_Control First
        sai_control_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        result = subprocess.run(['mdb-export', mdb_filepath, 'SAI_Control'], 
                              stdout=open(sai_control_csv, 'w'), 
                              stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to export SAI_Control table: {result.stderr}")
        
        df_sai = pd.read_csv(sai_control_csv)
        length_unit = df_sai.iloc[0, 2].strip() if not df_sai.empty else ''

        # Extract Required Tables
        sheet_names = ['InstSection', 'InstPrimaryTransformers', 'Node']
        sheet_csv_paths = {}

        for sheet in sheet_names:
            temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
            result = subprocess.run(['mdb-export', mdb_filepath, sheet], 
                                  stdout=open(temp_csv, 'w'),
                                  stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to export {sheet} table: {result.stderr}")
            
            sheet_csv_paths[sheet] = temp_csv

        # Load DataFrames
        df_section = pd.read_csv(sheet_csv_paths['InstSection'])
        df_transformers = pd.read_csv(sheet_csv_paths['InstPrimaryTransformers'])
        df_node = pd.read_csv(sheet_csv_paths['Node'])

        # Preprocess Section Data
        columns_to_keep = ['FeederId', 'FromNodeId', 'ToNodeId', 'Description', 'PhaseConductorId', 'SectionLength_MUL']
        df_section = df_section[columns_to_keep].copy()
        df_section['SectionLength_MUL'] = df_section['SectionLength_MUL'].apply(lambda x: float(f"{x:.6f}"))

        # Apply unit conversion
        if length_unit == 'English2':
            df_section['SectionLength_MUL'] = df_section['SectionLength_MUL'] / 3.2808

        # Custom sorting
        df_section['SortKey'] = df_section.apply(sort_key, axis=1)
        df_section = df_section.sort_values(by='SortKey').drop(columns='SortKey').reset_index(drop=True)

        # Map UTM X/Y to Lat/Lon
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

        # Insert Lat/Lon Columns
        from_idx = df_section.columns.get_loc('FromNodeId') + 1
        to_idx = df_section.columns.get_loc('ToNodeId') + 3

        df_section.insert(from_idx, 'From_X', from_x)
        df_section.insert(from_idx + 1, 'From_Y', from_y)
        df_section.insert(to_idx, 'To_X', to_x)
        df_section.insert(to_idx + 1, 'To_Y', to_y)

        # Replace Description with TransformerType
        transformer_map = dict(zip(df_transformers['SectionId'], df_transformers['TransformerType']))
        df_section['Description'] = df_section['Description'].apply(lambda val: transformer_map.get(val, val))
        df_section['Description'] = df_section['Description'].apply(
            lambda x: x if isinstance(x, str) and 'KVA' in x.upper() else ''
        )

        # Extract and Map Capacitor Data
        try:
            capacitor_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
            result = subprocess.run(['mdb-export', mdb_filepath, 'InstCapacitors'], 
                                  stdout=open(capacitor_csv, 'w'),
                                  stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                df_capacitors = pd.read_csv(capacitor_csv)
                inst_section_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
                subprocess.run(['mdb-export', mdb_filepath, 'InstSection'], stdout=open(inst_section_csv, 'w'))
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

        # Apply all processing steps
        df_section = add_logical_numbering(df_section)
        df_section = drop_latlon_columns(df_section)
        df_section = shift_transformers_to_parent(df_section)
        df_section = remove_redundant_gopher_rows(df_section)
        df_section = shift_description_chain(df_section)
        df_section = remove_lnode_rows(df_section)

        # Export to Excel
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
        df_section.to_excel(output_path, index=False, engine='openpyxl')

        # Clean up temporary files
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
        
        # Verify required columns exist
        required_columns = ['FeederId', 'FromNodeId_Logical', 'ToNodeId_Logical', 'Description', 'PhaseConductorId', 'SectionLength_MUL']
        missing_columns = [col for col in required_columns if col not in df_section.columns]
        
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")

        # Apply processing steps (some may not be needed for pre-processed XLSX)
        df_section = shift_transformers_to_parent(df_section)
        df_section = remove_redundant_gopher_rows(df_section)
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
                # Process MDB file
                df_processed, excel_output_path = process_mdb_file(filepath)
            elif file_ext == 'xlsx':
                # Process XLSX file
                df_processed = process_xlsx_file(filepath)
                # Create Excel output for consistency
                excel_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
                df_processed.to_excel(excel_output_path, index=False, engine='openpyxl')
            
            # Generate diagrams
            plots = generate_diagram_plots(df_processed)
            
            # Create a zip file with Excel and diagrams
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add Excel file
                zip_file.write(excel_output_path, f"processed_{filename.rsplit('.', 1)[0]}.xlsx")
                
                # Add diagram images
                for plot_name, plot_data in plots.items():
                    img_buffer = BytesIO(base64.b64decode(plot_data))
                    zip_file.writestr(f"{plot_name}.png", img_buffer.getvalue())
            
            zip_buffer.seek(0)
            
            # Clean up uploaded file and temporary Excel file
            os.unlink(filepath)
            os.unlink(excel_output_path)
            
            # Store plots in session for display
            session['plots'] = plots
            session['processed_filename'] = filename.rsplit('.', 1)[0]
            
            # Create a temporary file for the zip
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

@app.route('/view_results')
def view_results():
    if not check_passcode():
        return redirect(url_for('login'))
    
    plots = session.get('plots', {})
    filename = session.get('processed_filename', 'Unknown')
    
    if not plots:
        flash('No results to display. Please upload a file first.')
        return redirect(url_for('index'))
    
    return render_template('results.html', plots=plots, filename=filename)
if __name__ == '__main__':
    port = int(os.environ.get('Key', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
    

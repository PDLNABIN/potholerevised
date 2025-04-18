# --- Imports ---
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd # Added for PyDeck
import laspy
import os
import plotly.graph_objects as go
import pydeck as pdk # Added for PyDeck
import tempfile
from scipy.spatial import KDTree
import base64
import random 
from pyproj import Transformer

# UTM Zone example: Replace zone and hemisphere based on your LAS data
transformer = Transformer.from_crs("epsg:32645", "epsg:4326", always_xy=True)  # UTM zone 45N -> WGS84

def convert_to_lonlat(x, y):
    """Convert X/Y coordinates to approximate lon/lat using pyproj"""
    lon, lat = transformer.transform(x, y)
    return lon, lat


# Try to import leafmap for visualization options
try:
    import leafmap
    # import open3d as o3d # No longer explicitly needed for backend selection
    LEAFMAP_AVAILABLE = True
except ImportError:
    LEAFMAP_AVAILABLE = False
    # Display warning in the main area if leafmap is selected but not available later
    # st.sidebar.warning("Leafmap library not found. Leafmap visualization disabled.")


# Import model definition
try:
    from model_definition import DGCNN, knn, get_graph_feature
except ImportError:
    st.error("Error: Could not import model_definition.py")
    st.stop()

# --- Constants and Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
K_NEIGHBORS_FIXED = 20
PLOTLY_MAX_POINTS = 500_000 # Max points for Plotly before downsampling
PYDECK_WARNING_THRESHOLD = 1_000_000 # Warn if points exceed this for PyDeck

# --- Model Loading Function ---
@st.cache_resource
def load_model(model_path, k_neighbors=K_NEIGHBORS_FIXED):
    """Loads the pre-trained DGCNN model."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model checkpoint file not found at '{model_path}'.")
        return None

    try:
        model = DGCNN(input_channels=6, output_channels=2, k=k_neighbors).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None

# --- Inference Function ---
def run_inference(model, las_data, device, num_points_per_region=1024, sample_fraction=0.1):
    """Perform inference on LAS data using KDTree optimization"""
    x, y, z = las_data.x, las_data.y, las_data.z
    num_all_points = len(x)

    if num_all_points == 0:
        st.warning("No points found in the LAS file.")
        return None, None, None, None, None

    points_xyz = np.column_stack((x, y, z))

    # Handle RGB data (normalize 0-255 for display, 0-1 for features)
    default_rgb_val = 128
    has_rgb = False
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue') and \
       len(las_data.red) == num_all_points:
        r_orig, g_orig, b_orig = las_data.red, las_data.green, las_data.blue
        max_val = max(np.max(r_orig), np.max(g_orig), np.max(b_orig), 1)
        if max_val > 255: # Assume 16-bit
            r_display = (r_orig / 65535.0 * 255.0).astype(np.uint8)
            g_display = (g_orig / 65535.0 * 255.0).astype(np.uint8)
            b_display = (b_orig / 65535.0 * 255.0).astype(np.uint8)
            r_feat = (r_orig / 65535.0).astype(np.float32)
            g_feat = (g_orig / 65535.0).astype(np.float32)
            b_feat = (b_orig / 65535.0).astype(np.float32)
        else: # Assume 8-bit
            r_display = r_orig.astype(np.uint8)
            g_display = g_orig.astype(np.uint8)
            b_display = b_orig.astype(np.uint8)
            r_feat = (r_orig / 255.0).astype(np.float32)
            g_feat = (g_orig / 255.0).astype(np.float32)
            b_feat = (b_orig / 255.0).astype(np.float32)
        has_rgb = True
    else:
        # Use default gray if no RGB
        r_display = g_display = b_display = np.full(num_all_points, default_rgb_val, dtype=np.uint8)
        r_feat = g_feat = b_feat = np.full(num_all_points, 0.5, dtype=np.float32)

    # Prepare features for the model (X, Y, Z, R, G, B)
    all_points_features = np.column_stack((x, y, z, r_feat, g_feat, b_feat))
    predictions = np.zeros(num_all_points, dtype=int) # Initialize predictions as 0 (non-pothole)

    # Build KDTree
    st.info("Building KDTree...")
    try:
        kdtree = KDTree(points_xyz)
    except Exception as e:
        st.error(f"Error building KDTree: {e}")
        return None, None, None, None, None

    # Sample points for inference to speed up
    sample_size = max(min(int(num_all_points * sample_fraction), num_all_points), 1)
    sampled_indices = np.random.choice(num_all_points, sample_size, replace=False)

    st.info(f"Running inference on {sample_size} sampled regions...")
    progress_bar = st.progress(0)

    # Run inference in batches/regions
    with torch.no_grad():
        for i, idx in enumerate(sampled_indices):
            center_point_xyz = points_xyz[idx]

            # Query KDTree for nearest neighbors for the current region
            k_query = min(num_points_per_region + 1, num_all_points) # +1 to exclude center later
            try:
                _, nearest_indices = kdtree.query(center_point_xyz, k=k_query)
            except Exception as e:
                # Handle potential query errors gracefully
                continue

            if isinstance(nearest_indices, (int, np.integer)): # Handle case of single neighbor
                nearest_indices = [nearest_indices]

            # Get indices of neighbors (excluding the center point itself)
            valid_neighbor_indices = [ni for ni in nearest_indices if ni != idx][:num_points_per_region]

            if not valid_neighbor_indices:
                continue # Skip if no valid neighbors found

            # Get features for the model input
            region_features_model = all_points_features[valid_neighbor_indices]

            # Center coordinates relative to the region's center point
            centered_region_features_model = region_features_model.copy()
            centered_region_features_model[:, :3] = region_features_model[:, :3] - center_point_xyz

            # Ensure the region has exactly num_points_per_region (pad if needed)
            current_region_size = centered_region_features_model.shape[0]
            if current_region_size < num_points_per_region:
                num_to_pad = num_points_per_region - current_region_size
                # Repeat the last point's features for padding
                padding = np.repeat(centered_region_features_model[-1:], num_to_pad, axis=0)
                centered_region_features_model = np.vstack((centered_region_features_model, padding))

            # Prepare input tensor for the model
            inputs = torch.FloatTensor(centered_region_features_model).unsqueeze(0).permute(0, 2, 1).to(device) # [B=1, F=6, N=num_region_points]

            # Forward pass through the model
            outputs = model(inputs)
            _, predicted_label = outputs.max(1) # Get the class index with the highest score

            # Mark points in the original cloud as pothole (class 1) if the region is predicted as such
            if predicted_label.item() == 1:
                # Ensure indices are within bounds before assignment
                valid_indices_mask = np.array(valid_neighbor_indices) < len(predictions)
                valid_indices_to_assign = np.array(valid_neighbor_indices)[valid_indices_mask]
                predictions[valid_indices_to_assign] = 1 # Mark as pothole (class 1)

            # Update progress bar
            progress_bar.progress((i + 1) / sample_size)

    progress_bar.empty() # Remove progress bar
    st.success("Inference complete.")
    num_potholes = np.sum(predictions == 1) # Count points classified as 1
    st.info(f"Found {num_potholes:,} potential pothole points (Class 1) out of {num_all_points:,} total points.")

    # Return coordinates, predictions (0 or 1), and original 8-bit display colors
    return points_xyz, predictions, r_display, g_display, b_display

# --- Leafmap Visualization Function ---
def visualize_with_leafmap(points_xyz, r_disp, g_disp, b_disp, predictions=None,
                           backend="pyvista", point_size=1.0, color_by_classification=False):
    """Create visualization with leafmap, coloring points based on selection."""
    if not LEAFMAP_AVAILABLE:
        st.error("Leafmap visualization requires leafmap package to be installed.")
        return None

    num_points = len(points_xyz)
    if num_points == 0:
        st.warning("No points to visualize.")
        return None

    # Create a temporary LAS file to pass to leafmap
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Use point format 3 (includes RGB)
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)

        # Set coordinates
        las.x = points_xyz[:, 0]
        las.y = points_xyz[:, 1]
        las.z = points_xyz[:, 2]

        # --- Color Logic ---
        if color_by_classification and predictions is not None:
            st.info("Leafmap: Coloring points - Red=Pothole(1), Blue=Non-Pothole(0)")
            pothole_color_rgb = np.array([255, 0, 0], dtype=np.uint8)  # Red
            non_pothole_color_rgb = np.array([0, 0, 255], dtype=np.uint8) # Blue

            vis_r = np.zeros(num_points, dtype=np.uint8)
            vis_g = np.zeros(num_points, dtype=np.uint8)
            vis_b = np.zeros(num_points, dtype=np.uint8)

            pothole_indices = np.where(predictions == 1)[0]
            non_pothole_indices = np.where(predictions == 0)[0] # Explicitly check for 0

            vis_r[pothole_indices], vis_g[pothole_indices], vis_b[pothole_indices] = pothole_color_rgb
            vis_r[non_pothole_indices], vis_g[non_pothole_indices], vis_b[non_pothole_indices] = non_pothole_color_rgb

            las.red = (vis_r.astype(np.uint16) * 257)
            las.green = (vis_g.astype(np.uint16) * 257)
            las.blue = (vis_b.astype(np.uint16) * 257)

        else:
            st.info("Leafmap: Coloring points using original RGB values.")
            las.red = r_disp.astype(np.uint16) * 257
            las.green = g_disp.astype(np.uint16) * 257
            las.blue = b_disp.astype(np.uint16) * 257

        # Set classification field (useful for export/analysis)
        if predictions is not None:
            classification = np.ones(len(predictions), dtype=np.uint8) # Default class 1
            classification[predictions == 1] = 7  # Mark potholes (class 1) as LAS class 7
            las.classification = classification
        else:
             las.classification = np.ones(num_points, dtype=np.uint8) # Default if no predictions

        # Write to temporary file
        las.write(tmp_path)

        # Generate visualization with specified backend, using the embedded RGB
        st.info(f"Generating Leafmap HTML with {backend} backend...")
        html = leafmap.view_lidar(
            tmp_path,
            background="white",
            backend=backend,
            return_as="html",
            point_size=point_size,
            width=800, # Adjust as needed
            height=600, # Adjust as needed
        )
        return html

    except Exception as e:
        st.error(f"Error creating Leafmap visualization: {e}")
        st.exception(e) # Print full traceback for debugging
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file {tmp_path}: {e}")

# --- Plotly Visualization Function ---
def visualize_with_plotly(points_xyz, r_disp, g_disp, b_disp, predictions=None,
                          point_size=1.5, color_by_classification=False, max_points=PLOTLY_MAX_POINTS):
    """Create 3D scatter plot visualization with Plotly."""
    num_points = len(points_xyz)
    if num_points == 0:
        st.warning("No points to visualize.")
        return None

    indices = np.arange(num_points)
    # Downsample if too many points for Plotly performance
    if num_points > max_points:
        st.warning(f"Plotly: Displaying a random sample of {max_points:,} points (out of {num_points:,}) for performance.")
        indices = np.random.choice(indices, max_points, replace=False)

    sampled_points_xyz = points_xyz[indices]
    sampled_r = r_disp[indices]
    sampled_g = g_disp[indices]
    sampled_b = b_disp[indices]
    sampled_preds = predictions[indices] if predictions is not None else None

    # --- Color Logic ---
    colors = []
    if color_by_classification and sampled_preds is not None:
        st.info("Plotly: Coloring points - Red=Pothole(1), Blue=Non-Pothole(0)")
        pothole_color = 'red'
        non_pothole_color = 'blue'
        for pred in sampled_preds:
            colors.append(pothole_color if pred == 1 else non_pothole_color)
    else:
        st.info("Plotly: Coloring points using original RGB values.")
        colors = [f'rgb({r},{g},{b})' for r, g, b in zip(sampled_r, sampled_g, sampled_b)]

    # Create Plotly figure
    try:
        st.info("Generating Plotly 3D Scatter Plot...")
        fig = go.Figure(data=[go.Scatter3d(
            x=sampled_points_xyz[:, 0],
            y=sampled_points_xyz[:, 1],
            z=sampled_points_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors, # Assign prepared colors
                opacity=0.8
            )
        )])

        # Update layout for better appearance
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30), # Reduce margins
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate',
                aspectmode='data' # Maintain aspect ratio
            ),
            title="Plotly 3D Point Cloud Visualization"
        )
        return fig

    except Exception as e:
        st.error(f"Error creating Plotly visualization: {e}")
        st.exception(e)
        return None

# --- PyDeck Visualization Function (Corrected ViewState) ---
def visualize_with_pydeck(points_xyz, r_disp, g_disp, b_disp, predictions=None,
                          point_size=1.5, color_by_classification=False):
    """Create 3D scatter plot visualization with PyDeck."""
    num_points = len(points_xyz)
    if num_points == 0:
        st.warning("No points to visualize.")
        return None

    # Create DataFrame
    df = pd.DataFrame({
        'x': points_xyz[:, 0],
        'y': points_xyz[:, 1],
        'z': points_xyz[:, 2],
        'r': r_disp,
        'g': g_disp,
        'b': b_disp,
        # Ensure prediction column exists even if predictions are None
        'prediction': predictions if predictions is not None else np.zeros(num_points, dtype=int)
    })

    # --- Color Logic (RGBA for PyDeck) ---
    # User confirmed: 0=non-pothole (Blue), 1=pothole (Red)
    alpha = 180 # Opacity
    if color_by_classification and predictions is not None:
        st.info("PyDeck: Coloring points - Red=Pothole(1), Blue=Non-Pothole(0)")
        pothole_color = [255, 0, 0, alpha]
        non_pothole_color = [0, 0, 255, alpha]
        # Apply color based on the 'prediction' column
        df['color'] = df['prediction'].apply(lambda p: pothole_color if p == 1 else non_pothole_color)
    else:
        st.info("PyDeck: Coloring points using original RGB values.")
        df['color'] = df.apply(lambda row: [int(row['r']), int(row['g']), int(row['b']), alpha], axis=1) # Ensure int for color

    # --- Calculate View State based on Data Bounds ---
    try:
        min_x, max_x = df['x'].min(), df['x'].max()
        min_y, max_y = df['y'].min(), df['y'].max()
        min_z, max_z = df['z'].min(), df['z'].max()

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # Start with a reasonable zoom level, adjust if needed based on data scale
        initial_zoom = 5 # Lower zoom level to start further out

        # Convert X/Y to Lon/Lat
        center_lon, center_lat = convert_to_lonlat(center_x, center_y)

        view_state = pdk.ViewState(latitude=center_lat,longitude=center_lon,zoom=initial_zoom,pitch=45,bearing=0)

        st.caption(f"Calculated View Center: X={center_x:.2f}, Y={center_y:.2f}, Z={center_z:.2f}")
    except Exception as e:
        st.error(f"Could not calculate view state bounds: {e}")
        # Fallback view state if bounds calculation fails
        view_state = pdk.ViewState(target=[0, 0, 0], zoom=1, pitch=45)


    # Define PyDeck layer
    layer = pdk.Layer(
        'PointCloudLayer',
        data=df,
        get_position='[x, y, z]',
        get_color='color',
        get_normal=[0, 0, 1], # Optional: affects lighting if used
        auto_highlight=True,
        pickable=True,
        point_size=point_size
    )

    # Tooltip (using prediction=0/1 based on user confirmation)
    tooltip = {
        "html": "<b>X:</b> {x:.2f}<br/><b>Y:</b> {y:.2f}<br/><b>Z:</b> {z:.2f}<br/><b>Class:</b> {prediction} (1=Pothole)",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # Create Deck object
    try:
        st.info("Generating PyDeck Point Cloud...")
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None, # Use None for no basemap
            tooltip=tooltip
        )
        return deck
    except Exception as e:
        st.error(f"Error creating PyDeck visualization: {e}")
        st.exception(e)
        return None


# --- Export Function ---
def export_las_with_predictions(las_data, predictions, output_path="pothole_results.las"):
    """Export LAS file with pothole classifications"""
    try:
        # Create a new LAS object, preserving header info but ensuring classification exists
        header = las_data.header
        point_format_id = header.point_format.id
        if 'classification' not in header.point_format.dimension_names:
            st.warning(f"Original Point Format ({point_format_id}) lacked 'classification'. Upgrading format for export.")
            point_format_id = 3 if 'red' in header.point_format.dimension_names else 1

        new_las = laspy.create(point_format=point_format_id, file_version=header.version)
        for dim_name in header.point_format.dimension_names:
             if dim_name != 'classification' and hasattr(las_data, dim_name):
                 try:
                     setattr(new_las, dim_name, getattr(las_data, dim_name))
                 except Exception:
                     pass # Skip if dimension cannot be copied

        # Explicitly copy XYZ
        new_las.x = las_data.x
        new_las.y = las_data.y
        new_las.z = las_data.z

        # Copy RGB if supported
        if 'red' in new_las.point_format.dimension_names and hasattr(las_data, 'red'):
            new_las.red = las_data.red
            new_las.green = las_data.green
            new_las.blue = las_data.blue

        # Add or update classification field (0=Non-Pothole -> LAS Class 1, 1=Pothole -> LAS Class 7)
        classification = np.ones(len(predictions), dtype=np.uint8) # Default LAS class 1 (Unclassified)
        classification[predictions == 1] = 7  # Mark predicted potholes (value 1) as LAS class 7
        new_las.classification = classification

        new_las.write(output_path)
        st.success(f"LAS file exported with classification to {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Error exporting LAS file: {e}")
        st.exception(e) # Show detailed error
        return None

# --- Download Link Helper ---
def get_download_link(file_path, link_text):
    """Create download link for a file"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except FileNotFoundError:
        st.error(f"Exported file not found at {file_path}")
        return "Export failed."
    except Exception as e:
         st.error(f"Error creating download link: {e}")
         return "Export failed."

# ==============================
# === Streamlit App UI Layout ===
# ==============================
st.set_page_config(layout="wide", page_title="3D Pothole Detection", initial_sidebar_state="expanded")

# --- Header ---
st.title("🛣️ 3D Pothole Detection & Visualization")
st.markdown("Upload a `.las` or `.laz` file, detect potholes using DGCNN, and visualize the results.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.info(f"Using device: **{str(device).upper()}**")

# Model/Inference Parameters
st.sidebar.subheader("Inference Settings")
num_region_points = st.sidebar.slider("Points per Region (Model Input)", 512, 2048, 1024, 128,
                                      help="Number of neighboring points considered for classifying each region.")
inference_sample_fraction = st.sidebar.slider("Inference Sample Fraction", 0.01, 1.0, 0.2, 0.01,
                                               help="Fraction of points used as centers for KDTree neighbor search during inference (higher=more thorough but slower).")

# Visualization Parameters
st.sidebar.subheader("Visualization Settings")
vis_options = ["Plotly 3D", "PyDeck"] # Default options
if LEAFMAP_AVAILABLE:
    vis_options.insert(0, "Leafmap") # Add Leafmap if available

default_vis_index = 0 # Default to Leafmap if available, else Plotly

vis_method = st.sidebar.selectbox("Visualization Method",
                                 vis_options,
                                 index=default_vis_index,
                                 help="Choose the tool for 3D visualization.")

# Shared Visualization Options
colorby_option = st.sidebar.radio(
    "Color By",
    ["Classification (Pothole=Red)", "Original RGB"],
    index=0, # Default to classification coloring
    help="How to color points: Red(1)/Blue(0) based on prediction, or using original file colors."
)
color_by_classification_flag = (colorby_option == "Classification (Pothole=Red)")


# Method-Specific Options
point_size = 1.5 # Default point size
if vis_method == "Leafmap":
    if LEAFMAP_AVAILABLE:
        # Removed "open3d" from the list
        leafmap_backend = st.sidebar.selectbox(
            "Leafmap Backend", ["pyvista", "ipygany", "panel"], index=0,
            help="Rendering engine for Leafmap (pyvista often works well)."
        )
        point_size = st.sidebar.slider( # Assign to generic point_size
            "Leafmap Point Size", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
            key="leafmap_ps", help="Size of points in the Leafmap visualization."
        )
    else:
        # Should not be reachable if Leafmap is selected, but good practice
        st.error("Leafmap selected but not available. Please install it.")


elif vis_method == "Plotly 3D":
    point_size = st.sidebar.slider( # Assign to generic point_size
        "Plotly Point Size", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
        key="plotly_ps", help="Size of points in the Plotly visualization."
    )
elif vis_method == "PyDeck":
    point_size = st.sidebar.slider( # Assign to generic point_size
        "PyDeck Point Size", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
        key="pydeck_ps", help="Size of points in the PyDeck visualization."
    )

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Main UI Tabs ---
tab_upload, tab_results = st.tabs(["📤 Upload & Process", "📊 Results & Visualization"])

# === Upload & Process Tab ===
with tab_upload:
    st.subheader("📂 Upload LAS/LAZ File")
    uploaded_file = st.file_uploader("Select file:", type=["las", "laz"], label_visibility="collapsed")

    if uploaded_file is not None and model is not None:
        st.write("---")
        st.info(f"Processing '{uploaded_file.name}'...")

        # Save uploaded file temporarily to read with laspy
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Reset session state for new file processing
            for key in ['points_xyz', 'predictions', 'r_display', 'g_display', 'b_display', 'las_data', 'filename', 'processed']:
                 if key in st.session_state:
                     del st.session_state[key]

            las_data = laspy.read(tmp_file_path)
            st.success(f"Successfully read file. Contains **{len(las_data.points):,}** points.")

            # Store raw las data and filename for later use (export, naming)
            st.session_state['las_data'] = las_data
            st.session_state['filename'] = uploaded_file.name

            # Run inference
            st.write("---")
            st.subheader("🧠 Running Pothole Detection")
            with st.spinner("Analyzing point cloud... This may take a while for large files."):
                 points_xyz, predictions, r_display, g_display, b_display = run_inference(
                    model, las_data, device,
                    num_points_per_region=num_region_points,
                    sample_fraction=inference_sample_fraction
                 )

                 if points_xyz is not None and predictions is not None:
                    # Store results in session state for the results tab
                    st.session_state['points_xyz'] = points_xyz
                    st.session_state['predictions'] = predictions # Contains 0s and 1s
                    st.session_state['r_display'] = r_display # Store 8-bit colors
                    st.session_state['g_display'] = g_display
                    st.session_state['b_display'] = b_display
                    st.session_state['processed'] = True # Flag indicating processing is done

                    # Display summary metrics after processing
                    st.success("✅ Processing complete! View results in the 'Results & Visualization' tab.")
                    num_potholes = np.sum(predictions == 1)
                    pothole_percentage = (num_potholes / len(predictions)) * 100 if len(predictions) > 0 else 0
                    st.subheader("📊 Quick Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Points Analyzed", f"{len(predictions):,}")
                    col2.metric("Detected Pothole Points (Class 1)", f"{num_potholes:,}")
                    col3.metric("Pothole Density", f"{pothole_percentage:.2f}%")
                 else:
                    st.error("⚠️ Inference failed or produced no results. Cannot proceed.")
                    st.session_state['processed'] = False

        except Exception as e:
            st.error(f"An error occurred while processing the file:")
            st.exception(e)
            st.session_state['processed'] = False
        finally:
            # Clean up the temporary file
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary file {tmp_file_path}: {e}")

    elif uploaded_file is not None and model is None:
        st.error("Model could not be loaded. Cannot process file. Check logs or model path.")
    else:
        st.info("👆 Upload a `.las` or `.laz` file to begin analysis.")

# === Results & Visualization Tab ===
with tab_results:
    st.subheader("📊 Visualization & Export")

    if st.session_state.get('processed', False): # Check if processing was successful
        # --- Export Section ---
        st.markdown("---")
        st.subheader("📥 Export Results")
        export_col1, export_col2 = st.columns([3, 1])
        with export_col1:
            st.write("Export the point cloud with classifications (Potholes marked as LAS Class 7).")
        with export_col2:
            # Generate unique output name based on input filename
            base_name = os.path.splitext(st.session_state.get('filename', 'results'))[0]
            output_filename = f"classified_{base_name}.las"
            export_button = st.button("Export Classified LAS", key="export_las", type="primary")

        if export_button:
            if 'las_data' in st.session_state and 'predictions' in st.session_state:
                with st.spinner(f"Exporting results to '{output_filename}'..."):
                    result_path = export_las_with_predictions(
                        st.session_state['las_data'],
                        st.session_state['predictions'], # Pass the 0/1 predictions
                        output_filename
                    )
                    if result_path and os.path.exists(result_path):
                        st.success(f"Results exported successfully!")
                        # Provide download link
                        st.markdown(get_download_link(result_path, f"💾 Download {output_filename}"), unsafe_allow_html=True)
                    else:
                        st.error("Export failed. Check logs or file permissions.")
            else:
                st.warning("No data available to export. Process a file first.")

        # --- Visualization Section ---
        st.markdown("---")
        st.subheader("👁️ Point Cloud Visualization")

        # Retrieve data from session state
        points_xyz = st.session_state.get('points_xyz')
        predictions = st.session_state.get('predictions')
        r_display = st.session_state.get('r_display')
        g_display = st.session_state.get('g_display')
        b_display = st.session_state.get('b_display')

        if points_xyz is not None and predictions is not None and r_display is not None:
            # Display general warning for large datasets
            if len(points_xyz) > PYDECK_WARNING_THRESHOLD and vis_method != "Leafmap": # Leafmap might handle larger data better via backends
                 st.warning(f"Large dataset detected ({len(points_xyz):,} points). {vis_method} visualization might be slow or unresponsive.")

            if vis_method == "Leafmap":
                if LEAFMAP_AVAILABLE:
                    st.markdown(f"Using **Leafmap** ({leafmap_backend} backend). Coloring by: **{colorby_option}**")
                    with st.spinner(f"Generating Leafmap visualization..."):
                        html_content = visualize_with_leafmap(
                            points_xyz, r_display, g_display, b_display,
                            predictions=predictions,
                            backend=leafmap_backend,
                            point_size=point_size,
                            color_by_classification=color_by_classification_flag
                        )
                        if html_content:
                            st.components.v1.html(html_content, height=700, scrolling=True)
                        else:
                            st.warning("Could not generate Leafmap visualization.")
                else:
                     st.error("Leafmap selected but library not found/installed.")

            elif vis_method == "Plotly 3D":
                st.markdown(f"Using **Plotly 3D**. Coloring by: **{colorby_option}**")
                with st.spinner("Generating Plotly visualization..."):
                    plotly_fig = visualize_with_plotly(
                        points_xyz, r_display, g_display, b_display,
                        predictions=predictions,
                        point_size=point_size, # Use shared point_size
                        color_by_classification=color_by_classification_flag,
                        max_points=PLOTLY_MAX_POINTS
                    )
                    if plotly_fig:
                        st.plotly_chart(plotly_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate Plotly visualization.")

            elif vis_method == "PyDeck":
                st.markdown(f"Using **PyDeck**. Coloring by: **{colorby_option}**")
                with st.spinner("Generating PyDeck visualization..."):
                    pydeck_obj = visualize_with_pydeck(
                        points_xyz, r_display, g_display, b_display,
                        predictions=predictions,
                        point_size=point_size, # Use shared point_size
                        color_by_classification=color_by_classification_flag
                    )
                    if pydeck_obj:
                        # Use container width might affect pydeck, test this
                        st.pydeck_chart(pydeck_obj)
                    else:
                        st.warning("Could not generate PyDeck visualization.")


            # Display pothole stats again for context
            st.markdown("---")
            st.subheader("📈 Detection Summary")
            num_potholes = np.sum(predictions == 1) # Count class 1
            total_points = len(predictions)
            pothole_percentage = (num_potholes / total_points * 100) if total_points > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Points Displayed", f"{len(points_xyz):,}")
            col2.metric("Pothole Points (Class 1)", f"{num_potholes:,}")
            col3.metric("Pothole Percentage", f"{pothole_percentage:.2f}%")

        else:
             st.warning("Result data missing from session state. Cannot visualize.")

    else:
        st.info("⬅️ Please upload and process a file on the 'Upload & Process' tab to view results.")

# --- Help Section in Sidebar ---
with st.sidebar.expander("❓ Need Help?"):
    st.markdown("""
    This app uses a Deep Graph CNN (DGCNN) model to detect potential potholes in LiDAR point cloud data (`.las` or `.laz` files). Potholes are classified as **Class 1**, non-potholes as **Class 0**.

    **Workflow:**
    1.  **Upload:** Select your LAS/LAZ file.
    2.  **Configure:** Adjust inference and visualization settings in the sidebar.
    3.  **Process:** The app reads the file, runs the DGCNN model, and classifies points.
    4.  **Visualize:** View results in the 'Results & Visualization' tab. Use 'Classification' coloring (Red=1, Blue=0) for clarity.
    5.  **Export:** Download a new LAS file where predicted potholes (Class 1) are marked with **LAS Class 7**.

    **Tips:**
    * **Performance:** Large files (> 1M points) can be slow. PyDeck and Plotly may downsample or become unresponsive. Leafmap/PyVista might handle larger clouds better.
    * **Coloring:** 'Classification (Pothole=Red)' uses Red for predicted Class 1 points, Blue for Class 0. 'Original RGB' uses file colors.
    * **Backends:** Leafmap requires `pip install leafmap pyvista vtk`. PyDeck uses `st.pydeck_chart`. Plotly uses `st.plotly_chart`.
    """)

st.sidebar.markdown("---")
st.sidebar.info("🚗 Drive Safer with Enhanced Road Analysis 🚗")
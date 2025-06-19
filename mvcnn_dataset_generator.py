import bpy
import math
import os
import json
import time
import sys
import argparse
from mathutils import Vector

# ========== Default Configuration ==========
# These defaults can be overridden by command line arguments
DEFAULTS = {
    "modelnet_dir": "ModelNet40",
    "output_dir": "mvcnn_dataset",
    "num_views": 12,
    "radius": 3.0,
    "image_size": 224,
    "max_models_per_class": None,
    "render_quality": 'DRAFT',
    "elevation_angles": [0, 30],
    "create_class_previews": True,
    # Demo options
    "demo_class": "airplane",
    "demo_speed": 2.0,
    "demo_pause": 1.0,
    "demo_output": "mvcnn_camera_demo.mp4"
}

# ========== Command Line Argument Parsing ==========
def parse_args():
    """Parse command line arguments passed after -- in Blender"""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
        
    parser = argparse.ArgumentParser(description='MVCNN Dataset Generator with Demo Mode')
    
    # Basic configuration
    parser.add_argument('--modelnet-dir', type=str, default=DEFAULTS["modelnet_dir"],
                        help=f'Path to ModelNet40 directory (default: {DEFAULTS["modelnet_dir"]})')
    parser.add_argument('--output-dir', type=str, default=DEFAULTS["output_dir"],
                        help=f'Output directory for rendered images (default: {DEFAULTS["output_dir"]})')
    parser.add_argument('--num-views', type=int, default=DEFAULTS["num_views"],
                        help=f'Number of camera views per elevation (default: {DEFAULTS["num_views"]})')
    parser.add_argument('--radius', type=float, default=DEFAULTS["radius"],
                        help=f'Camera distance from object (default: {DEFAULTS["radius"]})')
    parser.add_argument('--image-size', type=int, default=DEFAULTS["image_size"],
                        help=f'Output image size (default: {DEFAULTS["image_size"]})')
    parser.add_argument('--max-models', type=int, default=DEFAULTS["max_models_per_class"],
                        help='Maximum models per class (default: all)')
    parser.add_argument('--render-quality', type=str, default=DEFAULTS["render_quality"],
                        choices=['DRAFT', 'MEDIUM', 'HIGH'],
                        help=f'Rendering quality (default: {DEFAULTS["render_quality"]})')
    parser.add_argument('--elevation-angles', type=int, nargs='+', 
                        default=DEFAULTS["elevation_angles"],
                        help=f'Elevation angles in degrees (default: {DEFAULTS["elevation_angles"]})')
    parser.add_argument('--no-class-previews', action='store_false', dest='create_class_previews',
                        help='Disable creation of class preview images')
    
    # Demo mode options
    parser.add_argument('--demo', action='store_true',
                        help='Run in demonstration mode (creates animation instead of dataset)')
    parser.add_argument('--demo-class', type=str, default=DEFAULTS["demo_class"],
                        help=f'Class to use for demo (default: {DEFAULTS["demo_class"]})')
    parser.add_argument('--demo-speed', type=float, default=DEFAULTS["demo_speed"],
                        help=f'Seconds per camera position in demo (default: {DEFAULTS["demo_speed"]})')
    parser.add_argument('--demo-pause', type=float, default=DEFAULTS["demo_pause"],
                        help=f'Pause time at each position in demo (default: {DEFAULTS["demo_pause"]})')
    parser.add_argument('--demo-output', type=str, default=DEFAULTS["demo_output"],
                        help=f'Output video file for demo (default: {DEFAULTS["demo_output"]})')
    
    # Class-specific processing
    parser.add_argument('--only-class', type=str,
                        help='Only process this specific class')
    parser.add_argument('--list-classes', action='store_true',
                        help='List all available classes and exit')
                        
    args = parser.parse_args(argv)
    return args

# ========== Functions ==========
def validate_paths(args):
    """Validate paths and print detailed information"""
    print("\n=== PATH VALIDATION ===")
    
    # Check ModelNet40 directory
    if not os.path.exists(args.modelnet_dir):
        print(f"ERROR: ModelNet40 directory not found: {args.modelnet_dir}")
        return False
        
    # Check for class directories
    class_dirs = [d for d in os.listdir(args.modelnet_dir) 
                 if os.path.isdir(os.path.join(args.modelnet_dir, d))]
    if not class_dirs:
        print(f"ERROR: No class directories found in: {args.modelnet_dir}")
        return False
    
    print(f"Found {len(class_dirs)} class directories: {', '.join(class_dirs[:5])}...")
    
    # If only listing classes, print them all and exit
    if args.list_classes:
        print("\nAvailable classes:")
        for i, class_name in enumerate(sorted(class_dirs)):
            print(f"{i+1}. {class_name}")
        return False
    
    # Check for train/test structure in first class (or specified class)
    check_class = args.only_class if args.only_class in class_dirs else class_dirs[0]
    class_dir = os.path.join(args.modelnet_dir, check_class)
    
    if not (os.path.exists(os.path.join(class_dir, 'train')) and 
            os.path.exists(os.path.join(class_dir, 'test'))):
        print(f"ERROR: Expected 'train' and 'test' subdirectories in {class_dir}")
        return False
    
    # Check for OFF files
    train_dir = os.path.join(class_dir, 'train')
    test_dir = os.path.join(class_dir, 'test')
    
    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.off')]
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.off')]
    
    if not train_files or not test_files:
        print(f"ERROR: No .OFF files found in train/test directories")
        return False
    
    print(f"Found {len(train_files)} training and {len(test_files)} test files in {check_class}")
    
    # Ensure output directory is writable
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.demo:
        # For dataset generation, check renders directory
        renders_dir = os.path.join(args.output_dir, "renders")
        os.makedirs(renders_dir, exist_ok=True)
    
    try:
        test_file = os.path.join(args.output_dir, "write_test.txt")
        with open(test_file, 'w') as f:
            f.write("Write test")
        os.remove(test_file)
    except Exception as e:
        print(f"ERROR: Cannot write to output directory: {e}")
        return False
    
    print(f"Output directory is writable: {args.output_dir}")
    print("Path validation successful!\n")
    return True

def setup_scene(demo_mode=False):
    """Clear scene and set up rendering environment"""
    # Clear existing scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear data blocks
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for cam in bpy.data.cameras:
        if cam.name != "MVCNNCamera":
            bpy.data.cameras.remove(cam)
    
    # Set up lighting
    # Key light
    bpy.ops.object.light_add(type='AREA')
    key_light = bpy.context.object
    key_light.name = "Key Light"
    key_light.location = (4, -2, 4)
    key_light.rotation_euler = (math.radians(45), math.radians(0), math.radians(45))
    key_light.data.energy = 400
    key_light.data.size = 2.0
    
    # Fill light
    bpy.ops.object.light_add(type='AREA')
    fill_light = bpy.context.object
    fill_light.name = "Fill Light"
    fill_light.location = (-4, -1, 2)
    fill_light.rotation_euler = (math.radians(30), math.radians(0), math.radians(-60))
    fill_light.data.energy = 200
    fill_light.data.size = 3.0
    
    # Back light
    bpy.ops.object.light_add(type='AREA')
    back_light = bpy.context.object
    back_light.name = "Back Light" 
    back_light.location = (0, 3, 2)
    back_light.rotation_euler = (math.radians(-30), math.radians(0), math.radians(180))
    back_light.data.energy = 150
    back_light.data.size = 2.0
    
    # Set up world lighting
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    
    # Set up floor for demo (only if in demo mode)
    if demo_mode:
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.5))
        floor = bpy.context.object
        floor.name = "Floor"
        
        # Create floor material
        floor_mat = bpy.data.materials.new("FloorMaterial")
        floor_mat.use_nodes = True
        nodes = floor_mat.node_tree.nodes
        
        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
        
        # Create nodes for simple grid texture
        output = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        checker = nodes.new(type='ShaderNodeTexChecker')
        
        # Set up checker properties
        checker.inputs["Scale"].default_value = 10.0
        checker.inputs["Color1"].default_value = (0.2, 0.2, 0.2, 1.0)
        checker.inputs["Color2"].default_value = (0.3, 0.3, 0.3, 1.0)
        
        # Link nodes
        links = floor_mat.node_tree.links
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
        links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        
        # Assign material
        floor.data.materials.append(floor_mat)

def configure_render_settings(args, demo_mode=False):
    """Configure rendering settings"""
    render = bpy.context.scene.render
    
    # Set resolution
    render.resolution_x = args.image_size
    render.resolution_y = args.image_size
    render.resolution_percentage = 100
    
    # Set quality
    if args.render_quality == 'DRAFT':
        render.engine = 'BLENDER_WORKBENCH'
        bpy.context.scene.display.shading.light = 'STUDIO'
        bpy.context.scene.display.shading.color_type = 'MATERIAL'
    elif args.render_quality == 'MEDIUM':
        render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = 32
        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.eevee.use_ssr = True
        bpy.context.scene.eevee.use_soft_shadows = True
    else:  # HIGH
        render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 64
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.use_denoising = True
    
    # Output format for demo
    if demo_mode:
        render.image_settings.file_format = 'FFMPEG'
        render.ffmpeg.format = 'MPEG4'
        render.ffmpeg.codec = 'H264'
        render.ffmpeg.constant_rate_factor = 'MEDIUM'
        render.ffmpeg.ffmpeg_preset = 'GOOD'
        # Set to 30 fps for smoother animation
        bpy.context.scene.render.fps = 30
    else:
        render.image_settings.file_format = 'PNG'

def setup_camera():
    """Create and configure the camera"""
    # Create a new camera if it doesn't exist
    if "MVCNNCamera" in bpy.data.cameras:
        cam_data = bpy.data.cameras["MVCNNCamera"]
    else:
        cam_data = bpy.data.cameras.new("MVCNNCamera")
    
    if "MVCNNCamera" in bpy.data.objects:
        cam = bpy.data.objects["MVCNNCamera"]
    else:
        cam = bpy.data.objects.new("MVCNNCamera", cam_data)
        bpy.context.collection.objects.link(cam)
    
    # Configure camera
    cam_data.lens = 35  # 35mm lens
    cam_data.sensor_width = 32
    
    # Set as active camera
    bpy.context.scene.camera = cam
    
    return cam

def import_off_file(filepath):
    """Import an OFF file and return the object"""
    # Parse OFF file manually since Blender doesn't support it natively
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip comments and empty lines
    lines = [l.strip() for l in lines if not l.startswith('#') and l.strip()]
    
    # Check OFF header
    if lines[0] != "OFF":
        lines.insert(0, "")  # Handle case where "OFF" is on same line as counts
    
    # Parse vertex and face counts
    if lines[0] == "OFF":
        counts = lines[1].split()
    else:
        header = lines[0].replace("OFF", "").strip()
        counts = header.split()
    
    n_verts = int(counts[0])
    n_faces = int(counts[1])
    
    # Read vertices and faces
    vertices = []
    for i in range(n_verts):
        vcoords = lines[i + 2].split()
        vertices.append((float(vcoords[0]), float(vcoords[1]), float(vcoords[2])))
    
    faces = []
    for i in range(n_faces):
        fverts = lines[i + 2 + n_verts].split()
        face_vert_count = int(fverts[0])
        face_indices = [int(fverts[j]) for j in range(1, face_vert_count + 1)]
        faces.append(face_indices)
    
    # Create mesh
    mesh = bpy.data.meshes.new(os.path.basename(filepath))
    mesh.from_pydata(vertices, [], faces)
    mesh.validate()
    mesh.update()
    
    # Create object
    obj = bpy.data.objects.new(os.path.basename(filepath), mesh)
    bpy.context.collection.objects.link(obj)
    
    # Center and normalize object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Center the object
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    obj.location = (0, 0, 0)
    
    # Scale to unit size
    max_dim = max(obj.dimensions)
    obj.scale = (1.0/max_dim, 1.0/max_dim, 1.0/max_dim)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Create material
    mat = bpy.data.materials.new(name="ModelMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    principled.inputs["Metallic"].default_value = 0.2
    principled.inputs["Specular"].default_value = 0.3
    principled.inputs["Roughness"].default_value = 0.4
    
    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return obj

def generate_camera_positions(args):
    """Generate camera positions for MVCNN views"""
    positions = []
    
    for elev_idx, elevation in enumerate(args.elevation_angles):
        elev_rad = math.radians(elevation)
        z = args.radius * math.sin(elev_rad)
        circle_radius = args.radius * math.cos(elev_rad)
        
        for i in range(args.num_views):
            angle = i * 2 * math.pi / args.num_views
            x = circle_radius * math.cos(angle)
            y = circle_radius * math.sin(angle)
            azimuth = i * 360 / args.num_views;
            
            positions.append({
                "position": (x, y, z),
                "azimuth": azimuth,
                "elevation": elevation,
                "view_idx": i + elev_idx * args.num_views
            })
    
    return positions

def create_visualization_markers(camera_positions, args):
    """Create visual markers to show camera positions"""
    markers = []
    
    # Define some vibrant colors
    colors = [
        (1.0, 0.2, 0.2, 1.0),  # Red
        (0.2, 1.0, 0.2, 1.0),  # Green
        (0.2, 0.2, 1.0, 1.0),  # Blue
        (1.0, 1.0, 0.2, 1.0),  # Yellow
        (1.0, 0.2, 1.0, 1.0),  # Magenta
        (0.2, 1.0, 1.0, 1.0),  # Cyan
        (1.0, 0.6, 0.0, 1.0),  # Orange
        (0.6, 0.0, 1.0, 1.0),  # Purple
    ]
    
    for i, pos in enumerate(camera_positions):
        # Create marker for camera position
        bpy.ops.mesh.primitive_cone_add(
            radius1=0.05, 
            radius2=0, 
            depth=0.1, 
            location=pos["position"]
        )
        marker = bpy.context.object
        marker.name = f"CameraMarker_{i}"
        
        # Point the cone toward the origin
        direction = Vector((0, 0, 0)) - Vector(pos["position"])
        rot_quat = direction.to_track_quat('Z', 'Y')
        marker.rotation_euler = rot_quat.to_euler()
        
        # Create material with simple diffuse setup
        mat = bpy.data.materials.new(name=f"MarkerMaterial_{i}")
        
        # Select color based on elevation
        elev_idx = i // args.num_views
        color_idx = elev_idx % len(colors)
        
        # Make material with basic color - no nodes
        mat.use_nodes = False
        mat.diffuse_color = colors[color_idx]
        
        # Assign material
        if len(marker.data.materials) > 0:
            marker.data.materials[0] = mat
        else:
            marker.data.materials.append(mat)
        
        # Make sure the material is visible in viewport
        marker.active_material = mat
        
        # Mark as not renderable (hide from renders)
        marker.hide_render = True
        
        markers.append(marker)
    
    # Path lines are removed - no longer connecting camera markers
    
    return markers

def render_class_preview(class_name, obj, camera, camera_positions, args):
    """Create visualization renders of the camera setup"""
    print(f"  Creating class preview for {class_name}")
    
    # Create the visualization markers
    markers = create_visualization_markers(camera_positions, args)
    
    # Set up output directory for class preview
    preview_dir = os.path.join(args.output_dir, "class_previews", class_name)
    os.makedirs(preview_dir, exist_ok=True)
    
    # Render top-down view
    top_cam_data = bpy.data.cameras.new("PreviewCamera")
    top_cam = bpy.data.objects.new("PreviewCamera", top_cam_data)
    bpy.context.collection.objects.link(top_cam)
    
    # Position for top view
    top_cam.location = (0, 0, args.radius * 2)
    top_cam.rotation_euler = (0, 0, 0)
    
    # Save current camera
    current_cam = bpy.context.scene.camera
    
    # Render top view
    bpy.context.scene.camera = top_cam
    bpy.context.scene.render.filepath = os.path.join(preview_dir, "top_view.png")
    print(f"  Rendering class preview: top view to {bpy.context.scene.render.filepath}")
    bpy.ops.render.render(write_still=True)
    
    # Position for side view
    top_cam.location = (args.radius * 2, 0, 0)
    top_cam.rotation_euler = (math.pi/2, 0, math.pi/2)
    
    # Render side view
    bpy.context.scene.render.filepath = os.path.join(preview_dir, "side_view.png")
    print(f"  Rendering class preview: side view to {bpy.context.scene.render.filepath}")
    bpy.ops.render.render(write_still=True)
    
    # Restore original camera
    bpy.context.scene.camera = current_cam
    
    # Clean up
    bpy.data.objects.remove(top_cam)
    bpy.data.cameras.remove(top_cam_data)
    
    # Remove visualization markers
    for marker in markers:
        bpy.data.objects.remove(marker)
    
    # Remove visualization paths
    for elevation in args.elevation_angles:
        path = bpy.data.objects.get(f"CameraPath_{elevation}deg")
        if path:
            bpy.data.objects.remove(path)

def generate_renderings(obj, class_name, model_name, camera, camera_positions, split_type, args):
    """Generate and save renderings for a model"""
    # Create output directory structure
    model_output_dir = os.path.join(args.output_dir, "renders", class_name, split_type, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Store metadata
    metadata = {
        "class": class_name,
        "model": model_name,
        "split": split_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "views": []
    }
    
    # Render from each camera position
    for pos in camera_positions:
        view_idx = pos["view_idx"]
        elev = pos["elevation"]
        azimuth = pos["azimuth"]
        
        # Position the camera
        camera.location = pos["position"]
        
        # Point the camera at the object
        direction = Vector((0, 0, 0)) - Vector(camera.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
        # Set output filename
        view_filename = f"view_{view_idx:02d}_elev{elev:02d}_azim{azimuth:03.0f}.png"
        view_filepath = os.path.join(model_output_dir, view_filename)
        
        # Set render path
        bpy.context.scene.render.filepath = view_filepath
        
        # Render
        print(f"    Rendering view {view_idx}: {view_filepath}")
        bpy.ops.render.render(write_still=True)
        
        # Add to metadata
        metadata["views"].append({
            "filename": view_filename,
            "view_idx": view_idx,
            "elevation": elev,
            "azimuth": azimuth,
            "camera_position": list(pos["position"])
        })
    
    # Save metadata
    metadata_path = os.path.join(model_output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved metadata to {metadata_path}")
    
    return model_output_dir

def run_camera_animation_demo(args):
    """Run a demonstration of the camera movement animation"""
    print(f"Running MVCNN camera animation demo with {args.demo_class} class")
    
    # Find a model from the specified class
    class_dir = os.path.join(args.modelnet_dir, args.demo_class)
    if not os.path.exists(class_dir):
        print(f"Error: Class directory not found: {class_dir}")
        class_dirs = [d for d in os.listdir(args.modelnet_dir) 
                    if os.path.isdir(os.path.join(args.modelnet_dir, d))]
        print(f"Available classes: {', '.join(sorted(class_dirs[:10]))}...")
        return
    
    # Find a model file (prefer train directory)
    train_dir = os.path.join(class_dir, 'train')
    off_files = []
    if os.path.exists(train_dir):
        off_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.off')]
        if off_files:
            model_path = os.path.join(train_dir, off_files[0])
    
    # If no train files, check test directory
    if not off_files:
        test_dir = os.path.join(class_dir, 'test')
        if os.path.exists(test_dir):
            off_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.off')]
            if off_files:
                model_path = os.path.join(test_dir, off_files[0])
    
    if not off_files:
        print(f"Error: No OFF files found in {class_dir}")
        return
    
    # Set up scene and camera
    setup_scene(demo_mode=True)
    configure_render_settings(args, demo_mode=True)
    camera = setup_camera()
    
    # Import the model
    model_obj = import_off_file(model_path)
    print(f"Imported model: {model_path}")
    
    # Generate camera positions
    camera_positions = generate_camera_positions(args)
    print(f"Generated {len(camera_positions)} camera positions")
    
    # Create path visualizations and markers
    markers = create_visualization_markers(camera_positions, args)
    
    # Add text to show current view 
    bpy.ops.object.text_add(location=(0, 0, 2.5))
    text_obj = bpy.context.object
    text_obj.name = "ViewInfoText"
    text_obj.data.body = "MVCNN Camera Views"
    text_obj.data.align_x = 'CENTER'
    text_obj.data.size = 0.15
    
    # Create a material for the text
    text_mat = bpy.data.materials.new(name="TextMaterial")
    text_mat.use_nodes = False
    text_mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    text_obj.data.materials.append(text_mat)
    
    # Initialize animation
    scene = bpy.context.scene
    scene.frame_start = 1
    fps = scene.render.fps
    
    # Calculate total frames
    # Add pause time at each view and movement time between views
    frames_per_view = int(args.demo_speed * fps)  # Frames for camera movement
    frames_per_pause = int(args.demo_pause * fps)  # Frames for pause at each position
    total_frames = len(camera_positions) * (frames_per_view + frames_per_pause)
    scene.frame_end = total_frames
    
    print(f"Creating animation with {total_frames} frames ({total_frames/fps:.1f} seconds)")
    
    # Set initial camera position
    first_pos = camera_positions[0]
    camera.location = first_pos["position"]
    direction = Vector((0, 0, 0)) - Vector(camera.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # Store global reference to camera positions for the frame handler
    bpy.app.driver_namespace["camera_positions"] = camera_positions
    bpy.app.driver_namespace["frames_per_view"] = frames_per_view
    bpy.app.driver_namespace["frames_per_pause"] = frames_per_pause
    bpy.app.driver_namespace["text_obj"] = text_obj
    
    # Create frame change handler to update text
    def update_text_frame_handler(scene, depsgraph):
        frame = scene.frame_current
        positions = bpy.app.driver_namespace["camera_positions"]
        frames_per_view = bpy.app.driver_namespace["frames_per_view"]
        frames_per_pause = bpy.app.driver_namespace["frames_per_pause"]
        text_obj = bpy.app.driver_namespace["text_obj"]
        
        # Calculate which camera position we're at
        total_frames_per_pos = frames_per_view + frames_per_pause
        pos_idx = (frame - 1) // total_frames_per_pos
        pos_idx = min(pos_idx, len(positions) - 1)  # Clamp to valid index
        
        pos = positions[pos_idx]
        text_obj.data.body = f"View {pos['view_idx']+1}/{len(positions)}\nAzimuth: {pos['azimuth']:.0f}° Elevation: {pos['elevation']}°"
    
    # Add the handler
    bpy.app.handlers.frame_change_pre.clear()  # Clear existing handlers
    bpy.app.handlers.frame_change_pre.append(update_text_frame_handler)
    
    # Set animation keyframes for the camera
    current_frame = 1
    for i, pos in enumerate(camera_positions):
        # Position camera
        camera.location = pos["position"]
        direction = Vector((0, 0, 0)) - Vector(camera.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
        # Insert keyframe for location and rotation
        camera.keyframe_insert(data_path="location", frame=current_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=current_frame)
        
        # Pause at this position
        current_frame += frames_per_pause
        
        # Move to next position (or back to first if this is the last one)
        next_idx = (i + 1) % len(camera_positions)
        next_pos = camera_positions[next_idx]
        
        # Set keyframe for next position
        camera.location = next_pos["position"]
        direction = Vector((0, 0, 0)) - Vector(camera.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler();
        
        camera.keyframe_insert(data_path="location", frame=current_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=current_frame)
        
        # Update frame counter
        current_frame += frames_per_view
    
    # Fix the animation curves to be smoother
    for fcurve in camera.animation_data.action.fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'BEZIER'
            kf.handle_left_type = 'AUTO'
            kf.handle_right_type = 'AUTO'
    
    # Configure the animation better for preview rendering
    scene = bpy.context.scene
    scene.render.use_motion_blur = False  # Disable motion blur for cleaner preview
    
    # Configure viewport display settings for better preview
    space = None
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            break
    
    if space:
        space.shading.type = 'SOLID'  # Use solid shading for faster viewport updates
        space.overlay.show_overlays = True
    
    # Configure render settings for smooth animation
    render = scene.render
    render.fps = 30  # Standard frame rate
    render.frame_map_old = 100
    render.frame_map_new = 100
    
    # Set output for the animation
    output_file = os.path.join(args.output_dir, args.demo_output)
    print(f"Configuring animation output to: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Set output format
    render.image_settings.file_format = 'FFMPEG'
    render.ffmpeg.format = 'MPEG4'
    render.ffmpeg.codec = 'H264'
    render.ffmpeg.constant_rate_factor = 'MEDIUM'
    render.filepath = output_file

    # Just to be safe, return to initial frame
    bpy.context.scene.frame_set(1)
    
    print(f"Animation setup complete. Use Ctrl+F12 to render animation or press play in the timeline to preview.")
    print(f"Output will be saved to: {output_file}")
    
    # ...rest of existing code...
def process_modelnet40(args):
    """Process ModelNet40 dataset to create MVCNN dataset"""
    print(f"Starting MVCNN dataset generation")
    print(f"ModelNet40 directory: {args.modelnet_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Validate paths
    if not validate_paths(args):
        print("ERROR: Path validation failed. Please fix the issues and try again.")
        return
    
    # Ensure output directories exist
    os.makedirs(os.path.join(args.output_dir, "renders"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "class_previews"), exist_ok=True)
    
    # Get camera positions for MVCNN
    camera_positions = generate_camera_positions(args)
    print(f"Generated {len(camera_positions)} camera positions")
    
    # Set up scene and camera
    setup_scene()
    configure_render_settings(args)
    camera = setup_camera()
    
    # Create dataset metadata
    dataset_metadata = {
        "name": "ModelNet40-MVCNN",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "views_per_model": len(camera_positions),
        "image_size": args.image_size,
        "classes": []
    }
    
    # Get list of class directories
    class_dirs = sorted([d for d in os.listdir(args.modelnet_dir) 
                        if os.path.isdir(os.path.join(args.modelnet_dir, d))])
    
    # Filter to specific class if requested
    if args.only_class:
        if args.only_class in class_dirs:
            class_dirs = [args.only_class]
            print(f"Processing only class: {args.only_class}")
        else:
            print(f"ERROR: Specified class '{args.only_class}' not found")
            return
    
    # Process each class
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(args.modelnet_dir, class_name)
        print(f"Processing class {class_idx+1}/{len(class_dirs)}: {class_name}")
        
        # Create class metadata
        class_metadata = {
            "class_name": class_name,
            "class_idx": class_idx,
            "train_models": [],
            "test_models": []
        }
        
        # Process training models
        train_dir = os.path.join(class_dir, 'train')
        off_files_train = sorted([f for f in os.listdir(train_dir) if f.lower().endswith('.off')])
        
        # Limit models per class if specified
        if args.max_models is not None:
            off_files_train = off_files_train[:args.max_models]
        
        print(f"  Found {len(off_files_train)} training models")
        
        # Create a class preview with the first model
        if args.create_class_previews and off_files_train:
            obj = import_off_file(os.path.join(train_dir, off_files_train[0]))
            render_class_preview(class_name, obj, camera, camera_positions, args)
            
            # Clean up after preview
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()
        
        # Process each training model
        for model_idx, off_file in enumerate(off_files_train):
            model_name = os.path.splitext(off_file)[0]
            off_path = os.path.join(train_dir, off_file)
            print(f"  Processing training model {model_idx+1}/{len(off_files_train)}: {model_name}")
            
            # Import the model
            obj = import_off_file(off_path)
            
            # Generate renderings
            model_dir = generate_renderings(obj, class_name, model_name, camera, camera_positions, "train", args)
            
            # Add to class metadata
            class_metadata["train_models"].append({
                "model_name": model_name,
                "model_idx": model_idx,
                "model_dir": model_dir
            })
            
            # Clean up
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()
        
        # Process test models
        test_dir = os.path.join(class_dir, 'test')
        off_files_test = sorted([f for f in os.listdir(test_dir) if f.lower().endswith('.off')])
        
        # Limit test models per class if specified
        if args.max_models is not None:
            off_files_test = off_files_test[:args.max_models]
        
        print(f"  Found {len(off_files_test)} test models")
        
        # Process each test model
        for model_idx, off_file in enumerate(off_files_test):
            model_name = os.path.splitext(off_file)[0]
            off_path = os.path.join(test_dir, off_file)
            print(f"  Processing test model {model_idx+1}/{len(off_files_test)}: {model_name}")
            
            # Import the model
            obj = import_off_file(off_path)
            
            # Generate renderings
            model_dir = generate_renderings(obj, class_name, model_name, camera, camera_positions, "test", args)
            
            # Add to class metadata
            class_metadata["test_models"].append({
                "model_name": model_name,
                "model_idx": model_idx,
                "model_dir": model_dir
            })
            
            # Clean up
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()
        
        # Add class to dataset metadata
        dataset_metadata["classes"].append(class_metadata)
        
        # Save metadata after each class (for safety)
        with open(os.path.join(args.output_dir, "dataset_metadata.json"), 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
    
    # Final save of dataset metadata
    with open(os.path.join(args.output_dir, "dataset_metadata.json"), 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"MVCNN dataset creation complete. Output directory: {args.output_dir}")
    print(f"Dataset contains {len(dataset_metadata['classes'])} classes")

# ========== Main Execution ==========
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print banner
    print("=" * 70)
    print("MVCNN DATASET GENERATOR")
    print("=" * 70)
    
    # Check if we should run in demo mode
    if args.demo:
        print("Running in DEMO mode - generating camera animation")
        run_camera_animation_demo(args)
    elif args.list_classes:
        # Just list classes (validate_paths will handle this)
        validate_paths(args)
    else:
        # Regular dataset generation mode
        process_modelnet40(args)

if __name__ == "__main__":
    main()
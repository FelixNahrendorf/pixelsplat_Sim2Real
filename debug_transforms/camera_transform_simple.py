import numpy as np
from pyquaternion import Quaternion

def apply_coordinate_transformation(position):
    """
    Apply coordinate transformation:
    x_new = -z_old
    y_new = x_old
    z_new = -y_old
    """
    x_old, y_old, z_old = position
    return np.array([-z_old, x_old, -y_old])

def apply_rotation_transformation(quaternion):
    """
    Apply rotation transformation to match the coordinate system change
    """
    # Convert to rotation matrix
    rotation_matrix = quaternion.rotation_matrix
    
    # Transformation matrix
    T = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ])
    
    # Apply transformation: R_new = T * R_old
    transformed_rotation_matrix = T @ rotation_matrix
    
    # Convert back to quaternion
    transformed_quaternion = Quaternion(matrix=transformed_rotation_matrix)
    
    return transformed_quaternion

def quaternion_to_transform_matrix(quaternion, translation):
    """
    Convert quaternion and translation to 4x4 transformation matrix
    """
    rotation_matrix = quaternion.rotation_matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def reorder_camera_data(camera_data):
    """
    Reorder camera data from original order (0,1,2,3,4,5) to new order (0,1,5,3,4,2)
    """
    reorder_mapping = {0: 0, 1: 1, 2: 5, 3: 3, 4: 4, 5: 2}
    camera_list = list(camera_data.items())
    reordered_camera_data = {}
    
    for new_idx in range(len(camera_list)):
        if new_idx < len(camera_list):
            old_idx = reorder_mapping[new_idx]
            if old_idx < len(camera_list):
                camera_name, data = camera_list[old_idx]
                reordered_camera_data[camera_name] = data
    
    return reordered_camera_data

def transform_camera_poses(camera_input_data):
    """
    Transform camera pose data and return the transformed transforms JSON structure.
    
    Args:
        camera_input_data: Dictionary with camera data containing:
            - For each camera: {
                'translation': [x, y, z],
                'rotation': [w, x, y, z] or Quaternion object,
                'camera_intrinsic': 3x3 matrix or list,
                'image_width': int (optional, default 1600),
                'image_height': int (optional, default 900)
              }
    
    Returns:
        dict: Transformed transforms JSON structure
    """
    
    # Process each camera
    camera_data = {}
    for camera_name, data in camera_input_data.items():
        # Get original position and rotation
        original_translation = np.array(data['translation'])
        
        # Handle rotation input (could be quaternion object or list)
        if isinstance(data['rotation'], Quaternion):
            original_quaternion = data['rotation']
        else:
            # Assume [w, x, y, z] format
            rot = data['rotation']
            original_quaternion = Quaternion(w=rot[0], x=rot[1], y=rot[2], z=rot[3])
        
        # Apply x-axis flip
        x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
        original_quaternion = original_quaternion * x_axis_flip
        
        # Apply coordinate transformation
        transformed_translation = apply_coordinate_transformation(original_translation)
        transformed_quaternion = apply_rotation_transformation(original_quaternion)
        
        # Create transformation matrix
        transformed_transform_matrix = quaternion_to_transform_matrix(
            transformed_quaternion, transformed_translation
        )
        
        # Get camera intrinsics
        camera_intrinsic = np.array(data['camera_intrinsic'])
        fl_x = camera_intrinsic[0, 0]
        fl_y = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        # Get image dimensions (with defaults)
        w = data.get('image_width', 1600)
        h = data.get('image_height', 900)
        
        camera_data[camera_name] = {
            'transform_matrix': transformed_transform_matrix,
            'fl_x': float(fl_x),
            'fl_y': float(fl_y),
            'cx': float(cx),
            'cy': float(cy),
            'w': w,
            'h': h
        }
    
    # Reorder cameras
    camera_data = reorder_camera_data(camera_data)
    
    # Create transformed transforms JSON structure
    transformed_json = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    # Process each camera in the reordered data
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['transform_matrix'].tolist(),
            "fl_x": data['fl_x'],
            "fl_y": data['fl_y'],
            "cx": data['cx'],
            "cy": data['cy'],
            "w": data['w'],
            "h": data['h'],
            "camera_name": camera_name
        }
        
        transformed_json["frames"].append(frame)
    
    return transformed_json

# Example usage:
"""
# Example input data format
camera_input = {
    'CAM_FRONT': {
        'translation': [1.5, 0.0, 1.8],
        'rotation': [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z]
        'camera_intrinsic': [[1266.417, 0, 816.267],
                            [0, 1266.417, 491.507],
                            [0, 0, 1]],
        'image_width': 1600,
        'image_height': 900
    },
    # ... more cameras
}

# Transform and get result
result = transform_camera_poses(camera_input)
"""
import torch
import numpy as np
from PIL import Image


def img_path_to_Torch(image_path, resolution):
    pil_image = Image.open(image_path)
    # Resampling PIL iamge to the desire shape
    resized_image = np.array(pil_image.resize(resolution), dtype=float) #, Image.LANCZOS
    # print(f"Read image ## {image_path} ## with shape={pil_image.size} tried to be resized to {resolution}") 
    # return torch.from_numpy(resized_image).permute(2, 0, 1)[:3, :, :]/255.0 
    # we need to pass images of [0, 255.0] to the transforms, not [0, 1]
    return torch.from_numpy(resized_image).permute(2, 0, 1)[:3, :, :]/255.0
    
def depth_path_to_Torch(depth_map_path, resolution):
    depth_information = Image.open(depth_map_path)
    # Resampling PIL depth map to the desire shape
    depth_information = np.array(depth_information.resize(resolution), dtype=float) #, Image.LANCZOS
    # print(f"Read depth_map ## {depth_map_path} ## with shape={depth_information.size} tried to be resized to {resolution}") 
    return torch.from_numpy(depth_information) / 1000.0
import json
import cv2
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from preprocess import Preprocessing

class PlantDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, coords_json_path, ground_truths_json_path, augment=False, rgb_only=True, augment_multiplier=1):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.augment = augment
        self.rgb_only = rgb_only
        self.augment_multiplier = augment_multiplier if augment else 1

        # Load coordinates
        with open(coords_json_path) as c:
            coords_data = json.load(c)
        
        # Format the coordinates dictionary properly
        self.coords_dict = {}
        for item in coords_data:
            image_path = item["image_path"]
            print(image_path)
            # Extract just the image name without prefix
            if image_path.startswith("RGB_"):
                key = image_path[4:]  # Remove "RGB_" prefix
            elif image_path.startswith("Depth_"):
                key = image_path[6:]  # Remove "Depth_" prefix
            else:
                key = image_path
                
            self.coords_dict[key] = (item["x"], item["y"])
        
        print("Coordinate Dictionary created from array of coords dictionaries:\n")
        print(f"{type(self.coords_dict)}")
        first_3 = {k: v for i, (k, v) in enumerate(self.coords_dict.items()) if i < 3}
        print(f"Index first 3 items of new dict: {first_3}")

        # Load measurements/biometrics data
        with open(ground_truths_json_path) as m:
            ground_truth_data = json.load(m)
        self.measurements = ground_truth_data["Measurements"] # Remove ["Measurements"] to use INESC dataset
        
        # Get lists of all image files
        self.rgb_files = sorted([f for f in os.listdir(rgb_dir)])
        print(f"There's {len(self.rgb_files)} RGB images")
        self.depth_files = sorted([f for f in os.listdir(depth_dir)])
        print(f"There's {len(self.depth_files)} Depth images")
        
        # Create a list of valid images that have both coordinates and measurements
        self.valid_images = []
        
        for rgb_file in self.rgb_files:
            # Extract image number for lookup in measurements
            if rgb_file.startswith("RGB_"):
                image_number = rgb_file.replace("RGB_", "").replace(".png", "")
                image_key = f"Image{image_number}"
            else:
                image_key = rgb_file.replace(".png", "")
            print(f"Image key: {image_key}\n")

            # Check if image has both coordinates and measurements
            if rgb_file.startswith("RGB_"):
                basic_name = rgb_file[4:]  # Remove "RGB_" prefix
                
            else:
                basic_name = rgb_file
        
            
            if basic_name in self.coords_dict.keys():
                print(f"{basic_name} has coordinates")
            else:
                print(f"{basic_name} has NO coordinates avaialble")

            if image_key in self.measurements.keys():
                print(f"{image_key} has measurements")
            else:
                print(f"{image_key} has NO measurements")

            if basic_name in self.coords_dict.keys() and image_key in self.measurements.keys():
                self.valid_images.append({
                    "rgb_file": rgb_file,
                    "depth_file": f"Depth_{basic_name}", # need to change nomenclature depending on dataset. NEED TO SOLVE
                    "basic_name": basic_name,
                    "image_key": image_key
                })
        
        print(f"Found {len(self.valid_images)} valid images with coordinates and measurements")
        print(f"Out of {len(self.rgb_files)} RGB images and {len(self.measurements)} measurements")
        
        # Find min/max values for normalization
        self.min_max_values = self._get_min_max_values()

        # Preprocessing tools
        self.preprocessor = Preprocessing()

    def _get_min_max_values(self):
        """Calculate min and max values for each biometric parameter for normalization"""
        metrics = {
            "FreshWeightShoot": {"min": float('inf'), "max": float('-inf')},
            "DryWeightShoot": {"min": float('inf'), "max": float('-inf')},
            "Height": {"min": float('inf'), "max": float('-inf')},
            "Diameter": {"min": float('inf'), "max": float('-inf')},
            "LeafArea": {"min": float('inf'), "max": float('-inf')}
        }

        
        for image_key in [img["image_key"] for img in self.valid_images]:
            image_data = self.measurements[image_key]
            for metric in metrics.keys():
                if metric in image_data:
                    # Handle string value to float conversion
                    #print(f"{image_data}: {image_data[metric]} | {metric}")
                    if isinstance(image_data[metric], str):
                        image_data[metric] = float(image_data[metric].replace(',', '.'))
                    metrics[metric]["min"] = min(metrics[metric]["min"], image_data[metric])
                    metrics[metric]["max"] = max(metrics[metric]["max"], image_data[metric])
        
        return metrics

    def normalize_biometrics(self, biometrics):
        """Normalize biometric values to range [0,1] as described in the paper"""
        normalized = {}
        
        for metric, value in biometrics.items():
            min_val = self.min_max_values[metric]["min"]
            max_val = self.min_max_values[metric]["max"]
            normalized[metric] = (value - min_val) / (max_val - min_val)
            
        return normalized

    def __len__(self):
        return len(self.valid_images) * self.augment_multiplier
    
    def __getitem__(self, idx):

        # Map to original image
        original_idx = idx % len(self.valid_images)
        aug_version = idx // len(self.valid_images)
        
        # Get the image info from our valid images list
        image_info = self.valid_images[original_idx]
        rgb_file = image_info["rgb_file"]
        depth_file = image_info["depth_file"]
        basic_name = image_info["basic_name"]
        image_key = image_info["image_key"]
        
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (C, H, W)

        # Get RGB center point
        rgb_center = self.coords_dict[basic_name]
        
        # Process RGB image
        normalized_rgb = self.preprocessor.normalize(rgb_tensor, is_depth=False)
        
        if self.augment and aug_version > 0:
            cropped_rgb = self.preprocessor.augment(normalized_rgb, rgb_center, is_depth=False)
        else:
            cropped_rgb = self.preprocessor.crop(normalized_rgb, rgb_center, is_depth=False)
        

        # For INESC data where integers are represented by strings: NEED TO CHANGE IT DIRECTLY IN THE IMAGE GTs LATER!
        def convert_value(value):
            if isinstance(value, str):
                return float(value.replace(',', '.'))
            return float(value) if value is not None else 0.0

        # Get biometrics data with safe defaults
        measurement_data = self.measurements[image_key]
        biometrics = {
            "FreshWeightShoot": convert_value(measurement_data.get("FreshWeightShoot", 0.0)),
            "DryWeightShoot": convert_value(measurement_data.get("DryWeightShoot", 0.0)),
            "Height": convert_value(measurement_data.get("Height", 0.0)),
            "Diameter": convert_value(measurement_data.get("Diameter", 0.0)),
            "LeafArea": convert_value(measurement_data.get("LeafArea", 0.0))
        }
        normalized_biometrics = self.normalize_biometrics(biometrics)
        
        # If RGB only mode, return just RGB data
        if self.rgb_only:
            return {
                'rgb': cropped_rgb,
                'biometrics': normalized_biometrics,
                'image_name': image_key
            }
            
        # Process depth image
        depth_path = os.path.join(self.depth_dir, depth_file)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        print(f"DEPTH TENSOR TYPE: {type(depth)}")
        print(f"DEPTH: {depth}")
        depth_tensor = torch.from_numpy(depth).float()  # (H, W)
        depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension
        
        depth_center = self.coords_dict[basic_name]  # Use same basic name for depth
        
        normalized_depth = self.preprocessor.normalize(depth_tensor, is_depth=True)
        
        if self.augment and aug_version > 0:
            cropped_depth = self.preprocessor.augment(normalized_depth, depth_center, is_depth=True)
        else:
            cropped_depth = self.preprocessor.crop(normalized_depth, depth_center, is_depth=True)
        
        # Return both RGB and depth data
        return {
            'rgb': cropped_rgb,
            'pre_cropped_depth' : depth_tensor, # Use this tensor for grayscale visualization (not cropped and not normalized)
            'depth': cropped_depth,
            'biometrics': normalized_biometrics,
            'image_name': image_key
        }
    
if __name__ == "__main__":
    print(__name__)
    #NewRGBImages/images_with_gt
    plant_dataset = PlantDataset(rgb_dir="../RGBImages", depth_dir= "../DepthImages", coords_json_path="../Coordinates/coordinates.json", ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json", augment=False, rgb_only=False)
    print(len(plant_dataset))
    plant_info = plant_dataset[180] # 180
    
    # Preprocesor
    preproc = Preprocessing()


    # Plant
    print(f"\nPlant: {plant_info["image_name"]}")

    # Depth 
    depth_tensor = plant_info["depth"]
    pre_crop_depth = plant_info["pre_cropped_depth"]
    preproc.normalize_for_visualization(depth_tensor=pre_crop_depth, display=True)
    print(f"Depth tensor type: {depth_tensor.dtype}")
    print(f"Depth tensor shape: {depth_tensor.shape}\n\n")

    

    # RGB
    rgb_tensor = plant_info["rgb"]
    #cropped_for_viz = preproc.normalize_for_visualization(rgb_tensor=normalized_rgb, display=True)
    plt.imshow(rgb_tensor.permute(1, 2, 0).cpu().numpy())
    plt.title("RGB normalized")
    plt.show()
    print(f"RGB tensor type: {rgb_tensor.dtype}")
    print(f"RGB tensor shape: {rgb_tensor.shape}\n\n")

    # Ground Truth
    ground_truth = plant_info["biometrics"]
    print(f"Ground Truth: {ground_truth}")
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import cv2
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np



class ManualCenterCrop:
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_width = self.crop_height = crop_size
        else:
            self.crop_width, self.crop_height = crop_size
            
    '''
    (PIL based cropping)

    def __call__(self, image: Image.Image, center: tuple):
        cx, cy = center
        left = int(cx - self.crop_width / 2)
        upper = int(cy - self.crop_height / 2)
        right = left + self.crop_width
        lower = upper + self.crop_height

        # Ensure crop box stays within image bounds
        left = max(0, left)
        upper = max(0, upper)
        right = min(image.width, right)
        lower = min(image.height, lower)

        return image.crop((left, upper, right, lower))
    '''
    
    def __call__(self, tensor: torch.Tensor, center: tuple) -> torch.Tensor:
        """
        (Torch based cropping, processing is faster and can be converted to PIL image in case we want to visually inspect it)

        Manually crop a tensor image (C x H x W) around a center point.

        Args:
            tensor: Torch tensor image [C, H, W]
            center: (x, y) coordinates in the image

        Returns:
            Cropped tensor image [C, crop_height, crop_width]
        """
        c, h, w = tensor.shape
        cx, cy = center

        half_w = self.crop_width // 2
        half_h = self.crop_height // 2

        left = max(0, cx - half_w)
        right = min(w, cx + half_w)
        top = max(0, cy - half_h)
        bottom = min(h, cy + half_h)

        cropped = tensor[:, top:bottom, left:right]

        # Pad if necessary to maintain exact size
        pad_h = self.crop_height - cropped.shape[1]
        pad_w = self.crop_width - cropped.shape[2]

        if pad_h > 0 or pad_w > 0:
            cropped = F.pad(
                cropped,
                (0, pad_w, 0, pad_h),  # pad (left, right, top, bottom)
                mode='constant',
                value=0
            )

        return cropped


class Preprocessing:

    def __init__(self, crop_size_rgb=512, crop_size_depth=512, translate=0.05): # change crop_size_depth to 256

        self.crop_size_depth= crop_size_depth
        self.crop_size_rgb = crop_size_rgb

        self.cropper_rgb = ManualCenterCrop(self.crop_size_rgb)
        self.cropper_depth = ManualCenterCrop(self.crop_size_depth)


        self.augmentation_transform = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(translate, translate)),
        ])

        # Acrescentar augmentations | Remover background das imagens | Explainability face ao background | Fator escala (alturas das câmaras): averiguar como resolver



    def crop(self, image_tensor: torch.Tensor, center: tuple, is_depth=False) -> torch.Tensor: #image: Image.Image
        if is_depth:
            cropper = self.cropper_depth
            cropped = cropper(image_tensor, center)

        else:
            cropper = self.cropper_rgb
            cropped = cropper(image_tensor, center)
        return cropped #self.to_tensor(cropped)
    

    def augment(self, image_tensor: torch.Tensor, center: tuple,  is_depth=False) -> torch.Tensor: #image: Image.Image
        augmented = self.augmentation_transform(image_tensor)
        return self.crop(augmented, center, is_depth)
    
    def knn_fill_depth(self, depth_tensor: torch.Tensor, missing_val=0, max_iterations=100, stop_threshold=100):
        """
        Fill missing values in a depth tensor using iterative 3x3 mean filtering,
        with early stopping when few or no missing values remain.

        Args:
            depth_tensor: Tensor of shape [1, H, W]
            missing_val: Value to treat as missing (usually 0)
            max_iterations: Max number of passes
            stop_threshold: Stop if remaining zeros fall below this number

        Returns:
            Filled depth tensor
        """
        filled = depth_tensor.clone()

        kernel = torch.ones((1, 1, 3, 3), device=filled.device)

        for i in range(max_iterations):
            padded = F.pad(filled, (1, 1, 1, 1), mode='replicate')

            sum_neighbors = F.conv2d(padded, kernel, padding=0)

            padded_mask = F.pad((filled != missing_val).float(), (1, 1, 1, 1), mode='constant', value=0)
            count_nonzero = F.conv2d(padded_mask, kernel, padding=0)

            avg = sum_neighbors / torch.clamp(count_nonzero, min=1.0)

            newly_filled = (filled == missing_val) & (count_nonzero > 0)
            filled[newly_filled] = avg[newly_filled]

            remaining_zeros = torch.sum(filled == missing_val).item()
            print(f"Pass {i+1}: Remaining zeros: {remaining_zeros}")

            if remaining_zeros <= stop_threshold:
                print("Stopping early due to low remaining zeros.")
                break

        return filled


    def normalize(self, image_tensor:torch.Tensor, is_depth=False, fill_missing=False):
        """
        Normalize image tensor with optional depth filling
        
        Args:
            image_tensor: Input tensor to normalize
            is_depth: Whether this is a depth tensor
            fill_missing: Whether to fill missing (0) values in depth maps
            
        Returns:
            Normalized tensor
        """
        image_tensor = image_tensor.float()

        if is_depth:
            # Keep original tensor for later operations
            depth_tensor = image_tensor.clone()
            
            if fill_missing:
                print("Zeros before fill:", torch.sum(depth_tensor == 0).item())
                # Fill missing values - pass tensor as is
                depth_tensor = self.knn_fill_depth(depth_tensor, missing_val=0)
                print("Zeros after fill:", torch.sum(depth_tensor == 0).item())
        
            # Clamp based on 98% of the actual scene
            vmin = np.percentile(depth_tensor.cpu().numpy(), 1)   
            vmax = np.percentile(depth_tensor.cpu().numpy(), 99) 
            
            print(f"[DEPTH] - Min: {vmin} | Max: {vmax}")

            # Clamp to valid range
            depth_tensor = torch.clamp(depth_tensor, min=vmin, max=vmax)
            
            # Normalize to [0, 1]
            normalized_image_tensor = (depth_tensor - vmin) / (vmax - vmin)
            print(f"Min of normalized depth tensor: {torch.min(normalized_image_tensor)}")
            print(f"Max of normalized depth tensor: {torch.max(normalized_image_tensor)}")
                    
        else:
            normalized_image_tensor = image_tensor/255.0

        return normalized_image_tensor
    
    def normalize_for_visualization(self, depth_tensor, display=False):
        """
        Normalize depth image as per the paper, then scale to [0, 255] for grayscale visualization.
        """

        depth_tensor = depth_tensor.squeeze(0)
        depth_tensor = depth_tensor.float()
        
        # Clamp based on 98% of the actual scene (we can exlude irrelevant extremes/outliers from the tensor that are not representative)
        vmin = np.percentile(depth_tensor, 1)
        vmax = np.percentile(depth_tensor, 99) 

        # Clamp to valid range
        depth_tensor = torch.clamp(depth_tensor, min=vmin, max=vmax)

        # Normalize to [0, 1]
        normalized = (depth_tensor - vmin) / (vmax - vmin)

        # Scale to [0, 255] for grayscale viewing
        grayscale = (normalized * 255)

        print(f"VIS - Min of normalized depth tensor: {torch.min(grayscale)}")
        print(f"VIS - Max of normalized depth tensor: {torch.max(grayscale)}")

        if display:
            plt.imshow(grayscale, cmap='gray')
            plt.title("Depth Visualization (Normalized)")
            plt.colorbar(label='Pixel Intensity (0-255)')
            plt.show()
            plt.show()


        return grayscale, vmin, vmax

    def full_pipeline(self,):
        return
    
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
            #"LeafArea": convert_value(measurement_data.get("LeafArea", 0.0))
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
        
        normalized_depth = self.preprocessor.normalize(depth_tensor, is_depth=True, fill_missing=True)
        
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
   
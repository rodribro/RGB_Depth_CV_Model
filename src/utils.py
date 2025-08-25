import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import torch

### Preprocessing functions
def depth_values_frequency_histogram_original():

    depth_path = "../DepthImages"
    images = [f for f in os.listdir(depth_path)]
    print(f"Images: {images}")
    num_images = len(images)
    print(num_images)
    height, width = 1080, 1920
    all_values = np.zeros((num_images * height * width,), dtype=np.uint16)

    print(f"Preallocated shape: {all_values.shape}")

    idx = 0 # For smart allocation
    for img in images:
        img_path = os.path.join(depth_path, img)
        # Read raw depth image with original bit-depth (likely uint16)
        depth_raw = cv2.imread(depth_path+'/'+img, cv2.IMREAD_UNCHANGED)


        if depth_raw is None:
            print(f"Warning: Failed to load {depth_path+'/'+img}")
            continue

        print(f"Reading {img_path} | dtype: {depth_raw.dtype} | shape: {depth_raw.shape}")

        # Flatten and append raw values
        flat = depth_raw.flatten()
        all_values[idx:idx + flat.size] = flat
        idx += flat.size
        


    # Plot histogram
    plt.hist(all_values, bins=100)
    plt.title("Raw Depth Value Distribution (All Images)")
    plt.xlabel("Depth Value")
    plt.ylabel("Count")
    plt.show()


def normalize_for_visualization(depth_tensor):
    """
    Normalize depth image as per the paper, then scale to [0, 255] for grayscale visualization.
    """
    #depth_tensor = depth_tensor.float()
    
    # Clamp based on 98% of the actual scene (we can exlude irrelevant extremes/outliers from the tensor that are not representative)
    vmin = np.percentile(depth_tensor, 1)   
    vmax = np.percentile(depth_tensor, 99) 

    # Clamp to valid range
    depth_tensor = torch.clamp(depth_tensor, min=vmin, max=vmax)

    # Normalize to [0, 1]
    normalized = (depth_tensor - vmin) / (vmax - vmin)

    # Scale to [0, 255] for grayscale viewing
    grayscale = (normalized * 255).byte()

    return grayscale, vmin, vmax

def depth_values_frequency_histogram(depth_path):
    '''
    Check height value distribution across all depth images
    '''
    images = [f for f in os.listdir(depth_path)]
    print(f"Images: {images}")
    num_images = len(images)
    print(num_images)
    height, width = 1092, 1456
    all_values = np.zeros((num_images * height * width,), dtype=np.uint16)

    print(f"Preallocated shape: {all_values.shape}")

    idx = 0 # For smart allocation
    for i,img in enumerate(images):
        img_path = os.path.join(depth_path, img)

        # Read raw depth image with original bit-depth (likely uint16)
        depth_raw = cv2.imread(depth_path+'/'+img, cv2.IMREAD_UNCHANGED)

        if depth_raw is None:
            print(f"Warning: Failed to load {depth_path+'/'+img}")
            continue

        print(f"({i}/{num_images-1}) Reading {img_path} | dtype: {depth_raw.dtype} | shape: {depth_raw.shape}")

        # Flatten and append raw values
        flat = depth_raw.flatten()
        all_values[idx:idx + flat.size] = flat
        idx += flat.size
        


    # Plot histogram
    plt.hist(all_values, bins=100)
    plt.title("Raw Depth Value Distribution (All Images)")
    plt.xlabel("Depth Value")
    plt.ylabel("Count")
    plt.show()


### [TRAINING LOOP] functions (metrics and visuals)

# Function to calculate evaluation metrics
def calculate_metrics(outputs, targets):
    # Convert to numpy for easier calculations
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate metrics for each biometric parameter
    metrics = {}
    parameter_names = ['FreshWeightShoot', 'DryWeightShoot', 'Diameter', 'Height']
    
    for i, param in enumerate(parameter_names):
        # Get predictions and actual values for this parameter
        y_pred = outputs_np[:, i]
        y_true = targets_np[:, i]
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((y_pred - y_true) ** 2)
        
        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # Calculate MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_pred - y_true))
        
        # Calculate R² (Coefficient of Determination)
        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to avoid division by zero
        
        # Store metrics for this parameter
        metrics[param] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    # Calculate average metrics across all parameters
    avg_metrics = {
        'MSE': np.mean([metrics[param]['MSE'] for param in parameter_names]),
        'RMSE': np.mean([metrics[param]['RMSE'] for param in parameter_names]),
        'MAE': np.mean([metrics[param]['MAE'] for param in parameter_names]),
        'R2': np.mean([metrics[param]['R2'] for param in parameter_names])
    }
    
    metrics['Average'] = avg_metrics
    
    return metrics

if __name__ == "__main__":
    print(__name__)
    depth_values_frequency_histogram(depth_path= "../Depth_INESC/")
    #depth_values_frequency_histogram_original()

    '''
    depth_np = cv2.imread("src/notebooks/pointcloud_topdown.png", cv2.IMREAD_UNCHANGED)#cv2.imread("DepthImages/Depth_135.png", cv2.IMREAD_UNCHANGED)
    depth_tensor = torch.from_numpy(depth_np)
    grayscale_img, vmin, vmax = normalize_for_visualization(depth_tensor)

    print(f"Min: {vmin} | Max: {vmax}")

    plt.imshow(grayscale_img.squeeze().numpy(), cmap='gray')
    plt.title("Depth Visualization (Normalized)")
    plt.colorbar(label='Pixel Intensity (0-255)')
    plt.show()
    '''
import torch
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


# Import your custom modules
from model import RGBBranch
from preprocess import Preprocessing, PlantDataset

def load_model(model_path, device=None):
    """Load the trained model."""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    model = RGBBranch()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def predict_single_image(model, image_path, center_coords, device, preprocessor):
    """Predict biometrics for a single image."""
    # Load and preprocess image (same as training)
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    
    # Apply same preprocessing as training
    normalized_rgb = preprocessor.normalize(rgb_tensor, is_depth=False)
    cropped_rgb = preprocessor.crop(normalized_rgb, center_coords, is_depth=False)
    
    # Make prediction
    input_tensor = cropped_rgb.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    return prediction.cpu().numpy()[0]

def denormalize_predictions(normalized_predictions, min_max_values):
    """Convert normalized predictions back to original units."""
    param_names = ['FreshWeightShoot', 'DryWeightShoot', 'Diameter', 'Height']
    denormalized = {}
    
    for i, param in enumerate(param_names):
        norm_value = float(normalized_predictions[i])  # Convert to Python float
        min_val = min_max_values[param]["min"]
        max_val = min_max_values[param]["max"]
        denormalized[param] = float(norm_value * (max_val - min_val) + min_val)  # Ensure Python float
    
    return denormalized

def get_min_max_values(ground_truth_json_path):
    """Extract min/max values from ground truth file for denormalization."""
    try:
        with open(ground_truth_json_path, 'r') as f:
            ground_truth_data = json.load(f)
        
        measurements = ground_truth_data #ground_truth_data["Measurements"]
        
        # Initialize min/max tracking
        metrics = {
            "FreshWeightShoot": {"min": float('inf'), "max": float('-inf')},
            "DryWeightShoot": {"min": float('inf'), "max": float('-inf')},
            "Height": {"min": float('inf'), "max": float('-inf')},
            "Diameter": {"min": float('inf'), "max": float('-inf')},
            "LeafArea": {"min": float('inf'), "max": float('-inf')}
        }
        
        def convert_to_float(value):
            """Convert value to float, handling comma decimal separators."""
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Replace comma with dot for European decimal format
                value_str = value.replace(',', '.')
                return float(value_str)
            else:
                raise ValueError(f"Cannot convert {value} to float")
        
        # Find min/max values across all measurements
        for image_key, image_data in measurements.items():
            for metric in metrics.keys():
                if metric in image_data:
                    try:
                        # Convert to float, handling comma decimal separators
                        value = convert_to_float(image_data[metric])
                        
                        # Skip invalid values (NaN, inf, negative values for weights/dimensions)
                        if not np.isfinite(value) or value < 0:
                            continue
                            
                        metrics[metric]["min"] = min(metrics[metric]["min"], value)
                        metrics[metric]["max"] = max(metrics[metric]["max"], value)
                        
                    except (ValueError, TypeError):
                        # Skip values that can't be converted to float
                        print(f"Warning: Invalid value for {metric} in {image_key}: {image_data[metric]}")
                        continue
        
        # Check if we found valid ranges for all metrics
        for metric in metrics:
            if metrics[metric]["min"] == float('inf'):
                print(f"Warning: No valid values found for {metric}")
                metrics[metric]["min"] = 0.0
                metrics[metric]["max"] = 1.0
        
        print(f"Loaded ranges:")
        for metric in metrics:
            print(f"  {metric}: {metrics[metric]['min']:.3f} - {metrics[metric]['max']:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None

def predict_with_coordinates(model_path, image_folder, coordinates_file, 
                           ground_truth_file=None, output_file=None):
    """
    Predict biometrics for images using coordinates file.
    
    Args:
        model_path: Path to trained model
        image_folder: Folder containing images
        coordinates_file: JSON with center coordinates
        ground_truth_file: Optional, for denormalization
        output_file: Optional, save results to JSON
    
    Returns:
        Dictionary with predictions
    """
    # Load model
    model, device = load_model(model_path)
    preprocessor = Preprocessing()
    
    # Load coordinates
    with open(coordinates_file, 'r') as f:
        coords_data = json.load(f)
    
    coords_dict = {}
    for item in coords_data:
        image_path = item["image_path"]
        if image_path.startswith("RGB_"):
            key = image_path[4:]  # Remove "RGB_" prefix
        else:
            key = image_path
        coords_dict[key] = (item["x"], item["y"])
    
    # Get min/max values for denormalization if available
    min_max_values = None
    if ground_truth_file and os.path.exists(ground_truth_file):
        min_max_values = get_min_max_values(ground_truth_file)
        if min_max_values:
            print("✓ Loaded min/max values for denormalization")
        else:
            print("✗ Could not parse ground truth file")
    
    # Process images
    results = {}
    print(f"Processing images from {image_folder}")
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Handle both RGB_ prefixed and regular filenames
            if filename.startswith("RGB_"):
                lookup_key = filename[4:]
            else:
                lookup_key = filename
            
            if lookup_key not in coords_dict:
                print(f"Warning: No coordinates for {filename}")
                continue
            
            image_path = os.path.join(image_folder, filename)
            center_coords = coords_dict[lookup_key]
            
            try:
                # Get prediction
                prediction = predict_single_image(
                    model, image_path, center_coords, device, preprocessor
                )
                
                # Store results
                result = {
                    'FreshWeightShoot': float(prediction[0]),
                    'DryWeightShoot': float(prediction[1]),
                    'Diameter': float(prediction[2]),
                    'Height': float(prediction[3])
                }
                
                # Add denormalized values if possible
                if min_max_values:
                    denorm = denormalize_predictions(prediction, min_max_values)
                    results[filename] = {
                        'normalized': result,
                        'denormalized': denorm
                    }
                else:
                    results[filename] = result
                
                print(f"✓ {filename}")
                
            except Exception as e:
                print(f"✗ Error with {filename}: {e}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def load_ground_truth(ground_truth_file):
    """Load ground truth data and handle comma decimal separators."""
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    def convert_to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return float(value.replace(',', '.'))
        else:
            return float(value)
    
    # Convert all values to floats
    processed_gt = {}
    for image_key, measurements in gt_data.items():
        processed_gt[image_key] = {}
        for param, value in measurements.items():
            try:
                processed_gt[image_key][param] = convert_to_float(value)
            except:
                processed_gt[image_key][param] = None
    
    return processed_gt

def match_predictions_to_ground_truth(predictions_file, ground_truth_file, image_mapping=None):
    """
    Match predictions with ground truth values.
    
    Args:
        predictions_file: JSON file with predictions
        ground_truth_file: JSON file with ground truth
        image_mapping: Optional dict to map prediction filenames to GT keys
    
    Returns:
        Dictionary with matched data
    """
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_file)
    
    matched_data = {
        'Diameter': {'predictions': [], 'ground_truth': [], 'filenames': []},
        'Height': {'predictions': [], 'ground_truth': [], 'filenames': []}
    }
    
    print("Matching predictions to ground truth...")
    print("=" * 50)
    
    for filename, pred_data in predictions.items():
        # Try to find matching ground truth
        gt_key = None
        
        if image_mapping and filename in image_mapping:
            gt_key = image_mapping[filename]
        else:
            # Try different matching strategies
            # Strategy 1: Direct filename match
            if filename in ground_truth:
                gt_key = filename
            else:
                # Strategy 2: Remove file extension
                base_name = filename.rsplit('.', 1)[0]
                if base_name in ground_truth:
                    gt_key = base_name
                else:
                    # Strategy 3: Remove RGB_ prefix and extension
                    if filename.startswith("RGB_"):
                        clean_name = filename[4:].rsplit('.', 1)[0]
                        if clean_name in ground_truth:
                            gt_key = clean_name
                    # Strategy 4: Try partial matching
                    else:
                        for gt_name in ground_truth.keys():
                            if base_name in gt_name or gt_name in base_name:
                                gt_key = gt_name
                                break
        
        if gt_key and gt_key in ground_truth:
            gt_values = ground_truth[gt_key]
            
            # Get denormalized predictions
            if 'denormalized' in pred_data:
                pred_values = pred_data['denormalized']
            else:
                pred_values = pred_data
            
            # Match Diameter and Height
            for param in ['Diameter', 'Height']:
                if param in pred_values and param in gt_values and gt_values[param] is not None:
                    matched_data[param]['predictions'].append(pred_values[param])
                    matched_data[param]['ground_truth'].append(gt_values[param])
                    matched_data[param]['filenames'].append(filename)
                    
                    print(f"✓ {filename} -> {gt_key}")
                    print(f"  {param}: Pred={pred_values[param]:.2f}, GT={gt_values[param]:.2f}")
        else:
            print(f"✗ No ground truth found for {filename}")
    
    print("=" * 50)
    print(f"Matched {len(matched_data['Diameter']['predictions'])} images for Diameter")
    print(f"Matched {len(matched_data['Height']['predictions'])} images for Height")
    
    return matched_data

def calculate_metrics(predictions, ground_truth, parameter_name):
    """Calculate accuracy metrics for a parameter."""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Basic metrics
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    
    # Additional metrics
    pearson_r, p_value = pearsonr(ground_truth, predictions)
    mape = np.mean(np.abs((ground_truth - predictions) / (ground_truth + 1e-8))) * 100
    
    # Error statistics
    errors = predictions - ground_truth
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    return {
        'Parameter': parameter_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Pearson_r': pearson_r,
        'P_value': p_value,
        'Mean_Error': mean_error,
        'Std_Error': std_error,
        'Min_GT': np.min(ground_truth),
        'Max_GT': np.max(ground_truth),
        'Mean_GT': np.mean(ground_truth),
        'Min_Pred': np.min(predictions),
        'Max_Pred': np.max(predictions),
        'Mean_Pred': np.mean(predictions),
        'N_samples': len(predictions)
    }

def create_comparison_plots(matched_data, output_file="prediction_comparison.png", 
                          diameter_range=(5, 35), height_range=(5, 25)):
    """
    Create scatter plots comparing predictions vs ground truth.
    
    Args:
        matched_data: Dictionary with matched predictions and ground truth
        output_file: Output filename for the plot
        diameter_range: Tuple (min, max) for diameter axis limits
        height_range: Tuple (min, max) for height axis limits
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define ranges for each parameter
    param_ranges = {
        'Diameter': diameter_range,
        'Height': height_range
    }
    
    for i, param in enumerate(['Diameter', 'Height']):
        ax = axes[i]
        
        predictions = np.array(matched_data[param]['predictions'])
        ground_truth = np.array(matched_data[param]['ground_truth'])
        
        # Scatter plot
        ax.scatter(ground_truth, predictions, alpha=0.6, s=60)
        
        # Get the range for this parameter
        x_min, x_max = param_ranges[param]
        
        # Perfect prediction line (only within the specified range)
        ax.plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display metrics
        r2 = r2_score(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set axis limits based on the specified range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max) 
        
        ax.set_xlabel(f'True {param} (cm)')
        ax.set_ylabel(f'Predicted {param} (cm)')
        ax.set_title(f'{param} Predictions vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plots saved to {output_file}")

def evaluate_predictions(predictions_file, ground_truth_file, image_mapping=None):
    """
    Main function to evaluate prediction quality.
    
    Args:
        predictions_file: Path to predictions JSON
        ground_truth_file: Path to ground truth JSON
        image_mapping: Optional dict mapping prediction filenames to GT keys
    """
    # Match predictions with ground truth
    matched_data = match_predictions_to_ground_truth(
        predictions_file, ground_truth_file, image_mapping
    )
    
    if not matched_data['Diameter']['predictions']:
        print("No matches found! Check your filename mapping.")
        return None
    
    # Calculate metrics for each parameter
    results = {}
    print("\n" + "="*60)
    print("PREDICTION QUALITY EVALUATION")
    print("="*60)
    
    for param in ['Diameter', 'Height']:
        if matched_data[param]['predictions']:
            metrics = calculate_metrics(
                matched_data[param]['predictions'],
                matched_data[param]['ground_truth'],
                param
            )
            results[param] = metrics
            
            print(f"\n{param.upper()} METRICS:")
            print(f"  R² Score: {metrics['R²']:.3f}")
            print(f"  RMSE: {metrics['RMSE']:.2f} cm")
            print(f"  MAE: {metrics['MAE']:.2f} cm")
            print(f"  MAPE: {metrics['MAPE']:.1f}%")
            print(f"  Pearson r: {metrics['Pearson_r']:.3f}")
            print(f"  Mean Error: {metrics['Mean_Error']:.2f} cm")
            print(f"  Samples: {metrics['N_samples']}")
            
            if param == 'Diameter':
                unit = "cm"
                print(f"  Ground Truth Range: {metrics['Min_GT']:.1f} - {metrics['Max_GT']:.1f} {unit}")
                print(f"  Prediction Range: {metrics['Min_Pred']:.1f} - {metrics['Max_Pred']:.1f} {unit}")
    
    # Create visualization
    create_comparison_plots(matched_data)
    
    # If you want to customize the axis ranges, you can also call:
    # create_comparison_plots(matched_data, diameter_range=(10, 40), height_range=(5, 25))
    
    # Print individual comparisons
    print(f"\n{'='*60}")
    print("INDIVIDUAL COMPARISONS")
    print(f"{'='*60}")
    print(f"{'Filename':<25} {'Param':<8} {'Predicted':<10} {'Actual':<10} {'Error':<8}")
    print("-" * 65)
    
    for param in ['Diameter', 'Height']:
        for i, filename in enumerate(matched_data[param]['filenames']):
            pred = matched_data[param]['predictions'][i]
            actual = matched_data[param]['ground_truth'][i]
            error = pred - actual
            print(f"{filename:<25} {param:<8} {pred:<10.2f} {actual:<10.2f} {error:<8.2f}")
    
    return results

if __name__ == "__main__":

    model_path = "checkpoints/2nd/rgb_branch_final.pt"
    
    predictions = predict_with_coordinates(
        model_path=model_path,
        image_folder="../NewRGBImages/images_with_gt",
        coordinates_file="inesc_coordinates.json",
        ground_truth_file="../GroundTruth_INESC/biometry_inesc_gt.json",
        output_file="inesc_predictions.json"
    )

    results = evaluate_predictions(
        predictions_file="inesc_predictions.json",
        ground_truth_file="../GroundTruth_INESC/biometry_inesc_gt.json"
    )
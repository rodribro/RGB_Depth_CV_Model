import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from model import RGBBranch
from preprocess import Preprocessing
import json

class GradCAM:
    """Generate Grad-CAM visualizations to see what the model is looking at."""
    
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.device = next(model.parameters()).device  # Get model's device
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Find the target layer (ResNet's last conv layer)
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                self.target_layer = module
                print(f"Found target layer: {name}")
                break
        
        if self.target_layer is None:
            # Fallback to ResNet layer4 directly
            self.target_layer = self.model.resnet.layer4
            print("Using ResNet layer4 as target layer")
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)  # Use full_backward_hook
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()  # Detach to avoid gradient issues
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # Detach to avoid gradient issues
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM for a specific output (or average of all outputs)."""
        self.model.eval()
        
        # Ensure input is on the same device as model
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If no class specified, use mean of all outputs for regression
        if class_idx is None:
            score = output.mean()
        else:
            score = output[0, class_idx]
        
        print(f"Score for class {class_idx}: {score.item():.4f}")
        
        # Backward pass
        score.backward(retain_graph=True)
        
        # Check if we captured gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations. Check hook registration.")
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension [C, H, W]
        activations = self.activations[0]  # Remove batch dimension [C, H, W]
        
        print(f"Gradients shape: {gradients.shape}")
        print(f"Activations shape: {activations.shape}")
        
        # Ensure everything is on the same device
        gradients = gradients.to(self.device)
        activations = activations.to(self.device)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        print(f"Weights shape: {weights.shape}")
        print(f"Weights device: {weights.device}")
        print(f"Activations device: {activations.device}")
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = torch.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()

def analyze_model_attention(model_path, test_images_info, output_dir="attention_analysis"):
    """
    Analyze what the model is looking at using Grad-CAM.
    
    Args:
        model_path: Path to trained model
        test_images_info: List of dicts with {'path': ..., 'coords': (x, y), 'name': ...}
        output_dir: Directory to save analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RGBBranch()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model)
    preprocessor = Preprocessing()
    
    param_names = ['FreshWeight', 'DryWeight', 'Diameter', 'Height']
    
    for img_info in test_images_info:
        print(f"\n{'='*50}")
        print(f"Analyzing {img_info['name']}...")
        print(f"{'='*50}")
        
        try:
            # Load and preprocess image
            rgb = cv2.imread(img_info['path'], cv2.IMREAD_UNCHANGED)
            if rgb is None:
                print(f"Error: Could not load image {img_info['path']}")
                continue
                
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
            
            print(f"Original image shape: {rgb.shape}")
            print(f"Image tensor shape: {rgb_tensor.shape}")
            
            # Apply preprocessing
            normalized_rgb = preprocessor.normalize(rgb_tensor, is_depth=False)
            cropped_rgb = preprocessor.crop(normalized_rgb, img_info['coords'], is_depth=False)
            
            print(f"Cropped image shape: {cropped_rgb.shape}")
            
            # Prepare input
            input_tensor = cropped_rgb.unsqueeze(0)  # Add batch dimension
            
            print(f"Input tensor shape: {input_tensor.shape}")
            
            # Get model prediction first
            with torch.no_grad():
                input_for_pred = input_tensor.to(device)
                prediction = model(input_for_pred)
                print(f"Model prediction: {prediction[0].cpu().numpy()}")
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original cropped image
            img_display = cropped_rgb.permute(1, 2, 0).cpu().numpy()
            img_display = np.clip(img_display, 0, 1)  # Ensure valid range
            
            axes[0, 0].imshow(img_display)
            axes[0, 0].set_title('Original Cropped Image')
            axes[0, 0].axis('off')
            
            # Generate Grad-CAM for each parameter
            for i, param in enumerate(param_names):
                if i >= 4:  # Only 4 parameters
                    break
                    
                row = (i + 1) // 3
                col = (i + 1) % 3
                if row >= 2:
                    break
                
                print(f"Generating CAM for {param} (index {i})...")
                
                try:
                    # Generate CAM for this parameter
                    cam = grad_cam.generate_cam(input_tensor.clone(), class_idx=i)
                    
                    # Resize CAM to match input image
                    cam_resized = cv2.resize(cam, (512, 512))
                    
                    # Create heatmap overlay
                    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                    
                    # Overlay on original image
                    overlay = 0.6 * img_display + 0.4 * heatmap
                    
                    axes[row, col].imshow(overlay)
                    axes[row, col].set_title(f'{param} Attention')
                    axes[row, col].axis('off')
                    
                    print(f"✓ Successfully generated CAM for {param}")
                    
                except Exception as e:
                    print(f"✗ Error generating CAM for {param}: {e}")
                    # Show original image in case of error
                    axes[row, col].imshow(img_display)
                    axes[row, col].set_title(f'{param} (Error)')
                    axes[row, col].axis('off')
            
            # Remove unused subplots
            for i in range(len(param_names) + 1, 6):
                row = i // 3
                col = i % 3
                if row < 2:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            save_path = f"{output_dir}/{img_info['name']}_attention.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved parameter-specific attention to {save_path}")
            
            # Generate overall attention (average across all parameters)
            print("Generating overall attention map...")
            try:
                overall_cam = grad_cam.generate_cam(input_tensor.clone(), class_idx=None)
                overall_cam_resized = cv2.resize(overall_cam, (512, 512))
                
                # Create side-by-side comparison
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original
                axes[0].imshow(img_display)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Heatmap only
                im = axes[1].imshow(overall_cam_resized, cmap='jet')
                axes[1].set_title('Model Attention Heatmap')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                # Overlay
                heatmap = plt.cm.jet(overall_cam_resized)[:, :, :3]
                overlay = 0.6 * img_display + 0.4 * heatmap
                axes[2].imshow(overlay)
                axes[2].set_title('Attention Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                overall_save_path = f"{output_dir}/{img_info['name']}_overall_attention.png"
                plt.savefig(overall_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved overall attention to {overall_save_path}")
                
            except Exception as e:
                print(f"✗ Error generating overall attention: {e}")
        
        except Exception as e:
            print(f"✗ Error processing {img_info['name']}: {e}")
            import traceback
            traceback.print_exc()

def simple_attention_test(model_path, image_path, coords):
    """Simple test function to debug attention analysis."""
    print("Running simple attention test...")
    
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RGBBranch()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load image
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    
    # Preprocess
    preprocessor = Preprocessing()
    normalized_rgb = preprocessor.normalize(rgb_tensor, is_depth=False)
    cropped_rgb = preprocessor.crop(normalized_rgb, coords, is_depth=False)
    input_tensor = cropped_rgb.unsqueeze(0)
    
    # Test basic prediction
    with torch.no_grad():
        input_for_pred = input_tensor.to(device)
        prediction = model(input_for_pred)
        print(f"Prediction successful: {prediction[0].cpu().numpy()}")
    
    # Test Grad-CAM
    grad_cam = GradCAM(model)
    
    try:
        cam = grad_cam.generate_cam(input_tensor.clone(), class_idx=2)  # Test diameter
        print(f"✓ Grad-CAM successful! CAM shape: {cam.shape}")
        print(f"CAM range: [{cam.min():.4f}, {cam.max():.4f}]")
        
        # Quick visualization
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        img_display = cropped_rgb.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_display)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (512, 512))
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.6 * img_display + 0.4 * heatmap
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('simple_attention_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"✗ Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First, run a simple test
    print("Running simple test first...")
    success = simple_attention_test(
        "checkpoints/2nd/rgb_branch_final.pt",
        "../RGBImages/RGB_112.png",
        (985, 611)
    )
    
    if success:
        print("\n" + "="*50)
        print("Simple test successful! Running full analysis...")
        print("="*50)
        
    
        test_images = [
            # Original Dataset
            {
                'path': '../RGBImages/RGB_8.png',
                'coords': (1139, 620),
                'name': 'test_RGB_8'
            },
            {
                'path': '../RGBImages/RGB_112.png',
                'coords': (985, 611),
                'name': 'test_RGB_112'
            },
            {
                'path': '../RGBImages/RGB_58.png',
                'coords': (1025, 551),
                'name': 'test_RGB_58'
            },
            {
                'path': '../RGBImages/RGB_58.png',
                'coords': (1025, 551),
                'name': 'test_RGB_58'
            },
            # INESC Dataset
            {
                "path": "../NewRGBImages/images_with_gt/RNGRA1_2024-09-11.png",
                'coords': (775,674),
                'name': 'INESC_test_RNGRA1'
            },
            {
                "path": "../NewRGBImages/images_with_gt/RWGRB4_2024-09-17.png",
                'coords': (712,547),
                'name': 'INESC_test_RWGBR4'
            },
            {
                "path": "../NewRGBImages/images_with_gt/RNGRA2_2024-09-07.png",
                'coords': (568,642),
                'name': 'INESC_test_RNGRA2'
            }

        ]
        
        # Run attention analysis
        analyze_model_attention("checkpoints/2nd/rgb_branch_final.pt", test_images)
    else:
        print("Simple test failed. Check the error messages above.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import json
from matplotlib.backends.backend_pdf import PdfPages
from model import * 
from utils import calculate_metrics
import os
import time
from preprocess import PlantDataset, Preprocessing  

def train_rgb_branch(rgb_branch, num_epochs=10, batch_size=32, lr=0.001, device='cpu', seed=30, augment_multiplier=4):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Create training dataset WITH augmentation
    train_dataset_with_aug = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="../Coordinates/coordinates.json",
        ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
        augment=True,
        rgb_only=True,
        augment_multiplier=augment_multiplier
    )

    # Create validation dataset WITHOUT augmentation
    val_dataset_no_aug = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="../Coordinates/coordinates.json",
        ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
        augment=False,
        rgb_only=True
    )
    
    # Get base count for splitting (divide by multiplier to get original image count)
    base_count = len(train_dataset_with_aug) // augment_multiplier
    indices = list(range(base_count))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_indices = indices[:int(0.8 * base_count)]
    val_indices = indices[int(0.8 * base_count):]
    
    print(f"RGB Branch - Base dataset size: {base_count} images")
    print(f"Training set: {len(train_indices)} images")
    print(f"Validation set: {len(val_indices)} images")
    print(f"With augment_multiplier={augment_multiplier}, training on {len(train_indices) * augment_multiplier} samples per epoch")

    # Create subsets for train/val split
    train_subset = Subset(train_dataset_with_aug, train_indices)
    val_subset = Subset(val_dataset_no_aug, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rgb_branch.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Evaluation metrics history
    val_metrics_history = []
    
    # Directories to store training metrics and logs
    metrics_dir = "../metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    cur_date = time.asctime() # current date to identify when train was performed
    cur_date_no_double_space = cur_date.replace("  ", "_") 
    specific_train = cur_date_no_double_space.replace(" ", "_")

    train_path = os.path.join(metrics_dir, f"RGB_{specific_train}") # create train specific directory
    os.makedirs(train_path)
    model_checkpoints_path = os.path.join(train_path, "checkpoints") #create model checkpoint path
    os.makedirs(model_checkpoints_path)
    visuals_path = os.path.join(train_path, "visuals") #create visuals path
    os.makedirs(visuals_path)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        rgb_branch.train()
        epoch_train_loss = 0.0
        total_train_samples = 0
        
        print(f"RGB Epoch {epoch+1}/{num_epochs} - Training with {augment_multiplier}x augmented data")
        for batch in tqdm(train_loader, desc=f"RGB Training"):
            rgb_images = batch['rgb'].to(device)
            biometrics = torch.stack([
                batch['biometrics'][param] for param in 
                ['FreshWeightShoot', 'DryWeightShoot', 'Diameter', 'Height'] 
            ], dim=1).float().to(device)
            
            optimizer.zero_grad()
            outputs = rgb_branch(rgb_images)
            loss = criterion(outputs, biometrics)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * rgb_images.size(0)
            total_train_samples += rgb_images.size(0)

        # Calculate average training loss
        avg_train_loss = epoch_train_loss / total_train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        rgb_branch.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        # Collect all predictions and targets for evaluation metrics
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"RGB Validation"):
                rgb_images = batch['rgb'].to(device)
                biometrics = torch.stack([
                    batch['biometrics'][param] for param in 
                    ['FreshWeightShoot', 'DryWeightShoot', 'Diameter', 'Height'] 
                ], dim=1).float().to(device)
                
                outputs = rgb_branch(rgb_images)
                loss = criterion(outputs, biometrics)
                val_loss += loss.item() * rgb_images.size(0)
                total_val_samples += rgb_images.size(0)
                
                # Store outputs and targets for metrics calculation
                all_outputs.append(outputs)
                all_targets.append(biometrics)
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(all_outputs, all_targets)
        val_metrics_history.append(metrics)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / total_val_samples
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress and metrics
        print(f"RGB Epoch {epoch+1}/{num_epochs} - Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
        
        # Print evaluation metrics
        print(f"RGB Validation Metrics (Average):")
        print(f"  MSE: {metrics['Average']['MSE']:.4f}")
        print(f"  RMSE: {metrics['Average']['RMSE']:.4f}")
        print(f"  MAE: {metrics['Average']['MAE']:.4f}")
        print(f"  R²: {metrics['Average']['R2']:.4f}")
        
        # Save model checkpoint with metrics
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': rgb_branch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_metrics': metrics,
            }, f'{model_checkpoints_path}/rgb_epoch_{epoch+1}.pt')
    
    print(f"RGB training completed. Results saved to {train_path}")
    
    return {
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'val_metrics': val_metrics_history,
        'checkpoints_path': model_checkpoints_path,
        'final_model_path': f'{model_checkpoints_path}/rgb_branch_final.pt'
    }

def train_depth_branch(depth_branch, num_epochs=10, batch_size=32, lr=0.001, device='cpu', seed=30, augment_multiplier=4):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Create training dataset WITH augmentation (need both RGB and depth)
    train_dataset_with_aug = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="../Coordinates/coordinates.json",
        ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
        augment=True,
        rgb_only=False,  # Need depth data
        augment_multiplier=augment_multiplier
    )

    # Create validation dataset WITHOUT augmentation
    val_dataset_no_aug = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="../Coordinates/coordinates.json",
        ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
        augment=False,
        rgb_only=False
    )
    
    # Get base count for splitting
    base_count = len(train_dataset_with_aug) // augment_multiplier
    indices = list(range(base_count))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_indices = indices[:int(0.8 * base_count)]
    val_indices = indices[int(0.8 * base_count):]
    
    print(f"Depth Branch - Base dataset size: {base_count} images")
    print(f"Training set: {len(train_indices)} images")
    print(f"Validation set: {len(val_indices)} images")
    print(f"With augment_multiplier={augment_multiplier}, training on {len(train_indices) * augment_multiplier} samples per epoch")

    # Create subsets
    train_subset = Subset(train_dataset_with_aug, train_indices)
    val_subset = Subset(val_dataset_no_aug, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(depth_branch.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Directories to store training metrics and logs
    metrics_dir = "../metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    cur_date = time.asctime()
    cur_date_no_double_space = cur_date.replace("  ", "_") 
    specific_train = cur_date_no_double_space.replace(" ", "_")

    train_path = os.path.join(metrics_dir, f"Depth_{specific_train}")
    os.makedirs(train_path)
    model_checkpoints_path = os.path.join(train_path, "checkpoints")
    os.makedirs(model_checkpoints_path)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        depth_branch.train()
        epoch_train_loss = 0.0
        total_train_samples = 0
        
        print(f"Depth Epoch {epoch+1}/{num_epochs}")
        
        for batch in tqdm(train_loader, desc=f"Depth Training"):
            depth_images = batch['depth'].to(device)
            # Only height target for depth branch
            height_target = batch['biometrics']['Height'].unsqueeze(1).float().to(device)
            
            optimizer.zero_grad()
            outputs = depth_branch(depth_images)
            loss = criterion(outputs, height_target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * depth_images.size(0)
            total_train_samples += depth_images.size(0)

        # Calculate average training loss
        avg_train_loss = epoch_train_loss / total_train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        depth_branch.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Depth Validation"):
                depth_images = batch['depth'].to(device)
                height_target = batch['biometrics']['Height'].unsqueeze(1).float().to(device)
                
                outputs = depth_branch(depth_images)
                loss = criterion(outputs, height_target)
                val_loss += loss.item() * depth_images.size(0)
                total_val_samples += depth_images.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / total_val_samples
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Depth Epoch {epoch+1}/{num_epochs} - Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': depth_branch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'{model_checkpoints_path}/depth_epoch_{epoch+1}.pt')
    
    print(f"Depth training completed. Results saved to {train_path}")
    
    return {
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'checkpoints_path': model_checkpoints_path,
        'final_model_path': f'{model_checkpoints_path}/depth_branch_final.pt'
    }

def train_joint_regressor(joint_model, num_epochs=10, batch_size=16, lr=0.0001, device='cpu', seed=30, augment_multiplier=2):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Create training dataset WITH augmentation (need both RGB and depth)
    train_dataset_with_aug = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="../Coordinates/coordinates.json",
        ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
        augment=True,
        rgb_only=False,  # Need both RGB and depth
        augment_multiplier=augment_multiplier
    )

    # Create validation dataset WITHOUT augmentation
    val_dataset_no_aug = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="../Coordinates/coordinates.json",
        ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
        augment=False,
        rgb_only=False
    )
    
    # Get base count for splitting
    base_count = len(train_dataset_with_aug) // augment_multiplier
    indices = list(range(base_count))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_indices = indices[:int(0.8 * base_count)]
    val_indices = indices[int(0.8 * base_count):]
    
    print(f"Joint Regressor - Base dataset size: {base_count} images")
    print(f"Training set: {len(train_indices)} images")
    print(f"Validation set: {len(val_indices)} images")
    print(f"With augment_multiplier={augment_multiplier}, training on {len(train_indices) * augment_multiplier} samples per epoch")

    # Create subsets
    train_subset = Subset(train_dataset_with_aug, train_indices)
    val_subset = Subset(val_dataset_no_aug, val_indices)

    # Create DataLoaders (smaller batch size for joint model)
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(joint_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Directories to store training metrics and logs
    metrics_dir = "../metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    cur_date = time.asctime()
    cur_date_no_double_space = cur_date.replace("  ", "_") 
    specific_train = cur_date_no_double_space.replace(" ", "_")

    train_path = os.path.join(metrics_dir, f"Joint_{specific_train}")
    os.makedirs(train_path)
    model_checkpoints_path = os.path.join(train_path, "checkpoints")
    os.makedirs(model_checkpoints_path)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        joint_model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0
        
        print(f"Joint Epoch {epoch+1}/{num_epochs}")
        
        for batch in tqdm(train_loader, desc=f"Joint Training"):
            rgb_images = batch['rgb'].to(device)
            depth_images = batch['depth'].to(device)
            
            # Prepare targets for joint outputs
            target1 = torch.stack([
                batch['biometrics']['FreshWeightShoot'],
                batch['biometrics']['DryWeightShoot'],
                batch['biometrics']['Diameter']
            ], dim=1).float().to(device)
            
            target2 = batch['biometrics']['Height'].unsqueeze(1).float().to(device)
            
            optimizer.zero_grad()
            outputs = joint_model(rgb_images, depth_images)
            
            # Calculate combined loss
            loss1 = criterion(outputs['output1'], target1)
            loss2 = criterion(outputs['output2'], target2)
            total_loss = loss1 + loss2
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item() * rgb_images.size(0)
            total_train_samples += rgb_images.size(0)

        # Calculate average training loss
        avg_train_loss = epoch_train_loss / total_train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        joint_model.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Joint Validation"):
                rgb_images = batch['rgb'].to(device)
                depth_images = batch['depth'].to(device)
                
                target1 = torch.stack([
                    batch['biometrics']['FreshWeightShoot'],
                    batch['biometrics']['DryWeightShoot'],
                    batch['biometrics']['Diameter']
                ], dim=1).float().to(device)
                
                target2 = batch['biometrics']['Height'].unsqueeze(1).float().to(device)
                
                outputs = joint_model(rgb_images, depth_images)
                
                loss1 = criterion(outputs['output1'], target1)
                loss2 = criterion(outputs['output2'], target2)
                total_loss = loss1 + loss2
                
                val_loss += total_loss.item() * rgb_images.size(0)
                total_val_samples += rgb_images.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / total_val_samples
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Joint Epoch {epoch+1}/{num_epochs} - Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': joint_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'{model_checkpoints_path}/joint_epoch_{epoch+1}.pt')
    
    print(f"Joint training completed. Results saved to {train_path}")
    
    return {
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'checkpoints_path': model_checkpoints_path,
        'final_model_path': f'{model_checkpoints_path}/joint_regressor_final.pt'
    }

def main():
    # Check for device availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Training parameters
    batch_size = 32
    num_epochs = 2
    augment_multiplier = 1
    
    print("="*50)
    print("STAGE 1: Training Individual Branches")
    print("="*50)
    
    # Stage 1a: Train RGB branch
    print("\n1. Training RGB branch...")
    rgb_branch = RGBBranch().to(device)
    rgb_history = train_rgb_branch(
        rgb_branch=rgb_branch,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=0.001,
        device=device,
        augment_multiplier=augment_multiplier
    )
    
    # Save RGB model
    rgb_model_path = rgb_history['final_model_path']
    torch.save(rgb_branch.state_dict(), rgb_model_path)
    print(f"RGB model saved to: {rgb_model_path}")
    
    # Stage 1b: Train Depth branch
    print("\n2. Training Depth branch...")
    depth_branch = DepthBranch().to(device)
    depth_history = train_depth_branch(
        depth_branch=depth_branch,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=0.001,
        device=device,
        augment_multiplier=augment_multiplier
    )
    
    # Save Depth model
    depth_model_path = depth_history['final_model_path']
    torch.save(depth_branch.state_dict(), depth_model_path)
    print(f"Depth model saved to: {depth_model_path}")
    
    print("="*50)
    print("STAGE 2: Training Joint Regressor")
    print("="*50)
    
    # Stage 2: Train Joint regressor
    print("\n3. Training Joint regressor...")
    joint_model = JointRegressor().to(device)
    
    # Load pre-trained weights
    print("Loading pre-trained weights...")
    joint_model.rgb_branch.load_state_dict(torch.load(rgb_model_path, map_location=device))
    joint_model.depth_branch.load_state_dict(torch.load(depth_model_path, map_location=device))
    print("Pre-trained weights loaded successfully!")
    
    joint_history = train_joint_regressor(
        joint_model=joint_model,
        num_epochs=num_epochs,
        batch_size=16,  # Smaller batch size for joint model
        lr=0.0001,      # Lower learning rate for fine-tuning
        device=device,
        augment_multiplier=augment_multiplier
    )
    
    # Save Joint model
    joint_model_path = joint_history['final_model_path']
    torch.save(joint_model.state_dict(), joint_model_path)
    print(f"Joint model saved to: {joint_model_path}")
    
    print("="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"RGB model: {rgb_model_path}")
    print(f"Depth model: {depth_model_path}")
    print(f"Joint model: {joint_model_path}")

if __name__ == "__main__":
    main()
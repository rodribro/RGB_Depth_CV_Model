from model import *
from preprocess import *

def test_rgb_processing_block():
    # Create an instance of the processing block
    rgb_block = RGBProcessingBlock()
    
    # Load your dataset
    plant_dataset = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="coordinates.json",
        ground_truths_json_path="../GroundTruth/GroundTruth_All_388_Images.json"
    )
    
    # Get a sample from your dataset
    sample_idx = 30
    sample = plant_dataset[sample_idx]
    
    # Get the preprocessed RGB image
    rgb_tensor = sample['rgb']
    
    # Add batch dimension for the model
    input_tensor = rgb_tensor.unsqueeze(0)
    
    # Forward pass through the block
    with torch.no_grad():  # No need to track gradients for testing
        output = rgb_block(input_tensor)
    
    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Visualize the original and processed images
    plt.figure(figsize=(12, 6))
    
    # Original preprocessed image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_tensor.permute(1, 2, 0).cpu().numpy())
    plt.title("Preprocessed RGB Image")
    
    # After RGB processing block
    plt.subplot(1, 2, 2)
    processed = output[0].permute(1, 2, 0).detach().cpu().numpy()
    # Normalize for better visualization
    processed = (processed - processed.min()) / (processed.max() - processed.min())
    plt.imshow(processed)
    plt.title("After RGB Processing Block")
    
    plt.savefig('rgb_processing_test.png')  # Save the figure
    plt.show()
    
    # Feature map visualization (showing a few feature maps from first conv layer)
    plt.figure(figsize=(16, 8))
    plt.suptitle("First Conv Layer Feature Maps (Sample)", fontsize=16)
    
    # Get the feature maps from the first conv layer
    with torch.no_grad():
        features = rgb_block.relu(rgb_block.conv1(input_tensor))
    
    # Display a subset of feature maps
    for i in range(min(32, features.shape[1])):
        plt.subplot(4, 8, i+1)
        feature_map = features[0, i].detach().cpu().numpy()
        # Normalize for visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        plt.imshow(feature_map, cmap='viridis')
        plt.title(f"Filter {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps_1st_conv.png')  # Save the figure
    plt.show()

     # Feature map visualization (showing a few feature maps from second conv layer)
    plt.figure(figsize=(16, 8))
    plt.suptitle("Second Conv Layer Feature Maps (Sample)", fontsize=16)
    
    # Get the feature maps from the first conv layer
    with torch.no_grad():
        features_2 =  rgb_block.relu(rgb_block.conv2(features))
    
    # Display a subset of feature maps
    for i in range(min(3, features_2.shape[1])):
        plt.subplot(1, 3, i+1)
        feature_map_2 = features_2[0, i].detach().cpu().numpy()
        # Normalize for visualization
        feature_map_2 = (feature_map_2 - feature_map_2.min()) / (feature_map_2.max() - feature_map_2.min() + 1e-8)
        plt.imshow(feature_map_2, cmap='viridis')
        plt.title(f"Filter {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps.png')  # Save the figure
    plt.show()
    
    print("Test passed! The RGB processing block works with your preprocessed image.")
    
    # Also print the biometrics for reference
    print("\nGround truth biometrics (normalized):")
    for key, value in sample['biometrics'].items():
        print(f"  {key}: {value:.4f}")
        
def test_rgb_branch():
    # Create an instance of the RGB branch
    rgb_branch = RGBBranch()
    
    # Set model to evaluation mode
    rgb_branch.eval()
    
    # Load your dataset
    plant_dataset = PlantDataset(
        rgb_dir="../RGBImages", 
        depth_dir="../DepthImages", 
        coords_json_path="coordinates.json",
        ground_truths_json_path="../GroundTruth/GroundTruth_All_388_Images.json"
    )
    
    # Get a sample from your dataset
    sample_idx = 30
    sample = plant_dataset[sample_idx]
    
    # Get the preprocessed RGB image
    rgb_tensor = sample['rgb']
    
    # Add batch dimension for the model
    input_tensor = rgb_tensor.unsqueeze(0)
    
    # Forward pass through the RGB branch
    with torch.no_grad():  # No need to track gradients for testing
        output = rgb_branch(input_tensor)
    
    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"RGB Branch Output shape: {output.shape}")
    print(f"Output values: {output[0].numpy()}")
    
    # Visualize intermediate activations
    # We'll track activations at different stages
    activations = {}
    
    def get_activations():
        with torch.no_grad():
            # Processing block output
            proc_out = rgb_branch.processing_block(input_tensor)
            activations['proc_block'] = proc_out
            
            # After first ResNet conv
            x = rgb_branch.resnet.conv1(proc_out)
            x = rgb_branch.resnet.bn1(x)
            x = rgb_branch.resnet.relu(x)
            activations['resnet_conv1'] = x
            
            # After maxpool
            x = rgb_branch.resnet.maxpool(x)
            activations['maxpool'] = x
            
            # After each ResNet layer
            x = rgb_branch.resnet.layer1(x)
            activations['layer1'] = x
            x = rgb_branch.resnet.layer2(x)
            activations['layer2'] = x
            x = rgb_branch.resnet.layer3(x)
            activations['layer3'] = x
            x = rgb_branch.resnet.layer4(x)
            activations['layer4'] = x
            
            # After global average pooling
            x = rgb_branch.global_avg_pool(x)
            activations['global_avg_pool'] = x
            
            # After flatten
            x = rgb_branch.flatten(x)
            activations['flatten'] = x
            
            # After first FC layer
            x = F.relu(rgb_branch.fc1(x))
            activations['fc1'] = x
            
            # After dropout
            x = rgb_branch.dropout1(x)
            activations['dropout'] = x
            
            # Final output
            x = rgb_branch.fc2(x)
            activations['output'] = x
    
    # Get all activations
    get_activations()
    
    # Print activation shapes
    print("\nActivation shapes at different stages:")
    for name, activation in activations.items():
        if isinstance(activation, torch.Tensor):
            print(f"{name}: {activation.shape}")
    
    # Visualize some feature maps from different ResNet layers
    plt.figure(figsize=(16, 12))
    plt.suptitle("Feature Maps at Different Stages", fontsize=16)
    
    # Function to visualize feature maps
    def show_feature_maps(activation, layer_name, max_maps=8, start_idx=0):
        # If activation is from a linear layer, reshape to 2D
        if len(activation.shape) == 2:
            # For 2D activations (after flatten), reshape to 2D grid
            grid_size = int(np.ceil(np.sqrt(activation.shape[1])))
            if grid_size > 32:  # Too many features to show meaningfully
                print(f"Too many features in {layer_name} to display: {activation.shape[1]}")
                return
            
            # Create a grid of feature values
            n_features = min(max_maps, activation.shape[1])
            for i in range(n_features):
                plt.subplot(2, 4, i+1)
                plt.bar(range(10), activation[0, i:i+10].cpu().numpy())
                plt.title(f"{layer_name} Units {i}-{i+9}")
                plt.xticks(range(10))
            
        elif len(activation.shape) == 4:
            # For 4D activations (feature maps), show a subset
            n_maps = min(max_maps, activation.shape[1])
            for i in range(n_maps):
                plt.subplot(2, 4, i+1)
                feature_map = activation[0, start_idx + i].cpu().numpy()
                # Normalize for visualization
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                plt.imshow(feature_map, cmap='viridis')
                plt.title(f"{layer_name} Map {start_idx + i}")
                plt.axis('off')
    
    # Show feature maps from processing block
    show_feature_maps(activations['proc_block'], "Proc Block")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('rgb_branch_feature_maps_1.png')
    plt.show()
    
    # Show feature maps from first ResNet layer
    plt.figure(figsize=(16, 12))
    plt.suptitle("ResNet Layer 1 Feature Maps", fontsize=16)
    show_feature_maps(activations['layer1'], "Layer 1")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('rgb_branch_feature_maps_2.png')
    plt.show()
    
    # Compare predicted vs actual biometrics
    print("\nGround truth biometrics (normalized):")
    biometrics = sample['biometrics']
    for key, value in biometrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nPredicted biometrics (from untrained model):")
    # Map output indices to biometric parameters
    predicted = {
        "FreshWeightShoot": output[0, 0].item(),
        "DryWeightShoot": output[0, 1].item(),
        "Diameter": output[0, 2].item(),
        "Height": output[0, 3].item()
    }
    for key, value in predicted.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nNote: Predictions are from an untrained model, so they won't match ground truth yet.")


if __name__ == "__main__":
    #test_rgb_processing_block()
    test_rgb_branch()
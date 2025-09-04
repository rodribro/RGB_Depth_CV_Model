import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RGBProcessingBlock(nn.Module):
    def __init__(self):
        super(RGBProcessingBlock, self).__init__()
        # First convolutional layer (3×3, 32 filters)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second convolutional layer (1×1, 3 filters) - matches input channels for ResNet
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class RGBBranch(nn.Module):
    def __init__(self):
        super(RGBBranch, self).__init__()
        # RGB processing block
        self.processing_block = RGBProcessingBlock()
        
        # Load pretrained ResNet50
        # Note: PyTorch doesn't have ResNetV2 built-in, so we use ResNet50 instead
        self.resnet = models.resnet50(pretrained=True)
        
        # Regressor 
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 256)
        self.dropout2 = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(256, 4)  # Outputs: FreshWeight, DryWeight, Diameter, Height
        
    def forward(self, x):
        # Process RGB image
        x = self.processing_block(x)
        
        # ResNet feature extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Regressor feature extraction
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class DepthProcessingBlock(nn.Module):
    def __init__(self):
        super(DepthProcessingBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
    
class DepthBranch(nn.Module):
    def __init__(self, pretrained=False):
        super(DepthBranch, self).__init__()
        self.processing_block = DepthProcessingBlock()
        
        # Load ResNet50 without pretraining for depth
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer for single channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        
        # Remove the final classification layer (it outputs classsification labels, we need continuous/discrete values)
        # Remove resnet pooling layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global Average Pooling + Flatten
        self.flatten = nn.Flatten()
        
        # Regressor
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048,256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,1)
        
    def forward(self, x):
        x = self.processing_block(x)
        x = self.resnet(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
class FirstStage(nn.Module):
    """Joint Regressor that generates input map for the SecondStage of the model architecture"""
    def __init__(self):
        super(FirstStage, self).__init__()
        
        self.rgb_branch = RGBBranch()
        self.depth_branch = DepthBranch()

        # Joint regressor (takes concatenated outputs)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(5, 2048)      # Inputs: RGB(4) + Depth(1) = 5 inputs
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.dropout3 = nn.Dropout(0.5)
        
        # Final outputs
        self.output1 = nn.Linear(2048, 3)  # FreshWeight, DryWeight, Diameter
        self.output2 = nn.Linear(2048, 1)  # Height only


    def forward(self, rgb, depth):
        # Get outputs from individual branches
        rgb_output = self.rgb_branch(rgb)      # [batch, 4]
        depth_output = self.depth_branch(depth) # [batch, 1]
        # Concatenate outputs (rgb and depth)
        combined = torch.cat([rgb_output, depth_output], dim=1)  # [batch, 5]
        
        # Process through joint regressor
        x = self.dropout1(combined)
        x = F.relu(self.fc1(x))            # [batch, 2048]
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))            # [batch, 2048]
        x = self.dropout3(x)
        
        # Generate final outputs
        output1 = self.output1(x)          # [batch, 3]
        output2 = self.output2(x)          # [batch, 1]

        input_map = torch.cat([output1, output2], dim=1)
        
        return {
            'input_map': input_map, # Output1 + Output2 (FreshWeight, DryWeight, Diameter, Height)
            'output1': output1,  # FreshWeight, DryWeight, Diameter
            'output2': output2,  # Height
            'rgb_output': rgb_output,    # Individual RGB predictions
            'depth_output': depth_output  # Individual depth predictions
        }
    
class SecondStageRegressor(nn.Module):
    def __init__(self):
        super(SecondStageRegressor, self).__init__()

        self.fc1 = nn.Linear(4, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048,2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048,1)

    def forward(self):

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

    
class SecondStage(nn.Module):
    def __init__(self):
        super(SecondStage, self).__init__()

        self.dry_weight_regressor = SecondStageRegressor()
        self.leaf_area_regressor = SecondStageRegressor()

    def forward(self, input_map):

        refined_dry_weight = self.dry_weight_regressor(input_map) # output 3
        leaf_area = self.leaf_area_regressor(input_map) # output 4

        return {'refined_dry_weight': refined_dry_weight, 'leaf_area': leaf_area}
    


class TwoStageModel(nn.Model):
    def __init__(self):
        super(TwoStageModel).__init__()
        self.first_stage = FirstStage()
        self.second_stage = SecondStage()

    def forward(self, rgb, depth):
        stage1_outputs = self.first_stage(rgb,depth)
        stage2_outputs = self.second_stage(stage1_outputs['input_map'])

        return {**stage1_outputs, **stage2_outputs}




if __name__ == "__main__":
    print(__name__)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
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
    
if __name__ == "__main__":
    print(__name__)
    
    # Preprocesor
    preproc = Preprocessing()

    # Replace zeros (0) in depth images. Zero means sensor failed to read properly the depth at a given point in the image.
    filled_depth_dir = "../DepthImagesNoZeros"
    if not os.path.exists(filled_depth_dir):
        os.makedirs(filled_depth_dir)
        depth_images = os.listdir("../DepthImages/")
        for img in depth_images:
            no_zeros_img = preproc.knn_fill_depth(img)
            

    
   
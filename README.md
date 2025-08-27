## <b> Lettuce Biomass Estimation – RGB & Depth Image </b>

### <b> Overview:</b>
This repository provides a comprehensive end-to-end pipeline for automated lettuce biomass estimation using computer vision and deep learning techniques. The system leverages both RGB and depth image modalities to achieve accurate plant biometry measurements through a two-stage (CNN-based) architecture.


### <b> Model Architecture: </b>
<img src="notebook_test/images/model_arch.png" alt="Architecture Diagram" width="30%">

This implementation is based on the methodology described in this [paper](https://www.mdpi.com/1424-8220/22/15/5499).

### <b> Project structure: </b>
```
src/
├── coord_script.py         # CV2 based script to manually pinpoint the center of the RGB and Depth images for data processing purposes (coordinates are saved to a JSON file)
├── model.py                # 2 stage CNN base model architecture
├── train.py                # Model training loop (Metrics and logs are automatically stored in a designated folder that is created upon initiating the training session)
└── utils.py                # Some visualization, preprocessing and metric extraction functions (will move these function to other files)


Coordinates/
├── coordinates.json        # Resulting JSON file from the coord_script.py script


DepthImages/ # Point Clouds, saved as tensors after appropriate processing, and then saved as PNG images 
├── Depth_1.png 
(...) 
├── Depth_391.png  



RGBImages/ # Top-down RGB images that pair with the corresponding Depth image
├── RGB_1.png 
(...) 
├── RGB_391.png

GroundTruths/ # Biometrics ground truths (Fresh Weight, Dry Weight, Diameter, Height, Leaf Area)
├── GroundTruths/GroundTruth_All_388_Images.json


metrics/ # Automatically created metrics and model checkpoints folder for each training session, proprely time-stamped, divided by model stage with accompanying graphics
├──RGB_Mon_Aug_25_12:16:41_2025
    ├── checkpoints/
    ├── visuals/



```
Note: The RGB and depth images are not included in this repository due to size constraints. However, the complete dataset used for training and validation is publicly available here: [dataset](link)
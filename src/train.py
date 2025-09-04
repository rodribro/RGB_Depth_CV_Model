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
from dataset import PlantDataset 

class Trainer:
    
    def __init__(self, model, num_epochs, batch_size, lr, device, augment_multiplier):

        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.augment_multiplier = augment_multiplier

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    
    def end_to_end_train(self):

        plant_dataset = PlantDataset(
            rgb_dir="../RGBImages", 
            depth_dir="../DepthImages", 
            coords_json_path="../Coordinates/coordinates.json",
            ground_truths_json_path="../GroundTruths/GroundTruth_All_388_Images.json",
            augment=True,
            rgb_only=False,
            augment_multiplier=self.augment_multiplier
        )

        train_loader, val_loader = create_data_loaders(plant_dataset)
        return 0

    def create_data_loaders(self, dataset):

        return 0
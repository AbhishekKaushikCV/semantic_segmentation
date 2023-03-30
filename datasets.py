from typing import Tuple, List
import torch
from torch.utils.data import Dataset

import numpy as np
import glob
import os


class KittiDataset(Dataset):
    """
    A PyTorch dataset class for loading KITTI semantic segmentation data.

    Args:
    - features_dir: A string representing the directory path where the feature files are stored.
    - labels_dir: A string representing the directory path where the label files are stored.
    - num_classes: An integer representing the number of classes in the dataset.
    """
    def __init__(self, features_dir: str, labels_dir: str, num_classes: int):
        self.num_classes = num_classes
        feature_files = sorted(glob.glob(os.path.join(features_dir, '*.pt')))
        self.features =[torch.load(f) for f in feature_files]  #torch.Size([13,720, 128, 256])
        label_files = sorted(glob.glob(os.path.join(labels_dir, '*.pt')))
        self.labels = [torch.load(f) for f in label_files]  #torch.Size([13,512, 1024])
        self.labels = self.create_masks(self.labels, num_classes)
    
    def create_masks(self, labels: List[np.ndarray], num_classes: int) -> List[np.ndarray]:
        """
        Preprocesses the labels by creating masks for each class.

        Args:
        - labels: A list of numpy arrays representing the label masks.
        - num_classes: An integer representing the number of classes in the dataset.

        Returns:
        - masks_labels: A list of numpy arrays representing the masks for each class.
        """
        masks_labels = []
        for label in labels:
            label[label == -1] = num_classes - 1 # map the unlabelled to 19 index
            print(np.unique(label))
            masks = np.zeros((num_classes,512,1024))
            for i in range(num_classes):
                mask = np.zeros((512,1024))
                mask[label == i] = 1
                masks[i] = mask
            masks_labels.append(masks)
        return masks_labels
    
    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - length: An integer representing the length of the dataset.
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single item from the dataset.

        Args:
        - idx: An integer representing the index of the item to retrieve.

        Returns:
        - features: A PyTorch tensor representing the features.
        - labels: A PyTorch tensor representing the labels.
        """
        features = torch.tensor(self.features[idx])   #torch.Size([720, 128, 256])
        labels = torch.tensor(self.labels[idx]).float()  #torch.Size([512, 1024]) # float()
        return features, labels

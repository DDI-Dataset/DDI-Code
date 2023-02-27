"""Code for loading DDI Dataset."""

from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import os
import pandas as pd
import numpy as np

means = [0.485, 0.456, 0.406]
stds  = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    lambda x: x.convert('RGB'),
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=means, std=stds)
])


class DDI_Dataset(ImageFolder):
    _DDI_download_link = "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
    """DDI Dataset.

    Note: assumes DDI data is organized as
        ./DDI
            /images
                /000001.png
                /000002.png
                ...
            /ddi_metadata.csv

    (After downloading from the Stanford AIMI repository, this requires moving all .png files into a new subdirectory titled "images".)

    Args:
        root     (str): Root directory of dataset.
        csv_path (str): Path to the metadata CSV file. Defaults to `{root}/ddi_metadata.csv`
        transform     : Function to transform and collate image input. (can use test_transform from this file) 
    """
    def __init__(self, root, csv_path=None, download=True, transform=None, *args, **kwargs):
        if csv_path is None:
            csv_path = os.path.join(root, "ddi_metadata.csv")
        if not os.path.exists(csv_path) and download:
            raise Exception(f"Please visit <{DDI_Dataset._DDI_download_link}> to download the DDI dataset.")
        assert os.path.exists(csv_path), f"Path not found <{csv_path}>."
        super(DDI_Dataset, self).__init__(root, *args, transform=transform, **kwargs)
        self.annotations = pd.read_csv(csv_path)
        m_key = 'malignant'
        if m_key not in self.annotations:
            self.annotations[m_key] = self.annotations['malignancy(malig=1)'].apply(lambda x: x==1)

    def __getitem__(self, index):
        img, target = super(DDI_Dataset, self).__getitem__(index)
        path = self.imgs[index][0]        
        annotation = dict(self.annotations[self.annotations.DDI_file==path.split("/")[-1]])
        target = int(annotation['malignant'].item()) # 1 if malignant, 0 if benign
        skin_tone = annotation['skin_tone'].item() # Fitzpatrick- 12, 34, or 56
        return path, img, target, skin_tone

    """Return a subset of the DDI dataset based on skin tones and malignancy of lesion.

    Args:
        skin_tone    (list of int): Which skin tones to include in the subset. Options are {12, 34, 56}.
        diagnosis    (list of str): Include malignant and/or benign images. Options are {"benign", "malignant"}
    """
    def subset(self, skin_tone=None, diagnosis=None):
        skin_tone = [12, 34, 56] if skin_tone is None else skin_tone
        diagnosis = ["benign", "malignant"] if diagnosis is None else diagnosis
        for si in skin_tone: 
            assert si in [12,34,56], f"{si} is not a valid skin tone"
        for di in diagnosis: 
            assert di in ["benign", "malignant"], f"{di} is not a valid diagnosis"
        indices = np.where(self.annotations['skin_tone'].isin(skin_tone) & \
                           self.annotations['malignant'].isin([di=="malignant" for di in diagnosis]))[0]
        return Subset(self, indices)
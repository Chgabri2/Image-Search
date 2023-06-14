import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import requests
from PIL import Image 
import PIL 
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode




class Amazon_products(Dataset):
    """Amazon_products dataset."""

    def __init__(self, parquet_file, root_dir, transform=None):
        """
        Arguments:
            parquet_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_parquet(parquet_file, engine='pyarrow')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get the data from the dataframe
        url = self.dataframe['primary_image'].iloc[idx]
        asin= self.dataframe['asin'].iloc[idx]
        description= self.dataframe['title'].iloc[idx]

        #get image from url
        response = requests.get(url)
        with open(str(asin)+'.jpg', 'wb') as image:
          image.write(response.content) 

        #open the image
        image= Image.open(str(asin)+'.jpg')
        plt.imshow(image)
        sample = {'image': image, 'asin': asin, 'description':description, 'url': url}

        #apply transormation to tensor
        if self.transform:
             transform = transforms.Compose([ transforms.ToTensor()])
             sample = {'tensor':transform(image), 'image': image, 'asin': asin, 'description':description, 'url': url}
             return sample


        return sample

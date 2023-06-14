from  dataset import Amazon_products
import requests
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import numpy as np
import pandas as pd
import sys


#importing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
print("model loaded")
amazon_dataset = Amazon_products(parquet_file= 'data/product_images.parquet', root_dir='data/images/', transform = True)
dataloader = DataLoader(amazon_dataset, batch_size=128, shuffle=True, num_workers=4)

def emmbed_images(idx):
    sample= amazon_dataset.__getitem__(idx)
#get encoding
    image = preprocess(sample['image']).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    return {'asin':sample['asin'], 'image_features':image_features}


   

def main():
# read input from command line
    i=  int(sys.argv[1])
    k=  int(sys.argv[2])

# create a pool of workers
    pool = Pool()
    urls= amazon_dataset.dataframe['primary_image'][i:i+k]
    idxs=amazon_dataset.dataframe['asin'][i:i+k]
    tuples = zip(urls,idxs)
    dicts= pool.starmap(emmbed_images, tuples)
    print("done")
# save the results  
    df = pd.DataFrame.from_dict(dicts)
    df.to_csv(f'df_{i}_{i+k}.csv')
    




if __name__ == "__main__":
    main()
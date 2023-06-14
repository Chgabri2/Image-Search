from  dataset import Amazon_products
import requests
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import numpy as np
import pandas as pd

#importing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)




def emmbed_images(url,asin):
    # get image from url
    response = requests.get(url)
    with open(str(asin)+'.jpg', 'wb') as handle:
        handle.write(response.content)
    image= Image.open(str(asin)+'.jpg')

    #get encoding
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    return {'asin':asin, 'image_features':image_features}


amazon_dataset = Amazon_products(parquet_file= 'data/product_images.parquet', root_dir='data/images/', transform = True)
dataloader = DataLoader(amazon_dataset, batch_size=128, shuffle=True, num_workers=4)
#for i_batch, sample_batched in enumerate(dataloader):
 #   pass


pool = Pool()
urls= amazon_dataset.dataframe['primary_image'][0:5]
idxs=amazon_dataset.dataframe['asin'][0:5]
tuples = zip(urls,idxs)
dicts= pool.starmap(emmbed_images, tuples)

df = pd.DataFrame.from_dict(dicts)    
print(df)

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool,freeze_support
import requests
from PIL import Image 
import PIL 
from concurrent.futures import ThreadPoolExecutor
import sys

def download(url,idx):
    response = requests.get(url)
    path='data/images/'+ str(idx)+'.jpg'
    with open(path, 'wb') as handle:
        handle.write(response.content)


def main():
    args = int(sys.argv[1])
    # enable support for multiprocessing
    freeze_support()
    result = pd.read_parquet('data/product_images.parquet', engine='pyarrow')
    pool = Pool()
    urls=result['primary_image'][args:args+2000]
    idxs=result['asin'][args:args+2000]
    tuples = zip(urls,idxs)
    pool.starmap(download, tuples)

# protect the entry point
if __name__ == '__main__':
   main()


#with ThreadPoolExecutor(max_workers=8) as executor:
 #   executor.map(download, tuples) #urls=[list of url]
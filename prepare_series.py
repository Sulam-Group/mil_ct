import os
import json
import numpy as np
import torch
import threading
import torchvision.transforms.functional as TF
from tqdm import tqdm
from dataset import Series
from multiprocessing import Pool

data_dir = "data"
series_dir = os.path.join(data_dir, "series")
images_dir = os.path.join(data_dir, "stage_2_train")

clean_series_path = os.path.join(data_dir, "clean_series.json")
with open(clean_series_path, "r", encoding="utf-8") as f:
    clean_series = json.load(f)
series_ids = list(clean_series.keys())
series_count = len(series_ids)
n = 40
chunk_size = series_count // n
l = range(0, series_count, chunk_size)


def prepare_series(series_id, pbar):
    series_path = os.path.join(series_dir, series_id)
    if os.path.exists(series_path):
        try:
            images = torch.load(series_path)
        except:
            series = clean_series[series_id]
            series_images = np.array(series["series"])

            images = series_images[:, 0]
            images = [np.load(os.path.join(images_dir, f"ID_{u}.npy")) for u in images]
            images = torch.Tensor(images).float().unsqueeze(1).repeat(1, 3, 1, 1)
            images = TF.normalize(
                images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            torch.save(images, series_path)
            read = False
            while read == False:
                try:
                    torch.load(series_path)
                    read = True
                except:
                    torch.save(images, series_path)
        finally:
            pbar.update(1)
    return 0


def f(l):
    chunk = (
        series_ids[l : l + chunk_size]
        if l + chunk_size < series_count
        else series_ids[l:]
    )
    with tqdm(total=len(chunk)) as pbar:
        return [prepare_series(u, pbar) for u in chunk]


with Pool(n) as p:
    p.map(f, l)
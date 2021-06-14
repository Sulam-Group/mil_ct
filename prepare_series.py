import os
import shutil
import numpy as np
import json
import pandas as pd
import time
from PIL import Image
from torch._C import dtype
from tqdm import tqdm
from pydicom import dcmread
from multiprocessing import Pool

data_dir = "/export/gaon1/data/jteneggi/data/rsna-intracranial-hemorrhage-detection"


def z_coord(op_dir, image):
    image_path = os.path.join(op_dir, f"ID_{image}.dcm")
    ds = dcmread(image_path)
    image_position = ds[0x20, 0x32].value
    return image_position[-1]


def get_series_id(op_dir, u, pbar):
    img_path = os.path.join(op_dir, f"ID_{u}.dcm")
    ds = dcmread(img_path)
    series_id = ds["SeriesInstanceUID"].value
    # series_id = ds["StudyInstanceUID"].value
    # series_id = ds["PatientID"].value
    pbar.update(1)
    return series_id


def sort_series(op_dir, series, pbar):
    series_images = series[:, 0]
    series_labels = series[:, 1]
    series_target = series_labels.astype(int).any()

    series_z = np.fromiter((z_coord(op_dir, im) for im in series_images), dtype=float)
    sorted_series_idx = np.argsort(series_z)
    sorted_series = series[sorted_series_idx]

    pbar.update(1)
    return {"series": sorted_series.tolist(), "target": str(series_target.astype(int))}


op = "train"
op_dir = os.path.join(data_dir, f"stage_2_{op}")
op_csv = os.path.join(data_dir, f"stage_2_{op}.csv")

op_df = pd.read_csv(op_csv)

img_ids = op_df["ID"].values.astype(str)
img_labels = op_df["Label"].values

any_rows = np.arange(5, len(op_df), 6)
any_ids = img_ids[any_rows]
any_labels = img_labels[any_rows]
print(f"There are {sum(any_labels)}/{len(any_labels)} images with hemorrhage.")

ids = np.fromiter((u.split("_")[1] for u in any_ids), dtype=any_ids.dtype)

L = 20000
ids = ids[:L]
any_labels = any_labels[:L]

n = len(ids) // 10
l = list(range(0, len(ids), n))


def f(l):
    _ids = ids[l : l + n] if l + n < len(ids) else ids[l:]
    with tqdm(total=len(_ids)) as pbar:
        return np.fromiter(
            (get_series_id(op_dir, u, pbar) for u in _ids),
            dtype=ids.dtype,
        )


t0 = time.time()
p = Pool(len(l))
series_ids = p.map(f, l)
p.close()
p.join()
series_ids = np.concatenate(series_ids)
t = time.time()
print(f"Read all series ids in {np.around(t - t0, 4)} s")

zipped_labels = np.stack((ids, any_labels), axis=1)
u_series_ids, u_series_inverse, u_series_counts = np.unique(
    series_ids, return_inverse=True, return_counts=True
)

n = len(u_series_ids) // 10
l = list(range(0, len(u_series_ids), n))


def g(l):
    _ids = u_series_ids[l : l + n] if l + n < len(u_series_ids) else u_series_ids[l:]
    with tqdm(total=len(_ids)) as pbar:
        return {
            s_id.split("_")[1]: sort_series(
                op_dir, zipped_labels[u_series_inverse == l + i], pbar
            )
            for i, s_id in enumerate(_ids)
        }


t0 = time.time()
p = Pool(len(l))
series_dictionaries = p.map(g, l)
p.close()
p.join()
series_dictionary = {}
for u in series_dictionaries:
    series_dictionary = {**series_dictionary, **u}
t = time.time()
print(f"Sorted {len(series_dictionary)} unique series in {np.around(t - t0, 4)} s")

with open(os.path.join(data_dir, "raw_series.json"), "w", encoding="utf-8") as f:
    json.dump(series_dictionary, f, ensure_ascii=False, indent=4)

positive_series = []
negative_series = []

for series in series_dictionary:
    if series_dictionary[series]["target"]:
        positive_series.append(series)
    else:
        negative_series.append(series)

print(f"There are {len(positive_series)} positive and {len(negative_series)} negative series")

keys = np.array(list(series_dictionary.keys()))

split = 0.8
total_len = len(keys)
train_len = int(split * total_len)

train_idx = np.random.choice(range(total_len), train_len, replace=False)
val_idx = np.setxor1d(np.arange(total_len), train_idx)

train_series = {key: series_dictionary[key] for key in keys[train_idx]}
val_series = {key: series_dictionary[key] for key in keys[val_idx]}

with open(os.path.join(data_dir, "train_series.json"), "w", encoding="utf-8") as f:
    json.dump(train_series, f, ensure_ascii=False, indent=4)

with open(os.path.join(data_dir, "val_series.json"), "w", encoding="utf-8") as f:
    json.dump(val_series, f, ensure_ascii=False, indent=4)

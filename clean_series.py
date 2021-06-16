import json
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

data_dir = "data"
train_dir = os.path.join(data_dir, "stage_2_train")
sorted_series_path = os.path.join(data_dir, "raw_series.json")


def filter_size(image):
    image, _ = image
    img_path = os.path.join(train_dir, f"ID_{image}.npy")
    try:
        img = np.load(img_path)
    except:
        return False
    else:
        return img.shape == (512, 512)


def clean_series(series, pbar):
    series_images = series["series"]
    clean_series = list(filter(filter_size, series_images))
    clean_series = np.array(clean_series)
    if clean_series.ndim == 1:
        pbar.update(1)
        return None
    else:
        pbar.update(1)
        return {
            "series": clean_series.tolist(),
            "target": str(clean_series[:, 1].astype(int).any().astype(int)),
        }


with open(sorted_series_path, "r", encoding="utf-8") as f:
    sorted_series = json.load(f)
series_ids = list(sorted_series.keys())

n = len(series_ids) // 10
l = list(range(0, len(series_ids), n))


def f(l):
    _ids = series_ids[l : l + n] if l + n < len(series_ids) else series_ids[l:]
    with tqdm(total=len(_ids)) as pbar:
        return dict(
            filter(
                lambda el: el[1] is not None,
                {u: clean_series(sorted_series[u], pbar) for u in _ids}.items(),
            )
        )


p = Pool(len(l))
results = p.map(f, l)
p.close()
p.join()
clean_series = {}
for u in results:
    clean_series = {**clean_series, **u}
print(f"There are {len(clean_series)} clean series.")

with open(os.path.join(data_dir, "clean_series.json"), "w", encoding="utf-8") as f:
    json.dump(clean_series, f, ensure_ascii=False, indent=4)
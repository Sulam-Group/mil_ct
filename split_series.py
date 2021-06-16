import json
import os
import numpy as np
from tqdm import tqdm

data_dir = "data"
train_dir = os.path.join(data_dir, "stage_2_train")
clean_series_path = os.path.join(data_dir, "clean_series.json")

with open(clean_series_path, "r", encoding="utf-8") as f:
    clean_series = json.load(f)
series_ids = list(clean_series.keys())

series_labels = {"0": [], "1": []}

for series_id in tqdm(series_ids):
    series_labels[clean_series[series_id]["target"]].append(series_id)
print(
    f"There are {len(series_labels['0'])}/{len(series_labels['1'])} negative/positive series"
)

ratio = 0.8
train_series_count = {k: int(ratio * len(series_labels[k])) for k in ["0", "1"]}
train_series = {
    k: np.random.choice(series_labels[k], train_series_count[k], replace=False)
    for k in ["0", "1"]
}
val_series = {k: np.setxor1d(series_labels[k], train_series[k]) for k in ["0", "1"]}
print(
    f"There are {len(train_series['0'])}/{len(train_series['1'])} negative/positive series in the training split."
)
print(
    f"There are {len(val_series['0'])}/{len(val_series['1'])} negative/positive series in the validation split."
)

train_series_ids = np.concatenate(tuple(train_series.values()))
np.random.shuffle(train_series_ids)
train_series_dict = {}
for series_id in train_series_ids:
    train_series_dict[series_id] = clean_series[series_id]
assert (
    len(train_series_dict)
    == len(train_series_ids)
    == sum([len(u) for u in train_series.values()])
)
with open(os.path.join(data_dir, "train_series.json"), "w", encoding="utf-8") as f:
    json.dump(train_series_dict, f, ensure_ascii=False, indent=4)

val_series_ids = np.concatenate(tuple(val_series.values()))
np.random.shuffle(val_series_ids)
val_series_dict = {}
for series_id in val_series_ids:
    val_series_dict[series_id] = clean_series[series_id]
assert (
    len(val_series_dict)
    == len(val_series_ids)
    == sum([len(u) for u in val_series.values()])
)
with open(os.path.join(data_dir, "val_series.json"), "w", encoding="utf-8") as f:
    json.dump(val_series_dict, f, ensure_ascii=False, indent=4)

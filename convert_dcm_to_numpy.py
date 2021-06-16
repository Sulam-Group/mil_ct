import os
from pydicom import dcmread
from tqdm import tqdm
import numpy as np

data_dir = "data"
train_dir = os.path.join(data_dir, "stage_2_train")
test_dir = os.path.join(data_dir, "stage_2_test")

stage_dirs = [train_dir, test_dir]

for stage_dir in stage_dirs:
    for root, _, images in os.walk(stage_dir):
        images = list(filter(lambda x: ".dcm" in x, images))
        for image in tqdm(images):
            img_id = image.split(".")[0]
            out_path = os.path.join(root, f"{img_id}.npy")
            if True:
                # if os.path.exists(out_path) is False:
                try:
                    ds = dcmread(os.path.join(root, image))
                    img = ds.pixel_array
                    np.save(out_path, img, allow_pickle=True)
                except:
                    print(f"failed {image}")

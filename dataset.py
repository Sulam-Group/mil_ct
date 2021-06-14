import os
import json
from numpy.core.fromnumeric import _transpose_dispatcher
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataset_provider import DatasetProviderInterface


class hemorrhage_dataset(Dataset):
    def __init__(
        self,
        data_dir="/export/gaon1/data/jteneggi/data/rsna-intracranial-hemorrhage-detection",
        op="train",
    ):
        series_path = os.path.join(data_dir, f"{op}_series.json")
        self.op_dir = os.path.join(data_dir, "stage_2_train")
        with open(series_path, "r", encoding="utf-8") as f:
            self.series_dictionary = json.load(f)
        self.series_ids = list(self.series_dictionary.keys())

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_idx = self.series_ids[idx]
        series_obj = self.series_dictionary[series_idx]

        series = series_obj["series"]
        target = series_obj["target"]

        series = np.array(series)
        images = series[:, 0]
        labels = series[:, 1]

        series = [np.load(os.path.join(self.op_dir, f"ID_{u}.npy")) for u in images]

        series = torch.Tensor(series).float().unsqueeze(1).repeat(1, 3, 1, 1)
        series = TF.normalize(
            series, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        target = torch.Tensor([int(target)]).long()
        labels = torch.Tensor([labels.astype(int)]).long()

        return series, target, labels


class SeriesBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.series = torch.cat(transposed_data[0], dim=0)
        self.targets = torch.cat(transposed_data[1], dim=0)
        self.labels = torch.cat(transposed_data[2], dim=1).squeeze()
        limits = [len(s) for s in transposed_data[0]]
        self.limits = np.cumsum(limits)
        assert self.labels.size(0) == self.limits[-1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.series = self.series.pin_memory()
        self.targets = self.targets.pin_memory()
        self.labels = self.labels.pin_memory()
        # self.limits = self.limits.pin_memory()
        return self


def create_hemorrhage_dataset(args, data_dir, op, num_workers, train_batch_size):
    op_data = hemorrhage_dataset(data_dir=data_dir, op=op)
    args.logger.info(f"Found {len(op_data)} series for op {op}")

    def collate_wrapper(batch):
        return SeriesBatch(batch)

    op_loader = DataLoader(
        op_data,
        shuffle=op == "train",
        batch_size=train_batch_size if op == "train" else 1,
        num_workers=num_workers,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )
    return op_loader


class HemorrhageDatasetProvider(DatasetProviderInterface):
    def __init__(self, args):
        self.num_workers = args.config["training"]["num_workers"]
        self.train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
        self.logger = args.logger

        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.data_dir = args.config["data"]["data_dir"]
        self.ops = args.config["data"]["ops"]
        self.dataloaders = {
            op: create_hemorrhage_dataset(
                args,
                self.data_dir,
                op,
                self.num_workers,
                self.train_micro_batch_size_per_gpu,
            )
            for op in self.ops
        }

        if self.global_rank == 0:
            self.logger.info(
                f"Initialized HemorrhageDatasetProvider with ops: {self.ops}"
            )

    def get_dataloaders(self):
        return self.dataloaders

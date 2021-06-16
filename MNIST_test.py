import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import torchvision
import numpy as np
import time
from model import HemorrhageDetector
from torchvision import transforms
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

model = HemorrhageDetector(n_dim=1024)
model = model.to(device)
model.train()

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

t = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="data", download=True, transform=t, train=True
)
test_dataset = torchvision.datasets.MNIST(
    root="daata", download=True, transform=t, train=False
)

dataloaders = {
    "train": DataLoader(train_dataset, shuffle=True, batch_size=32),
    "test": DataLoader(test_dataset, shuffle=False, batch_size=32),
}

digit = 5
num_epochs = 25
for epoch in range(num_epochs):
    print(f"Started epoch {epoch+1}/{num_epochs}")
    for op in ["train", "test"]:

        if op == "train":
            model.train()
        else:
            model.eval()

        op_size = 0.
        epoch_loss = 0.0
        epoch_corrects = 0.

        dataloader = dataloaders[op]
        for data in tqdm(dataloader):
            images, labels = data

            digit_indexes = (labels == digit).nonzero()
            other_digits_indexes = (labels != digit).nonzero()

            if len(digit_indexes) == 0:
                target = torch.Tensor([0]).float()
            else:
                if torch.rand(1) < 0.50:
                    images = images[other_digits_indexes.flatten()]
                    target = torch.Tensor([0]).float()
                else:
                    target = torch.Tensor([1]).float()
                    # n = int(torch.rand(1).item() * len(digit_indexes))
                    # if n > 0:
                    #     digit_indexes = np.random.choice(
                    #         digit_indexes.flatten(), n, replace=False
                    #     )
                    #     indexes = np.concatenate(
                    #         (digit_indexes, other_digits_indexes.flatten())
                    #     )
                    #     images = torch.index_select(images, 0, torch.from_numpy(indexes))
                    #     target = torch.Tensor([1]).float()
                    # else:
                    #     images = images[other_digits_indexes.flatten()]
                    #     target = torch.Tensor([0]).float()

            images = images.repeat(1, 3, 1, 1)
            images = TF.normalize(
                images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            images = images.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(images).unsqueeze(0)
            output = torch.clamp(output, min=1e-5, max=1.0 - 1e-5)
            prediction = (output > 0.5).float()
            loss = criterion(output, target)
            # print(
            #     f"Target: {target.item()}, prediction: {int(output.item() > 0.5)}, loss: {loss.item()}"
            # )
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_corrects += torch.sum(prediction == target)
            op_size += 1

        epoch_loss /= op_size
        epoch_corrects /= op_size

        print(f"{op} Loss {epoch_loss:4f} Acc: {epoch_corrects:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HemorrhageDetector(nn.Module):
    def __init__(self, n_dim=1024):
        super(HemorrhageDetector, self).__init__()
        self.n_dim = n_dim
        self.encoder = self.__encoder__()
        self.attention_mechanism = self.__attention_mechanism__()
        self.classifier = self.__classifier__()

    def __encoder__(self):
        # maybe freeze convolutional layers of resnet,
        # leave fully connected ones
        encoder = models.resnet18(pretrained=True)

        def freeze_convolutional_layers(m):
            if type(m) == nn.Conv2d:
                for p in m.parameters():
                    p.requires_grad = False

        encoder.apply(freeze_convolutional_layers)

        # inplanes = 64
        # # input has only one channel
        # encoder.conv1 = nn.Conv2d(
        #     1, inplanes, kernel_size=7, stride=2, padding=3, bias=False
        # )
        # encode to n_dim features
        num_ftrs = encoder.fc.in_features
        encoder.fc = nn.Linear(num_ftrs, self.n_dim)
        return encoder

    def __attention_mechanism__(self):
        hidden_width = 64

        return nn.Sequential(
            nn.Linear(self.n_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    def __classifier__(self):
        return nn.Sequential(nn.Linear(self.n_dim, 2), nn.Sigmoid())

    def forward(self, x, limits):
        H = self.encoder(x)
        A = self.attention_mechanism(H).t()
        x = torch.stack(
            [
                self.classifier(
                    torch.mm(F.softmax(A[:, l : limits[i]], dim=1), H[l : limits[i]])
                )
                for i, l in enumerate([0] + limits[:-1].tolist())
            ],
            dim=0,
        )
        return x.squeeze()

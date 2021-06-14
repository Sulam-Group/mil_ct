import os
import torch
from model import HemorrhageDetector

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

encoder, attention_mechanism, classifier = HemorrhageDetector.load(n_dim=128)

input = torch.empty([40, 512, 512])
output = encoder(input.unsqueeze(1))
print(output.size())
a = attention_mechanism(output)
print(a.size())

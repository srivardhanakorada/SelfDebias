import torch

h = torch.load("data/h/00001.pt")
print(len(h),h[0].shape)
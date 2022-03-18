from torch.utils.data import Dataset
import numpy as np
import torch


class ImagesDataset(Dataset):

	def __init__(self, source_root, source_label_root):
		self.latents = np.load(source_root)
		self.label = np.load(source_label_root)
		self.length = len(self.latents)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		latent = torch.tensor(self.latents[index])
		label = self.label[index]
		label = label[1:]
		label = list(map(int, label))
		label = torch.tensor(label)


		return latent, label

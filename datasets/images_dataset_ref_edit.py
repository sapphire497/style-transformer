from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch


class ImagesDataset(Dataset):

	def __init__(self, source_img_root, source_w_root, source_transform=None, training=True):
		source_paths = sorted(data_utils.make_dataset(source_img_root))
		self.source_transform = source_transform
		latents = np.load(source_w_root)

		train_len = int(0.9 * len(latents))
		if training:
			self.source_paths = source_paths[:train_len]
			self.latents = latents[:train_len]
		else:
			self.source_paths = source_paths[train_len:]
			self.latents = latents[train_len:]

		self.length = len(self.latents)

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		src_path = self.source_paths[index]
		src_img = Image.open(src_path)
		src_img = src_img.convert('RGB')
		src_img = self.source_transform(src_img)

		latent = torch.tensor(self.latents[index])

		return src_img, latent

import os
import matplotlib

matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils import common
from datasets.images_dataset_label_edit import ImagesDataset
from models.encoders.edit_transformer_encoders import LCNet_40
from models.stylegan2.model import Generator

attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0
		self.n_epochs = 1

		self.device = 'cuda'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		# Initialize network
		self.classifier = LCNet_40([9216, 2048, 512], n_classes=40, activ='leakyrelu').to(self.device)
		ckpt = torch.load((self.opts.classifier_weights), map_location='cpu')
		self.classifier.load_state_dict(ckpt['state_dict'])
		self.generator = Generator(self.opts.output_size, 512, 8).to(self.device)
		ckpt = torch.load((self.opts.stylegan_weights), map_location='cpu')
		self.generator.load_state_dict(ckpt['g_ema'], strict=False)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		self.attr_type = attr_dict['Male'] #Bangs=5, Eyeglasses=15, Male=20

		# Initialize dataset, including w and 40-class label
		self.train_dataset = ImagesDataset(source_root='test_w.npy',
									  	   source_label_root='test_attr.npy')
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=False,
										   num_workers=int(self.opts.workers),
										   drop_last=True)

	def train(self):
		out_path = os.path.join(self.opts.exp_dir, 'images')
		os.makedirs(out_path, exist_ok=True)
		for epoch in range(self.n_epochs):
			for batch in tqdm(self.train_dataloader):
				w, label = batch
				w, label = w.to(self.device), label.to(self.device)

				# First derivative
				# w_A = Variable(w, requires_grad=True)
				# label[label == -1] = 0
				# label[:, self.attr_type] = 0  # Edit to the specified direction.
				# pre_A, embed = self.classifier(w_A.view(w.size(0), -1))
				# loss_cls_A = F.binary_cross_entropy_with_logits(pre_A[:, self.attr_type], label[:, self.attr_type].float())
				# loss = loss_cls_A
				# loss.backward(retain_graph=True)
				# direct_ = w_A.grad
				# direct_ = direct_ / torch.norm(direct_, dim=2, keepdim=True)
				# direct = direct_

				# Hessian matrix
				a = torch.randn((w.size(0), 18, 512), device=self.device, requires_grad=True)
				zero = torch.zeros((w.size(0), 18, 512), device=self.device, requires_grad=True)
				direct = Variable(a, requires_grad=True)
				direct_zero = Variable(zero, requires_grad=True)
				w_A = Variable(w, requires_grad=True)
				w_F = w_A + 0.0001* direct
				w_F0 = w_A + direct_zero

				label[label==-1] = 0
				label[:, self.attr_type] = 1 # Edit to the specified direction.
				pre, embed_edit = self.classifier(w_F.view(w.size(0), -1))
				loss = F.binary_cross_entropy_with_logits(pre[:, self.attr_type], label[:, self.attr_type].float())
				loss.backward(retain_graph=True)
				direct_ = direct.grad

				self.classifier.zero_grad()
				pre_0, embed_edit_0 = self.classifier(w_F0.view(w.size(0), -1))
				loss = F.binary_cross_entropy_with_logits(pre_0[:, self.attr_type], label[:, self.attr_type].float())
				loss.backward(retain_graph=True)
				direct_ = direct_ - direct_zero.grad
				direct_ = direct_ / torch.norm(direct_, dim=2, keepdim=True)
				direct = direct - direct_

				with torch.no_grad():
					self.generator.eval()
					edit, _ = self.generator([w - direct * 3], return_latents=False, truncation=1, truncation_latent=None,
									  input_is_latent=True)
					edit = self.face_pool(edit)

					# Generate images.
					for i in range(self.opts.batch_size):
						img = common.tensor2im(edit[i])
						im_path = str(self.global_step).zfill(5) + '.jpg'
						im_save_path = os.path.join(out_path, os.path.basename(im_path))
						Image.fromarray(np.array(img.resize((256,256)))).save(im_save_path)
						self.global_step += 1


import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import common, train_utils
from configs import data_configs
from datasets.images_dataset_ref_edit import ImagesDataset
from models.edit_transformer import EditTransformer

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

		self.device = 'cuda'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		# Initialize network, optimizer and loss
		self.net = EditTransformer(self.opts).to(self.device)
		params = list(self.net.encoder.parameters())
		self.optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate, weight_decay=0.0005)
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		self.attr_type = attr_dict['Bangs'] # Attribute to be edit

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				img, w = batch
				img, w = img.to(self.device).float(), w.to(self.device).float()
				# Split batch into source and reference images
				B = img.size(0)
				src_w = w[:B // 2]
				ref_w = w[B // 2:]
				src_img = img[:B // 2]
				ref_img = img[B // 2:]
				mix_img, mix_w = self.net.forward(src_w, ref_w, return_images=True)
				loss, loss_dict = self.calc_loss(src_w, ref_w, mix_w, attr_type=self.attr_type)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 10000 and self.global_step % 25 == 0):
					self.parse_and_log_images(src_img, ref_img, mix_img, title='images/train')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			img, w = batch

			with torch.no_grad():
				img, w = img.to(self.device).float(), w.to(self.device).float()
				B = img.size(0)
				src_w = w[:B // 2]
				ref_w = w[B // 2:]
				src_img = img[:B // 2]
				ref_img = img[B // 2:]
				mix_img, mix_w = self.net.forward(src_w, ref_w, return_images=True)
				loss, cur_loss_dict = self.calc_loss(src_w, ref_w, mix_w, attr_type=self.attr_type)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(src_img, ref_img, mix_img,
									  title='images/test',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_img_root=dataset_args['train_edit_img_root'],
									  source_w_root=dataset_args['train_edit_w_root'],
									  source_transform=transforms_dict['transform_test'],
									  training=True)
		test_dataset = ImagesDataset(source_img_root=dataset_args['train_edit_img_root'],
									  source_w_root=dataset_args['train_edit_w_root'],
									  source_transform=transforms_dict['transform_test'],
									  training=False)
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, src, ref, mix, attr_type=0):
		loss_dict = {}
		loss = 0.0

		predict, embed_mix = self.net.classifier(mix.reshape(mix.shape[0], -1))
		_, embed_ref = self.net.classifier(ref.view(ref.shape[0], -1))  # B, n_classes, C
		_, embed_src = self.net.classifier(src.view(src.shape[0], -1))

		# Regularization on w
		loss_l2_src_code = self.mse_loss(src, mix)
		loss_dict['loss_l2_src_code'] = float(loss_l2_src_code)
		loss += loss_l2_src_code * self.opts.l2_lambda

		# Relevant attribute should be close to reference.
		attr_embed_ref = embed_ref[:, attr_type, :]
		attr_embed_mix = embed_mix[:, attr_type, :]
		# Irrelevant attributes should be close to source.
		irrel_embed_src = torch.cat([embed_src[:, :attr_type, :], embed_src[:, (attr_type+1):, :]], dim=1)
		irrel_embed_mix = torch.cat([embed_mix[:, :attr_type, :], embed_mix[:, (attr_type+1):, :]], dim=1)

		loss_l2_ref_attr = self.mse_loss(attr_embed_ref, attr_embed_mix)
		loss_dict['loss_l2_ref_attr'] = float(loss_l2_ref_attr)
		loss += loss_l2_ref_attr * self.opts.l2_ref_lambda
		loss_l2_src_attr = self.mse_loss(irrel_embed_src, irrel_embed_mix)
		loss_dict['loss_l2_src_attr'] = float(loss_l2_src_attr)
		loss += loss_l2_src_attr * self.opts.l2_src_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict


	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, src, ref, mix, title, subscript=None, display_count=1):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.tensor2im(src[i]),
				'target_face': common.tensor2im(ref[i]),
				'output_face': common.tensor2im(mix[i]),
			}
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict

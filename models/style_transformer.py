import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import style_transformer_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleTransformer(nn.Module):

	def __init__(self, opts):
		super(StyleTransformer, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = style_transformer_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		self.encoder = nn.DataParallel(self.encoder)
		self.decoder = nn.DataParallel(Generator(self.opts.output_size, 512, 8))
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading style transformer from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load((self.opts.checkpoint_path), map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			self.encoder.module.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load((self.opts.stylegan_weights), map_location='cpu')
			self.decoder.module.load_state_dict(ckpt['g_ema'], strict=False)
			# self.decoder.module.load_state_dict(get_keys(ckpt, 'decoder'), strict=True) # For cars dataset.
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt)

	def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):
		if input_code:
			codes = x
		else:
			# Get w from MLP
			z = self.encoder.module.z
			n, c = z.shape[1], z.shape[2]
			b = x.shape[0]
			z = z.expand(b, n, c).flatten(0, 1)
			query = self.decoder.module.style(z).reshape(b, n, c)
			codes = self.encoder(x, query)

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		images, result_latent = self.decoder([codes],
		                                     input_is_latent=True,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

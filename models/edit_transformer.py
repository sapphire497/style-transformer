import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders.edit_transformer_encoders import EditTransformerEncoder, LCNet_40
from models.stylegan2.model import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class EditTransformer(nn.Module):

	def __init__(self, opts):
		super(EditTransformer, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = nn.DataParallel(EditTransformerEncoder())
		self.decoder = nn.DataParallel(Generator(self.opts.output_size, 512, 8))
		self.classifier = nn.DataParallel(LCNet_40([9216, 2048, 512], n_classes=40, activ='leakyrelu'))
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()


	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load((self.opts.checkpoint_path), map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
		else:
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load((self.opts.stylegan_weights), map_location='cpu')
			self.decoder.module.load_state_dict(ckpt['g_ema'], strict=False)
			print('Loading classifier weights from pretrained!')
			ckpt = torch.load((self.opts.classifier_weights), map_location='cpu')
			self.classifier.module.load_state_dict(ckpt['state_dict'])

	def forward(self, src, ref, input_code=False, randomize_noise=True, return_latents=False, return_images=True):
		if input_code:
			codes = src
		else:
			codes = self.encoder(src, ref)

		if return_images:
			images, _ = self.decoder([codes], input_is_latent=True, randomize_noise=randomize_noise, return_latents=return_latents)
			images = self.face_pool(images)
			return images, codes
		else:
			return codes

	def set_opts(self, opts):
		self.opts = opts


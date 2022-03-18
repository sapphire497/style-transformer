from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, default='', help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default='', type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')

		self.parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=0, type=int, help='Number of test/inference dataloader workers')


	def parse(self):
		opts = self.parser.parse_args()
		return opts

'''Generate w+ with pretrained model'''

from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from models.style_transformer import StyleTransformer

def run():
    test_opts = TestOptions().parse()

    out_path_w = 'celebaha_w.npy'

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = StyleTransformer(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    global_i = 0
    latents = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            _, latent = net(input_cuda, randomize_noise=True, resize=opts.resize_outputs, return_latents=True)
            latent = latent.cpu().numpy()

        for i in range(opts.test_batch_size):
            result = latent[i]
            latents.append(result)
            global_i += 1

    np.save(out_path_w, latents)


if __name__ == '__main__':
    run()

import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from torchstat import stat
import torch.nn as nn

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from utils.common import tensor2im
from options.test_options import TestOptions
from models.edit_transformer import EditTransformer
# from models.psp_vit import pSp
import lpips
from datasets.images_dataset_edit import ImagesDataset

def run():
    test_opts = TestOptions().parse()

    out_path_results = test_opts.exp_dir
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = EditTransformer(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    # dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/bangs/testB_i',
    #                         source_label_root='/scratch/hxq/psp/celeba_attr/bangs/testB_w',
    #                         target_root='/scratch/hxq/psp/celeba_attr/bangs/testA_i',
    #                         target_label_root='/scratch/hxq/psp/celeba_attr/bangs/testA_w',
    #                         source_transform=None,
    #                         target_transform=transforms_dict['transform_test'],
    #                         opts=opts)
    # dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/lipstick/testB_i',
    #                         source_label_root='/scratch/hxq/psp/celeba_attr/lipstick/testB_w',
    #                         target_root='/scratch/hxq/psp/celeba_attr/lipstick/testA_i',
    #                         target_label_root='/scratch/hxq/psp/celeba_attr/lipstick/testA_w',
    #                         source_transform=None,
    #                         target_transform=transforms_dict['transform_test'],
    #                         opts=opts)
    # dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/beard/testB_i',
    #                         source_label_root='/scratch/hxq/psp/celeba_attr/beard/testB_w',
    #                         target_root='/scratch/hxq/psp/celeba_attr/beard/testA_i',
    #                         target_label_root='/scratch/hxq/psp/celeba_attr/beard/testA_w',
    #                         source_transform=None,
    #                         target_transform=transforms_dict['transform_test'],
    #                         opts=opts)
    # dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/gender/testB_i',
    #                         source_label_root='/scratch/hxq/psp/celeba_attr/gender/testB_w',
    #                         target_root='/scratch/hxq/psp/celeba_attr/gender/testA_i',
    #                         target_label_root='/scratch/hxq/psp/celeba_attr/gender/testA_w',
    #                         source_transform=None,
    #                         target_transform=transforms_dict['transform_test'],
    #                         opts=opts)
    # dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/gender/testA_i',
    #                         source_label_root='/scratch/hxq/psp/celeba_attr/gender/testA_w',
    #                         target_root='/scratch/hxq/psp/celeba_attr/gender/testB_i',
    #                         target_label_root='/scratch/hxq/psp/celeba_attr/gender/testB_w',
    #                         source_transform=None,
    #                         target_transform=transforms_dict['transform_test'],
    #                         opts=opts)
    # dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/mouth/testA_i',
    #                         source_label_root='/scratch/hxq/psp/celeba_attr/mouth/testA_w',
    #                         target_root='/scratch/hxq/psp/celeba_attr/mouth/testB_i',
    #                         target_label_root='/scratch/hxq/psp/celeba_attr/mouth/testB_w',
    #                         source_transform=None,
    #                         target_transform=transforms_dict['transform_test'],
    #                         opts=opts)
    dataset = ImagesDataset(source_root='/scratch/hxq/psp/celeba_attr/wavy/testA_i',
                            source_label_root='/scratch/hxq/psp/celeba_attr/wavy/testA_w',
                            target_root='/scratch/hxq/psp/celeba_attr/wavy/testB_i',
                            target_label_root='/scratch/hxq/psp/celeba_attr/wavy/testB_w',
                            source_transform=None,
                            target_transform=transforms_dict['transform_test'],
                            opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)


    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    for input_batch in tqdm(dataloader):
        # if global_i >= opts.n_images:
            # break
        with torch.no_grad():
            src_img, ref_img, src_w, ref_w = input_batch
            src_img, ref_img, src_w, ref_w = src_img.cuda().float(), ref_img.cuda().float(), \
                                             src_w.cuda().float(), ref_w.cuda().float()
            src_w_0 = src_w[0].expand(opts.test_batch_size, -1, -1)
            mix_img, mix_w = net.forward(src_w_0, ref_w, return_images=True)

            for i in range(opts.test_batch_size):
                im_path = dataset.source_paths[global_i]
                src = tensor2im(src_img[0])
                ref = tensor2im(ref_img[i])
                mix = tensor2im(mix_img[i])

                res = np.concatenate([np.array(src),
                                      np.array(ref),
                                      np.array(mix)], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_results, os.path.basename(im_path)))

                global_i += 1



if __name__ == '__main__':
    run()

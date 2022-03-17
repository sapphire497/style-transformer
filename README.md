# Style Transformer for Image Inversion and Editing (CVPR2022)
<a href="https://arxiv.org/abs/2203.07932"><img src="https://img.shields.io/badge/arXiv-2104.02699-b31b1b.svg" height=22.5></a>
> Existing GAN inversion methods fail to provide latent codes for reliable reconstruction and flexible editing simultaneously. This paper presents a transformer-based image inversion and editing model for pretrained StyleGAN which is not only with less distortions, but also of high quality and flexibility for editing. The proposed model employs a CNN encoder to provide multi-scale image features as keys and values. Meanwhile it regards the style code to be determined for different layers of the generator as queries. It first initializes query tokens as learnable parameters and maps them into $W^+$ space. Then the multi-stage alternate self- and cross-attention are utilized, updating queries with the purpose of inverting the input by the generator. Moreover, based on the inverted code, we investigate the reference- and label-based attribute editing through a pretrained latent classifier, and achieve flexible image-to-image translation with high quality results. Extensive experiments are carried out, showing better performances on both inversion and editing tasks within StyleGAN.

<p align="center">
<img src="./teaser.png" width="800px"/>
<br>
Our style transformer proposes novel multi-stage style transformer in w+ space to invert image accurately, and we characterize the image editing in StyleGAN into label-based and reference-based, and use a non-linear classifier to generate the editing vector.
</p>

## Getting Started
### Prerequisites
- Ubuntu 16.04
- NVIDIA GPU + CUDA CuDNN
- Python 3

## Pretrained Models
Coming soon!

## Training
Coming soon!

## Inference
Coming soon!

## Citation
If you use this code for your research, please cite

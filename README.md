# PhotoSketchMAN
## High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks

Lidan Wang, [Vishwanath Singagi](http://www.vishwanathsindagi.com/), [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/)

[Paper Link](https://arxiv.org/abs/1710.10182)

If you use this code for your research, please cite our paper

```
@inproceedings{wang2018high,
  title={High-quality facial photo-sketch synthesis using multi-adversarial networks},
  author={Wang, Lidan and Sindagi, Vishwanath and Patel, Vishal},
  booktitle={Automatic Face \& Gesture Recognition (FG 2018), 2018 13th IEEE International Conference on},
  pages={83--90},
  year={2018},
  organization={IEEE}
}
```

## Prerequisites
- Linux
- Python 2 or Python 3
- NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org/
- Install Torch vision from the source.
- Install python libraries [visdom](https://github.com/facebookresearch/visdom).
```bash
pip install visdom
```
### Datasets
- [CUHK Face Sketch Database (CUFS)](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html)
- [CUHK Face Sketch FERET Database (CUFSF)](http://mmlab.ie.cuhk.edu.hk/archive/cufsf/)

### Training
make sure the data folder has subfolders trainA, trainB, valA, valB, testA, testB

```bash
python PS2MAN.py --dataroot 'data/cuhkdata_augmented' --ckpt_path 'ckpt'
```


## Related Projects
[CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
[pix2pix](https://github.com/phillipi/pix2pix): Image-to-image translation with conditional adversarial nets  
[DualGAN](https://github.com/duxingren14/DualGAN): DualGAN: Unsupervised Dual Learning for Image-to-Image Translation

## Acknowledgments
This code is modified based on:

[CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

We thank the authors for their great work.  

# Pix2Pix model written in PyTorch

This repository is an implementation of model described in [Pix2Pix (Isola et al., 2016)](https://arxiv.org/pdf/1611.07004.pdf). 

The task is to translate picture to a different modality (e.g. day photo to night one) using conditional GAN. Model was trained on facades and day2night datasets. Those can be found [here](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

There are some good examples of trained models:

![alt text](https://github.com/Kirili4ik/pix2pix-pytorch/blob/main/Experiments/facades.png "facades")

![alt text](https://github.com/Kirili4ik/pix2pix-pytorch/blob/main/Experiments/d2n.png "d2n")

Also there were some discoveries and experiements, e.g.:

![alt text](https://github.com/Kirili4ik/pix2pix-pytorch/blob/main/Experiments/exp.png "augmentations-vs-no_augmentations")
On the picture above dashed line represents loss on validation set. It's clearly seen that augmentations help to avoid overfitting.

More experiements can be found in my [report](https://github.com/Kirili4ik/pix2pix-pytorch/blob/main/Experiments/Experiments.pdf) (in Russian).


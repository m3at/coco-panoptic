# coco-panoptic
Quick attempt at a panoptic segmentation model.

#WIP

# Description

This tackle the segmentation task proposed by the [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf) paper.

I take advantage of two existing implementation offering pretrained models for coco:
* [Deeplab implementation](https://github.com/kazuto1011/deeplab-pytorch) in pytorch, for the semantic segmentation part.
* [Mask RCNN implementation](https://github.com/wkentaro/chainer-mask-rcnn) in chainer, for the instance segmentation part.  
(Full credits goes to the respective authors.)

Added minor modifications to make it work for coco 2017 and with the constraints added by the panoptic task (under [toolbox](./toolbox)).

The code to combine the models into a panopric model is in `models.py`.

# Requirements
* Weights for instance and semantic models (TODO: add download script)
* Recent chainer and pytorch versions
* A bunch of smaller dependencies from repo listed above (TODO: list and trim)
* Python 3.5+

Tested on Ubuntu 18, python 3.7 with a 1070Ti.

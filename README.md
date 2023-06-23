# Deep Joint Demosaicking and Denoising
SiGGRAPH Asia 2016

Michaël Gharbi gharbi@mit.edu Gaurav Chaurasia Sylvain Paris Frédo Durand

A minimal pytorch implementation of "Deep Joint Demosaicking and Denoising" [Gharbi2016]

# Installation

From this repo:

```shell
python setup.py install
```

Using pip:

```shell
pip install demosaicnet
```

Then run the demo script with:

```shell
python scripts/demosaicnet_demo.py
```
# FAQ

- **How is noise handled? Where is the pretrained model?** The noise-aware model is not implementation, see the earlier Caffe implementation for that <https://github.com/mgharbi/demosaicnet_caffe>
- **How do I train this?** The script `scripts/train.py` is a good start to setup your training job, but I haven't tested it yet, I recommend rolling your own.

#!/usr/bin/env python
"""Demo script on using demosaicnet for inference."""

import os
from pkg_resources import resource_filename

import argparse
import numpy as np
import torch as th
import imageio

import demosaicnet

_TEST_INPUT = resource_filename("demosaicnet", 'data/test_input.png')

def main(args):
  print("Running demosaicnet demo on {}, outputing to {}".format(_TEST_INPUT, args.output))
  bayer = demosaicnet.BayerDemosaick()
  xtrans = demosaicnet.XTransDemosaick()

  # Load some ground-truth image
  gt = imageio.imread(args.input).astype(np.float32) / 255.0
  gt = np.array(gt)

  h, w, _ = gt.shape

  # Make the image size a multiple of 6 (for xtrans pattern)
  gt = gt[:6*(h//6), :6*(w//6)]


  # Network expects channel first
  gt = np.transpose(gt, [2, 0, 1])
  mosaicked = demosaicnet.bayer(gt)
  xmosaicked = demosaicnet.xtrans(gt)

  # Run the model (expects batch as first dimension)
  mosaicked = th.from_numpy(mosaicked).unsqueeze(0)
  xmosaicked = th.from_numpy(xmosaicked).unsqueeze(0)
  with th.no_grad():  # inference only
    out = bayer(mosaicked).squeeze(0).cpu().numpy()
    out = np.clip(out, 0, 1)
    xout = xtrans(xmosaicked).squeeze(0).cpu().numpy()
    xout = np.clip(xout, 0, 1)
  print("done")

  os.makedirs(args.output, exist_ok=True)
  output = args.output

  imageio.imsave(os.path.join(output, "bayer_mosaick.tif"), mosaicked.squeeze(0).permute([1, 2, 0]))
  imageio.imsave(os.path.join(output, "bayer_result.tif"), np.transpose(out, [1, 2, 0]))
  imageio.imsave(os.path.join(output, "xtrans_mosaick.tif"), xmosaicked.squeeze(0).permute([1, 2, 0]))
  imageio.imsave(os.path.join(output, "xtrans_result.tif"), np.transpose(xout, [1, 2, 0]))

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("output", help="output directory")
  parser.add_argument("--input", default=_TEST_INPUT, help="test input, uses the default test input provided if no argument.")
  args = parser.parse_args()
  main(args)
  

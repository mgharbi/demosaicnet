#!/bin/env python
"""Evaluate a demosaicking model."""
import argparse
import os
import time

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import ttools
from ttools.modules.image_operators import crop_like

import demosaicnet


LOG = ttools.get_logger(__name__)


def main(args):
    """Entrypoint to the training."""

    # Load model parameters from checkpoint, if any
    meta = ttools.Checkpointer.load_meta(args.checkpoint_dir)
    if meta is None:
        LOG.warning("No checkpoint found at %s, aborting.", args.checkpoint_dir)
        return

    data = demosaicnet.Dataset(args.data, download=False,
                               mode=meta["mode"],
                               subset=demosaicnet.TEST_SUBSET)
    dataloader = DataLoader(
        data, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

    if meta["mode"] == demosaicnet.BAYER_MODE:
        model = demosaicnet.BayerDemosaick(depth=meta["depth"],
                                           width=meta["width"],
                                           pretrained=True,
                                           pad=False)
    elif meta["mode"] == demosaicnet.XTRANS_MODE:
        model = demosaicnet.XTransDemosaick(depth=meta["depth"],
                                            width=meta["width"],
                                            pretrained=True,
                                            pad=False)

    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model, meta=meta)
    checkpointer.load_latest()  # Resume from checkpoint, if any.

    # No need for gradients
    for p in model.parameters():
        p.requires_grad = False

    mse_fn = th.nn.MSELoss()
    psnr_fn = ttools.modules.losses.PSNR()

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda"
        LOG.info("Using CUDA")

    count = 0
    mse = 0.0
    psnr = 0.0
    for idx, batch in enumerate(dataloader):
        mosaic = batch[0].to(device)
        target = batch[1].to(device)
        output = model(mosaic)

        target = crop_like(target, output)

        psnr_ = psnr_fn(th.clamp(output, 0, 1), target).item()
        mse_ = mse_fn(th.clamp(output, 0, 1), target).item()

        psnr += psnr_
        mse += mse_
        count += 1

        LOG.info("Image %04d, PSNR = %.1f dB, MSE = %.5f", idx, psnr_, mse_)

    mse /= count
    psnr /= count

    LOG.info("-----------------------------------")
    LOG.info("Average, PSNR = %.1f dB, MSE = %.5f", psnr, mse)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="root directory for the demosaicnet dataset.")
    parser.add_argument("checkpoint_dir", help="directory with the model checkpoints.")
    args = parser.parse_args()
    ttools.set_logger(False)
    main(args)

#!/bin/env python
"""Train a demosaicking model."""
import os
import time

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import ttools
from ttools.modules.image_operators import crop_like

import demosaicnet


LOG = ttools.get_logger(__name__)


class DemosaicnetInterface(ttools.ModelInterface):
    """Training and validation interface.

    Args:
        model(th.nn.Module): model to train.
        lr(float): learning rate for the optimizer.
        cuda(bool): whether to use CPU or GPU for training.
    """
    def __init__(self, model, lr=1e-4, cuda=th.cuda.is_available()):
        self.model = model
        self.device = "cpu"
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = th.nn.MSELoss()
        self.psnr = ttools.modules.losses.PSNR()

    def forward(self, batch):
        mosaic = batch[0]
        mosaic = mosaic.to(self.device)
        output = self.model(mosaic)
        return output

    def backward(self, batch, fwd_output):
        target = batch[1].to(self.device)

        # remove boundaries to match output size
        target = crop_like(target, fwd_output)

        loss = self.loss(fwd_output, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        with th.no_grad():
            psnr = self.psnr(th.clamp(fwd_output, 0, 1), target)

        return {"loss": loss.item(), "psnr": psnr.item()}

    def init_validation(self):
        return {"count": 0, "psnr": 0}

    def update_validation(self, batch, fwd_output, running_data):
        target = batch[1].to(self.device)

        # remove boundaries to match output size
        target = crop_like(target, fwd_output)

        with th.no_grad():
            psnr = self.psnr(th.clamp(fwd_output, 0, 1), target)
            n = target.shape[0]

        return {
            "psnr": running_data["psnr"] + psnr.item()*n,
            "count": running_data["count"] + n
        }

    def finalize_validation(self, running_data):
        return {
            "psnr": running_data["psnr"] / running_data["count"]
        }


class ImageCallback(ttools.callbacks.ImageDisplayCallback):
    def visualized_image(self, batch, fwd_output):
        fwd_output = fwd_output.cpu().detach()
        mosaic, target = batch
        mosaic = crop_like(mosaic.cpu().detach(), fwd_output)
        target = crop_like(target.cpu().detach(), fwd_output)
        diff = 4*(fwd_output-target).abs()
        vizdata = [mosaic, target, fwd_output, diff]
        viz = th.clamp(th.cat(vizdata, 2), 0, 1)
        return viz

    def caption(self, batch, fwd_result):
        return "mosaic, ref, ours, diff"


def main(args):
    """Entrypoint to the training."""

    # Load model parameters from checkpoint, if any
    meta = ttools.Checkpointer.load_meta(args.checkpoint_dir)
    if meta is None:
        LOG.info("No metadata or checkpoint, "
                 "parsing model parameters from command line.")
        meta = {
            "depth": args.depth,
            "width": args.width,
            "mode": args.mode,
        }

    data = demosaicnet.Dataset(args.data, download=False,
                               mode=meta["mode"],
                               subset=demosaicnet.TRAIN_SUBSET)
    dataloader = DataLoader(
        data, batch_size=args.bs, num_workers=args.num_worker_threads,
        pin_memory=True, shuffle=True)

    val_dataloader = None
    if args.val_data:
        val_data = demosaicnet.Dataset(args.data, download=False,
                                       mode=meta["mode"],
                                       subset=demosaicnet.VAL_SUBSET)
        val_dataloader = DataLoader(
            val_data, batch_size=args.bs, num_workers=1,
            pin_memory=True, shuffle=False)

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
    checkpointer = ttools.Checkpointer(
        args.checkpoint_dir, model, meta=meta)

    interface = DemosaicnetInterface(model, lr=args.lr, cuda=args.cuda)

    checkpointer.load_latest()  # Resume from checkpoint, if any.

    trainer = ttools.Trainer(interface)

    keys = ["loss", "psnr"]
    val_keys = ["psnr"]

    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=val_keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=val_keys, server=args.server,
        env=args.env, port=args.port))
    trainer.add_callback(ImageCallback(
        server=args.server, env=args.env, win="images", port=args.port))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=8, interval=3600, max_epochs=10))

    if args.cuda:
        LOG.info("Training with CUDA enabled")
    else:
        LOG.info("Training on CPU")

    trainer.train(
        dataloader, num_epochs=args.num_epochs,
        val_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--depth", default=15,
                        help="number of net layers.")
    parser.add_argument("--width", default=64,
                        help="number of features per layer.")
    parser.add_argument("--mode", default=demosaicnet.BAYER_MODE,
                        choices=[demosaicnet.BAYER_MODE,
                                 demosaicnet.XTRANS_MODE],
                        help="number of features per layer.")
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)

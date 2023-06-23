"""Helper functions."""

from abc import ABCMeta, abstractmethod
import argparse
import logging
import os
import re
import signal
import time

import torch as th
import numpy as np
import torch as th
from tqdm import tqdm


log = logging.getLogger(__name__)


def crop_like(src, tgt):
    """Crop a source image to match the spatial dimensions of a target.

    Assumes sizes are even.

    Args:
        src (th.Tensor or np.ndarray): image to be cropped
        tgt (th.Tensor or np.ndarray): reference image
    """
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    delta = (src_sz[2:4]-tgt_sz[2:4])
    crop = np.maximum(delta // 2, 0)  # no negative crop
    crop2 = delta - crop

    if (crop > 0).any() or (crop2 > 0).any():
        # NOTE: convert to ints to enable static slicing in ONNX conversion
        src_sz = [int(x) for x in src_sz]
        crop = [int(x) for x in crop]
        crop2 = [int(x) for x in crop2]
        return src[..., crop[0]:src_sz[-2]-crop2[0],
                   crop[1]:src_sz[-1]-crop2[1]]
    else:
        return src


class ExponentialMovingAverage(object):
    """Keyed tracker that maintains an exponential moving average for each key.

    Args:
      keys(list of str): keys to track.
      alpha(float): exponential smoothing factor (higher = smoother).
    """

    def __init__(self, keys, alpha=0.999):
        self._is_first_update = {k: True for k in keys}
        self._alpha = alpha
        self._values = {k: 0 for k in keys}

    def __getitem__(self, key):
        return self._values[key]

    def update(self, key, value):
        if value is None:
            return
        if self._is_first_update[key]:
            self._values[key] = value
            self._is_first_update[key] = False
        else:
            self._values[key] = self._values[key] * \
                self._alpha + value*(1.0-self._alpha)


class BasicArgumentParser(argparse.ArgumentParser):
    """A basic argument parser with commonly used training options."""

    def __init__(self, *args, **kwargs):
        super(BasicArgumentParser, self).__init__(*args, **kwargs)

        self.add_argument("--data", required=True, help="path to the training data.")
        self.add_argument("--val_data", help="path to the validation data.")
        self.add_argument("--config", help="path to a config file.")
        self.add_argument("--checkpoint_dir", required=True,
                          help="Output directory where checkpoints are saved")
        self.add_argument("--init_from", help="path to a checkpoint from which to try and initialize the weights.")

        self.add_argument("--lr", type=float, default=1e-4,
                          help="Learning rate for the optimizer")
        self.add_argument("--bs", type=int, default=4, help="Batch size")
        self.add_argument("--num_epochs", type=int,
                          help="Number of epochs to train for")
        self.add_argument("--num_worker_threads", default=4, type=int,
                          help="Number of threads that load data")

        # self.add_argument("--experiment_log",
        #                   help="csv file in which we log our experiments")

        self.add_argument("--cuda", action="store_true",
                          dest="cuda", help="Force GPU")
        self.add_argument("--no-cuda", action="store_false",
                          dest="cuda", help="Force CPU")

        self.add_argument("--server", help="Visdom server url")
        self.add_argument("--base_url", default="/", help="Visdom base url")
        self.add_argument("--env", default="main", help="Visdom environment")
        self.add_argument("--port", default=8097, type=int,
                          help="Visdom server port")

        self.add_argument('--debug', dest="debug", action="store_true")

        self.set_defaults(cuda=th.cuda.is_available(), debug=False)


class ModelInterface(metaclass=ABCMeta):
    """An adapter to run or train a model."""

    def __init__(self):
        pass

    @abstractmethod
    def training_step(self, batch):
        """Training step given a batch of data.

        This should implement a forward pass of the model, compute gradients,
        take an optimizer step and return useful metrics and tensors for
        visualization and training callbacks. 

        Args:
          batch (dict): batch of data provided by a data pipeline.

        Returns:
          train_step_data (dict): a dictionary of outputs.
        """
        return {}

    def init_validation(self):
        """Initializes the quantities to be reported during validation.

        The default implementation is a no-op

        Returns:
          data (dict): initialized values
        """
        log.warning("Running a ModelInterface validation initialization that was not overriden: this is a no-op.")
        data = {}
        return data

    def validation_step(self, batch, running_val_data):
        """Updates the running validataion with the current batch's results.

        The default implementation is a no-op

        Args:
          batch (dict): batch of data provided by a data pipeline.
          running_val_data (dict): current aggregates of the validation loop.

        Returns:
          updated_data (dict): new updated value for the running_val_data.
        """
        log.warning("Running a ModelInterface validation step that was not overriden: this is a no-op.")
        return {}

    def __repr__(self):
        return self.__class__.__name__


class Checkpointer(object):
    """Save and restore model and optimizer variables.

    Args:
      root (string): path to the root directory where the files are stored.
      model (torch.nn.Module):
      meta (dict): a dictionary of training or configuration parameters useful
          to initialize the model upon loading the checkpoint again.
      optimizers (single or list of torch.optimizer): optimizers whose parameters will
        be checkpointed together with the model.
      schedulers (single or list of
      torch.optim.lr_scheduler): schedulers whose
          parameters will be checkpointed together with
          the model.
      prefix (str): unique prefix name in case several models are stored in the
        same folder.
    """

    EXTENSION = ".pth"

    def __init__(self, root, model=None, meta=None, optimizers=None,
                 schedulers=None, prefix=None):
        self.root = root
        self.model = model
        self.meta = meta

        # TODO(mgharbi): verify the prefixes are unique.

        if optimizers is None:
            log.info("No optimizer state will be stored in the "
                        "checkpointer")
        else:
            # if we have only one optimizer, make it a list
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
        self.optimizers = optimizers
        if schedulers is not None:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
        self.schedulers = schedulers

        log.debug(self)

        self.prefix = ""
        if prefix is not None:
            self.prefix = prefix

    def __repr__(self):
        return "Checkpointer with root at \"{}\"".format(self.root)

    def __path(self, path, prefix=None):
        if prefix is None:
            prefix = ""
        return os.path.join(self.root, prefix+os.path.splitext(path)[0] + ".pth")

    def save(self, path, extras=None):
        """Save model, metaparams and extras to relative path.

        Args:
          path (string): relative path to the file being saved (without extension).
          extras (dict): extra user-provided information to be saved with the model.
        """

        if self.model is None:
            model_state = None
        else:
            log.debug("Saving model state dict")
            model_state = self.model.state_dict()

        opt_dicts = []
        if self.optimizers is not None:
            for opt in self.optimizers:
                opt_dicts.append(opt.state_dict())

        sched_dicts = []
        if self.schedulers is not None:
            for s in self.schedulers:
                sched_dicts.append(s.state_dict())

        filename = self.__path(path, prefix=self.prefix)
        os.makedirs(self.root, exist_ok=True)
        th.save({'model': model_state,
                 'meta': self.meta,
                 'extras': extras,
                 'optimizers': opt_dicts,
                 'schedulers': sched_dicts,
                 }, filename)
        log.debug("Checkpoint saved to \"{}\"".format(filename))

    def try_and_init_from(self, path):
        """Try to initialize the models's weights from an external checkpoint.

        Args:
            path(str): full path to the checkpoints to load model parameters
                from.
        """
        log.info("Loading weights from foreign checkpoint {}".format(path))
        if not os.path.exists(path):
            raise ValueError("Checkpoint {} does not exist".format(path))

        chkpt = th.load(path, map_location=th.device("cpu"))
        if "model" not in chkpt.keys() or chkpt["model"] is None:
            raise ValueError("{} has no model saved".format(path))

        mdl = chkpt["model"]
        for n, p in self.model.named_parameters():
            if n in mdl:
                p2 = mdl[n]
                if p2.shape != p.shape:
                    log.warning("Parameter {} ignored, checkpoint size does not match: {}, should be {}".format(n, p2.shape, p.shape))
                    continue
                log.debug("Parameter {} copied".format(n))
                p.data.copy_(p2)
            else:
                log.warning("Parameter {} ignored, not found in source checkpoint.".format(n))

        log.info("Weights loaded from foreign checkpoint {}".format(path))

    def load(self, path):
        """Loads a checkpoint, updates the model and returns extra data.

        Args:
          path (string): path to the checkpoint file, relative to the root dir.

        Returns:
          extras (dict): extra information passed by the user at save time.
          meta (dict): metaparameters of the model passed at save time.
        """

        filename = self.__path(path, prefix=None)
        chkpt = th.load(filename, map_location="cpu")  # TODO: check behavior

        if self.model is not None and chkpt["model"] is not None:
            log.debug("Loading model state dict")
            self.model.load_state_dict(chkpt["model"])

        if "optimizers" in chkpt.keys():
            if self.optimizers is not None and chkpt["optimizers"] is not None:
                try:
                    for opt, state in zip(self.optimizers,
                                          chkpt["optimizers"]):
                        log.debug("Loading optimizers state dict for %s", opt)
                        opt.load_state_dict(state)
                except:
                    # We do not raise an error here, e.g. in case the user simply
                    # changes optimizer
                    log.warning("Could not load optimizer state dicts, "
                                "starting from scratch")

        if "schedulers" in chkpt.keys():
            if self.schedulers is not None and chkpt["schedulers"] is not None:
                try:
                    for s, state in zip(self.schedulers,
                                          chkpt["schedulers"]):
                        log.debug("Loading scheduler state dict for %s", s)
                        s.load_state_dict(state)
                except:
                    log.warning("Could not load scheduler state dicts, "
                                "starting from scratch")

        log.debug("Loaded checkpoint \"{}\"".format(filename))
        return tuple(chkpt[k] for k in ["extras", "meta"])

    def load_latest(self):
        """Try to load the most recent checkpoint, skip failing files.

        Returns:
          extras (dict): extra user-defined information that was saved in the
              checkpoint.
          meta (dict): metaparameters of the model passed at save time.
        """
        all_checkpoints = self.sorted_checkpoints()

        extras = None
        meta = None
        for f in all_checkpoints:
            try:
                extras, meta = self.load(f)
                return extras, meta
            except Exception as e:
                log.debug(
                    "Could not load checkpoint \"{}\", moving on ({}).".format(f, e))
        log.debug("No checkpoint found to load.")
        return extras, meta

    def sorted_checkpoints(self):
        """Get list of all checkpoints in root directory, sorted by creation date.

        Returns:
            chkpts (list of str): sorted checkpoints in the root folder.
        """
        reg = re.compile(r"{}.*\{}".format(self.prefix, Checkpointer.EXTENSION))
        if not os.path.exists(self.root):
            all_checkpoints = []
        else:
            all_checkpoints = [f for f in os.listdir(
                self.root) if reg.match(f)]
        mtimes = []
        for f in all_checkpoints:
            mtimes.append(os.path.getmtime(os.path.join(self.root, f)))

        mf = sorted(zip(mtimes, all_checkpoints))
        chkpts = [m[1] for m in reversed(mf)]
        log.debug("Sorted checkpoints {}".format(chkpts))
        return chkpts

    def delete(self, path):
        """Delete checkpoint at path.

        Args:
            path(str): full path to the checkpoint to delete.
        """
        if path in self.sorted_checkpoints():
            os.remove(os.path.join(self.root, path))
        else:
            log.warning("Trying to delete a checkpoint that does not exists.")

    @staticmethod
    def load_meta(root, prefix=None):
        """Fetch model metadata without touching the saved parameters.

        This loads the metadata from the most recent checkpoint in the root
        directory.

        Args:
            root(str): path to the root directory containing the checkpoints
            prefix(str): unique prefix for the checkpoint to be loaded (e.g. if
                multiple models are saved in the same location)
        """
        chkptr = Checkpointer(root, model=None, meta=None, prefix=prefix, 
                              optimizers=[])
        log.debug("checkpoints: %s", chkptr.sorted_checkpoints())
        _, meta = chkptr.load_latest()
        return meta


class Trainer(object):
    """Implements a simple training loop with hooks for callbacks.

    Args:
      interface (ModelInterface): adapter to run forward and backward
        pass on the model being trained.

    Attributes:
      callbacks (list of Callbacks): hooks that will be called while training
        progresses.
    """

    def __init__(self, interface):
        super(Trainer, self).__init__()
        self.callbacks = []
        self.interface = interface
        log.debug("Creating {}".format(self))

        signal.signal(signal.SIGINT, self.interrupt_handler)

        self._keep_running = True

    def interrupt_handler(self, signo, frame):
        """Stop the training process upon receiving a SIGINT (Ctrl+C)."""
        log.debug("interrupting run")
        self._keep_running = False

    def _stop(self):
        # Reset the run flag
        self._keep_running = True
        self.__training_end()

    def add_callback(self, callback):
        """Adds a callback to the list of training hooks.

        Args:
            callback(ttools.Callback): callback to add.
        """
        log.debug("Adding callback {}".format(callback))
        # pass an interface reference to the callback
        callback.model_interface = self.interface
        self.callbacks.append(callback)

    def train(self, dataloader, starting_epoch=None, num_epochs=None,
              val_dataloader=None):
        """Main training loop. This starts the training procedure.

        Args:
          dataloader (DataLoader): loader that yields training batches.
          starting_epoch (int, optional): index of the epoch we are starting from.
          num_epochs (int, optional): max number of epochs to run.
          val_dataloader (DataLoader, optional): loader that yields validation
            batches
        """
        self.__training_start(dataloader)
        if starting_epoch is None:
            starting_epoch = 0

        log.info("Starting taining from epoch %d", starting_epoch)

        epoch = starting_epoch

        while num_epochs is None or epoch < starting_epoch + num_epochs:
            self.__epoch_start(epoch)
            for batch_idx, batch in enumerate(dataloader):
                if not self._keep_running:
                    self._stop()
                    return
                self.__batch_start(batch_idx, batch)
                train_step_data = self.__training_step(batch)
                self.__batch_end(batch, train_step_data)
            self.__epoch_end()

            # TODO: allow validation at intermediate steps during one epoch

            # Validate
            if val_dataloader:
                with th.no_grad():
                    running_val_data = self.__validation_start(val_dataloader)
                    for batch_idx, batch in enumerate(val_dataloader):
                        if not self._keep_running:
                            self._stop()
                            return
                        self.__val_batch_start(batch_idx, batch)
                        running_val_data = self.__validation_step(batch, running_val_data)
                        self.__val_batch_end(batch, running_val_data)
                    self.__validation_end(running_val_data)

            epoch += 1

            if not self._keep_running:
                self._stop()
                return

        self._stop()

    def __repr__(self):
        return "Trainer({}, {} callbacks)".format(
            self.interface, len(self.callbacks))

    def __training_start(self, dataloader):
        for cb in self.callbacks:
            cb.training_start(dataloader)

    def __training_end(self):
        for cb in self.callbacks:
            cb.training_end()

    def __epoch_start(self, epoch_idx):
        for cb in self.callbacks:
            cb.epoch_start(epoch_idx)

    def __epoch_end(self):
        for cb in self.callbacks:
            cb.epoch_end()

    def __batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.batch_start(batch_idx, batch)

    def __batch_end(self, batch, train_step_data):
        for cb in self.callbacks:
            cb.batch_end(batch, train_step_data)

    def __val_batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.val_batch_start(batch_idx, batch)

    def __val_batch_end(self, batch, running_val_data):
        for cb in self.callbacks:
            cb.val_batch_end(batch, running_val_data)

    def __validation_start(self, dataloader):
        for cb in self.callbacks:
            cb.validation_start(dataloader)
        return self.interface.init_validation()

    def __validation_end(self, running_val_data):
        for cb in self.callbacks:
            cb.validation_end(running_val_data)

    def __training_step(self, batch):
        return self.interface.training_step(batch)

    def __validation_step(self, batch, running_val_data):
        return self.interface.validation_step(batch, running_val_data)


class Callback(object):
    """Base class for all training callbacks.

    Attributes:
        epoch(int): current epoch index.
        batch(int): current batch index.
        datasize(int): number of batches in the training dataset.
        val_datasize(int): number of batches in the validation dataset.
        model_interface(ttools.ModelInterface): parent interface driving the training.
    """

    def __repr__(self):
        return self.__class__.__name__

    def __init__(self):
        super(Callback, self).__init__()
        self.epoch = 0
        self.batch = 0
        self.val_batch = 0
        self.datasize = 0
        self.val_datasize = 0
        self.model_interface = None

    def training_start(self, dataloader):
        """Hook to execute code when the training begins.

        Args:
            dataloader(th.utils.data.Dataloader): a data loading class that
            provides batches of data for training.
        """
        self.datasize = len(dataloader)

    def training_end(self):
        """Hook to execute code when the training ends."""
        pass

    def epoch_start(self, epoch):
        """Hook to execute code when a new epoch starts.

        Args:
            epoch(int): index of the current epoch.

        Note: self.epoch is never incremented. Instead, it should be set by the
        caller.
        """
        self.epoch = epoch

    def epoch_end(self):
        """Hook to execute code when an epoch ends.

        NOTE: self.epoch is not incremented. Instead it is set externally in
        the `epoch_start` method.
        """
        pass

    def validation_start(self, dataloader):
        """Hook to execute code when a validation run starts.

        Args:
            dataloader(th.utils.data.Dataloader): a data loading class that
            provides batches of data for evaluation.
        """
        self.val_datasize = len(dataloader)

    def validation_end(self, val_data):
        """Hook to execute code when a validation run ends."""
        pass

    def batch_start(self, batch_idx, batch_data):
        """Hook to execute code when a training step starts.

        Args:
            batch_idx(int): index of the current batch.
            batch_data: a Tensor, tuple of dict with the current batch of data.
        """
        self.batch = batch_idx

    def batch_end(self, batch_data, train_step_data):
        """Hook to execute code when a training step ends.

        Args:
            batch_data: a Tensor, tuple of dict with the current batch of data.
            train_setp_data(dict): outputs from the `training_step` of a
                ModelInterface.
        """
        pass

    def val_batch_start(self, batch_idx, batch_data):
        """Hook to execute code when a validation step starts.

        Args:
            batch_idx(int): index of the current batch.
            batch_data: a Tensor, tuple of dict with the current batch of data.
        """
        self.val_batch = batch_idx

    def val_batch_end(self, batch_data, running_val_data):
        """Hook to execute code when a validation step ends.

        Args:
            batch_data: a Tensor, tuple of dict with the current batch of data.
            train_setp_data(dict): running outputs produced by the `validation_step` of a
                ModelInterface.
        """
        pass

class CheckpointingCallback(Callback):
    """A callback that periodically saves model checkpoints to disk.

    Args:
      checkpointer (Checkpointer): actual checkpointer responsible for the I/O.
      interval (int, optional): minimum time in seconds between periodic
          checkpoints (within an epoch). There is not periodic checkpoint if
          this value is None.
      max_files (int, optional): maximum number of periodic checkpoints to keep
          on disk.
      max_epochs (int, optional): maximum number of epoch checkpoints to keep
          on disk.
    """

    PERIODIC_PREFIX = "periodic_"
    EPOCH_PREFIX = "epoch_"

    def __init__(self, checkpointer, interval=600,
                 max_files=5, max_epochs=10):
        super(CheckpointingCallback, self).__init__()
        self.checkpointer = checkpointer
        self.interval = interval
        self.max_files = max_files
        self.max_epochs = max_epochs

        self.last_checkpoint_time = time.time()

    def epoch_end(self):
        """Save a checkpoint at the end of each epoch."""
        super(CheckpointingCallback, self).epoch_end()
        path = "{}{}".format(CheckpointingCallback.EPOCH_PREFIX, self.epoch)
        self.checkpointer.save(path, extras={"epoch": self.epoch + 1})
        self.__purge_old_files()

    def training_end(self):
        super(CheckpointingCallback, self).training_end()
        self.checkpointer.save("training_end", extras={"epoch": self.epoch + 1})

    def batch_end(self, batch_data, train_step_data):
        """Save a periodic checkpoint if requested."""

        super(CheckpointingCallback, self).batch_end(
            batch_data, train_step_data)

        if self.interval is None:  # We skip periodic checkpoints
            return

        now = time.time()

        delta = now - self.last_checkpoint_time

        if delta < self.interval:  # last checkpoint is too recent
            return

        log.debug("Periodic checkpoint")
        self.last_checkpoint_time = now

        filename = "{}{}".format(CheckpointingCallback.PERIODIC_PREFIX,
                                   time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.checkpointer.save(filename, extras={"epoch": self.epoch})
        self.__purge_old_files()

    def __purge_old_files(self):
        """Delete checkpoints that are beyond the max to keep."""

        chkpts = self.checkpointer.sorted_checkpoints()
        p_chkpts = []
        e_chkpts = []
        for c in chkpts:
            if c.startswith(self.checkpointer.prefix + CheckpointingCallback.PERIODIC_PREFIX):
                p_chkpts.append(c)

            if c.startswith(self.checkpointer.prefix + CheckpointingCallback.EPOCH_PREFIX):
                e_chkpts.append(c)

        # Delete periodic checkpoints
        if self.max_files is not None and len(p_chkpts) > self.max_files:
            for c in p_chkpts[self.max_files:]:
                log.debug("CheckpointingCallback deleting {}".format(c))
                self.checkpointer.delete(c)

        # Delete older epochs
        if self.max_epochs is not None and len(e_chkpts) > self.max_epochs:
            for c in e_chkpts[self.max_epochs:]:
                log.debug("CheckpointingCallback deleting (epoch) {}".format(c))
                self.checkpointer.delete(c)


class KeyedCallback(Callback):
    """An abstract Callback that performs the same action for all keys in a list.

    The keys (resp. val_keys) are used to access the backward_data (resp.
    validation_data) produced by a ModelInterface.

    Args:
      keys (list of str or None): list of keys whose values will be logged during
          training.
      val_keys (list of str or None): list of keys whose values will be logged during
          validation
    """
    def __init__(self, keys=None, val_keys=None, smoothing=0.999):
        super(KeyedCallback, self).__init__()
        if keys is None and val_keys is None:
            log.warning("Logger has no keys, nor val_keys")

        if keys is None:
            self.keys = []
        else:
            self.keys = keys

        if val_keys is None:
            self.val_keys = []
        else:
            self.val_keys = val_keys

        # Only smooth the training keys
        self.ema = ExponentialMovingAverage(self.keys, alpha=smoothing)

    def batch_end(self, batch_data, train_step_data):
        for k in self.keys:
            self.ema.update(k, train_step_data[k])

class ProgressBarCallback(KeyedCallback):
    """A progress bar optimization logger.

    Args:
        label(str): a prefix label to identify the experiment currently
            running.
    """
    def __init__(self, keys=None, val_keys=None, smoothing=0.99, label=None):
        super(ProgressBarCallback, self).__init__(
            keys=keys, val_keys=val_keys, smoothing=smoothing)
        self.pbar = None
        if label is None:
            self.label = ""
        else:
            self.label = label

    def training_start(self, dataloader):
        super(ProgressBarCallback, self).training_start(dataloader)
        print("Training start")

    def training_end(self):
        super(ProgressBarCallback, self).training_end()
        print("Training ends")

    def epoch_start(self, epoch):
        super(ProgressBarCallback, self).epoch_start(epoch)
        desc = "Epoch {}".format(self.epoch)
        if self.label is not None:
            desc = "%s | " % self.label + desc
        self.pbar = tqdm(total=self.datasize, unit=" batches",
                         desc=desc)

    def epoch_end(self):
        super(ProgressBarCallback, self).epoch_end()
        self.pbar.close()
        self.pbar = None

    def validation_start(self, dataloader):
        super(ProgressBarCallback, self).validation_start(dataloader)
        print("Running validation...")
        self.pbar = tqdm(total=len(dataloader), unit=" batches",
                         desc="Validation {}".format(self.epoch))

    def val_batch_end(self, batch, running_val_data):
        self.pbar.update(1)

    def validation_end(self, val_data):
        super(ProgressBarCallback, self).validation_end(val_data)
        self.pbar.close()
        self.pbar = None
        s = " "*ProgressBarCallback.TABSTOPS + "Validation {} | ".format(
            self.epoch)
        for k in self.val_keys:
            s += "{} = {:.2f} ".format(k, val_data[k])
        print(s)

    def batch_end(self, batch_data, train_step_data):
        super(ProgressBarCallback, self).batch_end(batch_data, train_step_data)
        d = {}
        for k in self.keys:
            d[k] = self.ema[k]
        self.pbar.update(1)
        self.pbar.set_postfix(d)

    TABSTOPS = 2
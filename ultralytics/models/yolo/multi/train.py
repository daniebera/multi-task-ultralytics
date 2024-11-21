# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseMultiTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetSRModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import DEFAULT_CFG

class DetSRTrainer(BaseMultiTrainer):
    """
    A class extending the BaseMultiTrainer class for training based on a multi-task model.

    Example:
        ```python
        from ultralytics.models.yolo.multi import MultiTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = MultiTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, **kwargs):
        super().__init__(cfg, **kwargs)
        # Initialize trainers specific to detection and super-resolution tasks
        self.task_trainers["detection"] = yolo.detect.DetectionTrainer(cfg, **kwargs)
        # Fixme: change imgsz for Super Resolution data loading
        kwargs['overrides']['imgsz'] = kwargs['overrides']['imgsz'] * cfg.factor
        self.factor = cfg.factor
        self.task_trainers["classification"] = yolo.detect.DetectionTrainer(cfg, **kwargs)


    # Todo: Adjust preprocess_batch method to handle multi-dataset batch for multi-task training. Here for Super Resolution
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        # Create a unique batch with task-specific keys and concatenate 'img' values.
        tmp_batch = {}
        for idx, sub_batch in enumerate(batch):
            for key in sub_batch.keys():
                if key != "img":
                    tmp_batch[key + f"_{idx}"] = sub_batch[key]
                else:
                    if idx == 0:
                        tmp_batch["img"] = sub_batch["img"]
                    else:
                        tmp_batch["hr_img"+ f"_{idx}"] = sub_batch["img"]
                        sub_batch["img"] = nn.functional.interpolate(tmp_batch['hr_img'+ f"_{idx}"],
                                                                 size=[i // self.factor for i in
                                                                       tmp_batch['hr_img'+ f"_{idx}"].size()[2:]],
                                                                 mode='bilinear',
                                                                 align_corners=True)
                        tmp_batch["img"] = torch.cat((tmp_batch["img"], sub_batch["img"]), 0)

        batch = tmp_batch
        # Todo: avoid hardcode of 'hr_img' and 'hr_img_1' keys
        batch["hr_img_1"] = batch["hr_img_1"].to(self.device, non_blocking=True).float() / 255
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255

        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetSRModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    # Fixme: check if main task validation works in multi-task training and handle loss_names
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss","sr_loss"
        return yolo.multi.DetSRValidator(
            self.test_loader['detection'], save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        # Fixme: Implement plotting for SR task
        for task_train_loader in self.train_loader.values():
            boxes = np.concatenate([lb["bboxes"] for lb in task_train_loader.dataset.labels], 0)
            cls = np.concatenate([lb["cls"] for lb in task_train_loader.dataset.labels], 0)
            plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

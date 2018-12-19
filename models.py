import toolbox
import chainer
import numpy as np
import pandas as pd

import yaml

import cv2
import torch
import torch.nn.functional as F
from addict import Dict
import random

from toolbox.deeplab.models import DeepLabV2_ResNet101_MSC


class InstanceSeg():
    def __init__(self, params_path: str, snapshot_path: str, gpu: int):
        """Instance segmentation model.

        Args:
            params_path    : Location of model setup, yaml.
            snapshot_path  : Location of weights, npz file.
            gpu            : Which gpu to use.
        """
        self._params = yaml.load(open(params_path))
        self._snapshot = snapshot_path
        self._class_names = self._params['class_names']
        self._gpu = gpu

        self._setup_model()

    def _setup_model(self):
        self.model = toolbox.models.MaskRCNNResNet(
            n_layers=int(self._params["model"].lstrip('resnet')),
            n_fg_class=len(self._class_names),
            pretrained_model=self._snapshot,
            pooling_func=toolbox.functions.roi_align_2d,
            anchor_scales=self._params['anchor_scales'],
            mean=self._params.get('mean', (123.152, 115.903, 103.063)),
            min_size=self._params['min_size'],
            max_size=self._params['max_size'],
            roi_size=self._params.get('roi_size', 7),
        )
        if self._gpu >= 0:
            chainer.cuda.get_device_from_id(self._gpu).use()
            self.model.to_gpu()

    def predict(self, img):
        """Predict on one image or batch

        Return:
            bboxes, masks, labels, scores
        """
        # Predict for one image
        if isinstance(img, np.ndarray) and img.ndim == 3:
            bboxes, masks, labels, scores = self._predict_batch([img.transpose(2, 0, 1)])
            return bboxes, masks, labels, scores
        # Predict on batch
        return self._predict_batch([i.transpose(2, 0, 1) for i in img])

    def _predict_batch(self, imgs):
        return self.model.predict(imgs)


class SemanticSeg():
    def __init__(self, params_path: str, snapshot_path: str, gpu: int):
        """Semantic segmentation model.

        Args:
            params_path    : Location of model setup, yaml.
            snapshot_path  : Location of weights, npz file.
            gpu            : Which gpu to use.
        """
        self._config = Dict(yaml.load(open(params_path)))
        self._classes = {}
        self._snapshot = snapshot_path
        self._gpu = gpu

        with open(self._config.LABELS) as f:
            for label in f:
                label = label.rstrip().split("\t")
                self._classes[int(label[0])] = label[1].split(",")[0]

        self._setup_model()
        self._add_merged_stuff()

    def _setup_model(self):
        if self._gpu >= 0:
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        torch.set_grad_enabled(False)

        # Model
        self.model = DeepLabV2_ResNet101_MSC(n_classes=self._config.N_CLASSES)
        state_dict = torch.load(self._snapshot, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self._device)

    def _add_merged_stuff(self):
        # This model was not trained with the new stuff-merged, so map manually
        # using this string from http://cocodataset.org/#panoptic-eval
        # Also set to "VOID" (-1) deleted stuff.
        s = """\
tree-merged: branch, tree, bush, leaves
fence-merged: cage, fence, railing
ceiling-merged: ceiling-tile, ceiling-other
sky-other-merged: clouds, sky-other, fog
cabinet-merged: cupboard, cabinet
table-merged: desk-stuff, table
floor-other-merged: floor-marble, floor-other, floor-tile
pavement-merged: floor-stone, pavement
mountain-merged: hill, mountain
grass-merged: moss, grass, straw
dirt-merged: mud, dirt
paper-merged: napkin, paper
food-other-merged: salad, vegetable, food-other
building-other-merged: skyscraper, building-other
rock-merged: stone, rock
wall-other-merged: wall-other, wall-concrete, wall-panel
rug-merged: mat, rug, carpet"""
        # Turn string into useful mapping
        map_into_merged_int = {vv: idx+183 for idx, (k, v) in enumerate(
            x.split(": ") for x in s.split("\n")) for vv in v.split(", ")}
        # Add mapping for delete stuff
        map_into_merged_int.update({k: -1 for k in [
            "furniture-other", "metal", "plastic", "solid-other",
            "structural-other", "waterdrops", "textile-other", "cloth",
            "clothes", "plant-other", "wood", "ground-other"]})

        _inv = {v: k for k, v in self._classes.items()}
        self._map_to_merged = {_inv[k]: v for k, v in map_into_merged_int.items()}

        extend_stuff_merged = {idx+183: k for idx, (k, v) in enumerate(
            x.split(": ") for x in s.split("\n"))}
        self._classes.update(extend_stuff_merged)
        self._classes.update({-1: "VOID"})

    def _replace_labels_with_merged(self, labelmap):
        # Simpler, just use pandas
        return pd.DataFrame(labelmap).replace(self._map_to_merged).values

    def _preprocess_one(self, img):
        image = img.copy().astype(float)
        scale = self._config.IMAGE.SIZE.TEST / max(image.shape[:2])
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        image -= np.array(
            [
                float(self._config.IMAGE.MEAN.B),
                float(self._config.IMAGE.MEAN.G),
                float(self._config.IMAGE.MEAN.R),
            ]
        )
        return image.transpose(2, 0, 1)

    def _preprocess_image(self, imgs):
        buff = []
        for img in imgs:
            buff.append(self._preprocess_one(img))
        image = torch.from_numpy(np.array(buff)).float()
        return image.to(self._device)

    def predict(self, img):
        """Predict on one image or batch

        Return:
            labelmap, labels
        """
        if isinstance(img, np.ndarray) and img.ndim == 3:
            return self._predict_batch([img])
        return self._predict_batch(img)

    def _predict_batch(self, imgs):
        # Note: it surprisingly does not speedup to run on bacthes
        # TODO: check pytorch implem deeper and find why
        image = self._preprocess_image(imgs)
        output = self.model(image)
        # 0.2s
        output = F.interpolate(
            output,
            size=imgs[0].shape[:2],
            mode="bilinear", align_corners=True
        )
        output = F.softmax(output, dim=1)
        output = output.data.cpu().numpy()

        labelmaps = np.argmax(output, axis=1)
        labelmaps = np.array([
            self._replace_labels_with_merged(x) for x in labelmaps])
        labels = np.array([np.unique(l) for l in labelmaps])
        return labelmaps, labels


class PanopticSeg():
    def __init__(
            self, instaseg: InstanceSeg, semaseg: SemanticSeg,
            thresh: float = 0.7, frac: float = 0.2
            ):
        """Combine instance and semantic segmentation models into panoptic.

        Args:
            instaseg : Instance segmentation model.
            semaseg  : Semantic segmentation model.
            thresh   : Threshold for instance seg proposals, default 0.7.
            frac     : Fraction of mask remaining to retain proposal, default 0.2.
        """
        self.instaseg = instaseg
        self.semaseg = semaseg
        self._thresh = thresh
        self._frac = frac
        self._invert_mapping = {v: k for k, v in semaseg._classes.items()}

    def predict(self, img, img_id=0):
        """Predict panoptic segmentation on one image."""
        # Get the respective predictions
        bbox, mask, label, score = self._predict_instance(img)
        labelmap, labels_sema = self._predict_semantic(img)

        # Create random indpendants labels
        _ids = random.sample(range(1, 16711422), len(label) + len(
            labels_sema[labels_sema != -1]))
        ids_instance = _ids[:len(label)]
        ids_semantic = _ids[len(label):]

        RGB, canvas = self._merge_masks(
            img, labels_sema, labelmap, mask, ids_semantic, ids_instance)

        buff = self._create_segments_info(
            canvas, bbox, mask, ids_semantic, ids_instance, labels_sema, label)

        segment = {
            "segments_info": buff,
            "file_name": "{:0>12}.png".format(img_id),
            "image_id": img_id,
        }
        return segment, RGB

    def _predict_instance(self, img):
        bboxes, masks, labels, scores = self.instaseg.predict(img)
        bbox, mask, label, score = bboxes[0], masks[0], labels[0], scores[0]

        # Filter and remap
        bbox, mask, label, score = self._instance_seg_filter(
                img, bbox, mask, label, score)
        label = np.array([self._invert_mapping[
            self.instaseg._class_names[x]] for x in label], dtype=label.dtype)
        return bbox, mask, label, score

    def _predict_semantic(self, img):
        labelmaps, labels_semas = self.semaseg.predict(img)
        return labelmaps[0], labels_semas[0]

    def _instance_seg_filter(self, img, bbox, mask, label, score):
        """Filter instance segmentation prediction to match panoptic criterion"""

        if len(bbox) == 0:
            return bbox, mask, label, score
        # Sort by descending order
        bbox, mask, label, score = map(np.array, list(zip(
            *sorted(zip(bbox, mask, label, score), key=lambda x: x[3], reverse=True)
        )))

        filt_above = score >= self._thresh
        bbox, mask = bbox[filt_above], mask[filt_above]
        label, score = label[filt_above], score[filt_above]

        # Apply non-maximum suppression
        already_masked = np.full(img.shape[:2], True)
        frac_remain = np.full(score.shape, True)

        for idx, m in enumerate(mask):
            proposed_mask = already_masked & m
            remaining_fraction = np.sum(proposed_mask) / np.sum(m)
            if remaining_fraction < self._frac:
                frac_remain[idx] = False
            else:
                already_masked = already_masked & ~m

        return bbox[frac_remain], mask[frac_remain], label[frac_remain], score[frac_remain]

    @staticmethod
    def id_to_color(x):
        return x % 256, x % 256**2 // 256, x // 256**2

    @staticmethod
    def bbox_from_mask(a):
        x = np.any(a, axis=1)
        y = np.any(a, axis=0)
        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]

        return np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(int).tolist()

    @staticmethod
    def bbox_into_xywh(bbox):
        bbox = np.array([
            bbox[:, 0],
            bbox[:, 1],
            bbox[:, 2] - bbox[:, 0],
            bbox[:, 3] - bbox[:, 1],
        ]).T
        return np.round(bbox).astype(int)

    def _merge_masks(self, img, labels_sema, labelmap, mask, ids_semantic, ids_instance):
        """Merge masks into one image with simple overlay, and translate into colors."""
        canvas = np.zeros(img.shape[:2])

        # TODO: smarter merge, if instance overlap > 80% (?) of semantic, merge into
        # one. When multiple, merge with closest (how?)
        # Or just delete if overlap > 80%?

        for idx, lab in enumerate(labels_sema[labels_sema != -1]):
            canvas[labelmap == lab] = ids_semantic[idx]

        for idx, m in enumerate(mask):
            canvas[m] = ids_instance[idx]

        RGB = np.zeros(img.shape, dtype=np.uint8)
        for u in np.unique(canvas):
            r, g, b = self.id_to_color(u)
            RGB[canvas == u, 0] = r
            RGB[canvas == u, 1] = g
            RGB[canvas == u, 2] = b
        return RGB, canvas

    def _create_segments_info(
            self, canvas, bbox, mask, ids_semantic, ids_instance,
            labels_sema, label):
        """Create segments outputs."""
        buff = []
        for idx, lab in enumerate(ids_semantic):
            m = canvas == lab
            _sum = np.sum(m).astype(int)
            if _sum == 0:
                # painted over by instance seg
                continue
            d = {
                "area": int(_sum),
                "category_id": int(labels_sema[labels_sema != -1][idx] + 1),
                "iscrowd": 0,
                "id": lab,
                "bbox": self.bbox_from_mask(m)
            }
            buff.append(d)

        _deboxed = self.bbox_into_xywh(bbox)

        for idx, lab in enumerate(ids_instance):
            m = canvas == lab
            _sum = np.sum(m).astype(int)
            if _sum == 0:
                continue
            d = {
                "area": int(np.sum(mask[idx]).astype(int)),
                "category_id": int(label[idx] + 1),
                "iscrowd": 0,
                "id": lab,
                "bbox": _deboxed[idx].tolist()
            }
            buff.append(d)
        return buff

import logging
import os

from keras.models import load_model
import numpy as np

from MLcommon.AbcModel import AbcModel as AbcModel
from MLgeometry.geometries.Mask import Mask
from MLgeometry.Object import Object


from MLinference.architectures.maskrcnn.model import ProposalLayer, PyramidROIAlign, DetectionLayer
from MLinference.architectures.maskrcnn import model as modellib
from MLinference.architectures.maskrcnn import utils
from MLinference.architectures.maskrcnn.config import Config

LOGGER = logging.getLogger(__name__)


class MaskRcnn(AbcModel):

    _defaults = {
        "name": "segmentation_rules",
        "backbone": "resnet50", # or resnet101
        "mode": "inference", #or "inference"
        "images_per_gpu": 2,
        "max_gt_instances": 600,
        "loss_weights": {'rpn_class_loss': 1.3, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0,
                            'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0},
        "num_classes": 2,
        "steps_per_epoch": 350,
        "min_confidence": 0.7,
        "anchor_scales": (20, 26, 32, 38, 60),
        "rois_per_img": 800,
        "rpn_rois_per_img": 412,
        "validation_steps": 10,
        "init_with": "none",  # coco or imagenet
        "epochs": 3,
        "learning_r": 0.001,
        "training_layers": "heads",  # or "all" to fine tuning
        "labels": ['BG', 'Object'],
        "single_class": False,
        "max_mask_elements": None  # max number of elements to predict with unique mask. Avoid resources consumption

    }

    def __init__(self, conf):

        super().__init__(**conf)

    def load_config(self):
        new_config = Config()
        new_config.NAME = self.name
        new_config.BACKBONE = self.backbone
        new_config.IMAGES_PER_GPU = self.images_per_gpu
        new_config.MAX_GT_INSTANCES = self.max_gt_instances
        new_config.LOSS_WEIGHTS = self.loss_weights
        new_config.NUM_CLASSES = self.num_classes  # Background + varilla
        new_config.STEPS_PER_EPOCH = self.steps_per_epoch
        new_config.DETECTION_MIN_CONFIDENCE = self.min_confidence
        new_config.RPN_ANCHOR_SCALES = self.anchor_scales
        new_config.TRAIN_ROIS_PER_IMAGE = self.rois_per_img
        new_config.RPN_TRAIN_ANCHORS_PER_IMAGE = self.rpn_rois_per_img
        new_config.VALIDATION_STEPS = self.validation_steps
        new_config.DETECTION_MAX_INSTANCES = self.max_gt_instances
        new_config.reload()
        return new_config

    def load_architecture(self):
        LOGGER.info('Loading architecture: {}'.format(self.backbone))
        ROOT_DIR = os.path.abspath(".")
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        config = self.load_config()
        model = modellib.MaskRCNN(mode=self.mode, config=config,
                                  model_dir=MODEL_DIR)

        init_with = self.init_with

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
            if not os.path.exists(COCO_MODEL_PATH):
                utils.download_trained_weights(COCO_MODEL_PATH)
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        else:
            return model
        return model

    def set_weights(self, _weights):
        self.model.keras_model.set_weights(_weights)

    def load_weights(self, path):
        self.model.load_weights(path, by_name=True)

    def load_mask_model(self, path_to_model):
        model = load_model(path_to_model, custom_objects={"ProposalLayer": ProposalLayer,
                                                         "PyramidROIAlign": PyramidROIAlign,
                                                         "DetectionLayer": DetectionLayer})
        return model

    def train(self, train_dataset, val_dataset):
        self.model.train(train_dataset, val_dataset, learning_rate=self.learning_r, epochs=self.epochs, layers=self.training_layers)

    def get_weights(self):
        """Return required raw data for saving a self-included model"""
        return self.model.keras_model.get_weights()

    def predict(self, x, single_class=False, max_mask_elements=None, *args, **kwargs):
        """
        MaskRCNN internal model output:
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks

        :param single_class: (bool) Predictions belong only to one class. Masks will be collapsed into a single one
        :param max_mask_elements (int) Max number of elements to have unique mask
        """
        single_class = single_class if single_class else self.single_class
        max_mask_elements = max_mask_elements if max_mask_elements else self.max_mask_elements
        preds = self.model.detect([x], verbose=1)
        objs = []

        if single_class:
            mask_collapsed = np.sum(preds[0]['masks'], axis=2)
            mask_collapsed = np.clip(mask_collapsed, 0, 1)
            mask_geometry = Mask(mask_collapsed, preds[0]['rois'])

            objs.append(Object(
                geometry=mask_geometry,
                label=self.labels[preds[0]['class_ids'][0]],
                score=np.mean(preds[0]['scores'])))
        else:
            single_mask = None
            if max_mask_elements:
                if len(preds[0]['class_ids']) > max_mask_elements:
                    # use a unique mask for all
                    mask_collapsed = np.sum(preds[0]['masks'], axis=2)
                    mask_collapsed = np.clip(mask_collapsed, 0, 1)
                    single_mask = Mask(mask_collapsed, preds[0]['rois'])

            for i in range(preds[0]['class_ids'].shape[0]):
                if single_mask:
                    mask_geometry = single_mask
                else:
                    mask_geometry = Mask(preds[0]['masks'][:,:,i], [preds[0]['rois'][i]])
                objs.append(Object(
                    geometry=mask_geometry,
                    label=self.labels[preds[0]['class_ids'][i]],
                    score=preds[0]['scores'][i]))

        return objs


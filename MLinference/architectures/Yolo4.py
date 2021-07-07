"""
Yolo4 tf-lite
Model used for inference.
Extend MLcommon inference abstract class and return MLgeometry objects

* Model parameters:
- filepath (str) file path to:
    - .tflite
    - Path with .pb
- Labels: {idx:'label'}
- input_size: trained image size
- score_threshold: minimum prediction score
- overlapThresh: Minimum IOU of the same class to be considered the same object
- backend: (str) use auxiliary functions from tensorflow to predict

predict function recieves im or list of images

"""
import numpy as np
import cv2 as cv
import sys
import logging
import os

try:
    import tensorflow as tf
except Exception:
    import tflite_runtime.interpreter as tflite

from MLgeometry import Object
from MLgeometry import BoundBox
from MLcommon import InferenceModel


class Yolo4(InferenceModel):
    def __init__(self, filepath, labels=None, input_size=416, score_threshold=0.25, overlapThresh=0.5, backend='tf', *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.score_threshold = score_threshold
        self.overlapThresh = overlapThresh
        self.labels = {int(idx):label for idx,label in labels.items()} if labels else None# fix string idx to int
        self.input_size = input_size

        self.backend = backend
        is_path = os.path.isdir(filepath) 
        self.model = None # Only used when is loaded from .pb 
        if is_path:
            # If filepath is directory load entire saved model
            self.model = tf.keras.models.load_model(filepath)
            self.logger.info(f'Loaded .pb model from {filepath}')
            self.predict = self.tf_predict
            
        else:
            if 'tensorflow' in sys.modules.keys() and backend=='tf':
                try:
                    self.interpreter = tf.lite.Interpreter(model_path=filepath)
                except AttributeError as e:
                    self.logger.error('Cant not use tf backend. Try to update Tensorflow>2.2. Error: {}'.format(e))
                    import tflite_runtime.interpreter as tflite
                    self.interpreter = tflite.Interpreter(model_path=filepath)
                    self.backend = 'tflite'
            else:
                try:
                    import tflite_runtime.interpreter as tflite
                    self.interpreter = tflite.Interpreter(model_path=filepath)
                    self.backend = 'tflite'
                except Exception as e:
                    self.logger.error('Cant not use tflite backend. Error: {}'.format(e))
                    self.interpreter = tf.lite.Interpreter(model_path=filepath)
                    self.backend = 'tf'
            self.logger.info(f'Using {self.backend} backend')
            if self.backend == 'tf': self.predict = self.tf_predict

            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.logger.info(f'Loaded .tflite model from: {filepath}')


    def filter_boxes(self, box_xywh, scores, score_threshold=0.4, input_shape = [416,416]):
        scores_max = scores.max(axis=-1)
        mask = scores_max >= score_threshold
        class_boxes = box_xywh[mask==True]
        pred_conf = scores[mask==True]

        class_boxes = np.reshape(class_boxes, [scores.shape[0], -1, class_boxes.shape[-1]] )
        pred_conf = np.reshape(pred_conf, [scores.shape[0], -1, pred_conf.shape[-1]])

        box_xy, box_wh = np.split(class_boxes, 2, axis=-1)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = np.array(input_shape)

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = np.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        return (boxes, pred_conf)


    def predict(self, im, custom_labels=None, *args, **kwargs):
        """Prediction using tflite interpreter"""
        if self.labels:
            labels = dict(self.labels)
            if custom_labels:
                labels.update(custom_labels)
                self.logger.info('Using custom labels for prediction')
        elif not self.labels and custom_labels:
            labels = custom_labels
        else:
            labels = None

        # TODO: Make compatible with multiple images!
        original_image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        image_data = cv.resize(original_image, (self.input_size, self.input_size))
        image_data = image_data / 255.
        images_data = [image_data]
        images_data = np.asarray(images_data).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], images_data)
        self.interpreter.invoke()

        pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]

        boxes, pred_conf = self.filter_boxes(pred[0], pred[1], score_threshold=self.score_threshold, input_shape=[self.input_size, self.input_size])
        image_h, image_w, _ = im.shape
        pred_fixed = self.non_max_suppression(boxes[0], pred_conf[0], image_w, image_h, overlapThresh=self.overlapThresh)

        # TODO: Use tf_cast_bbox code
        res = []
        if pred_fixed:
            boxes, classes, scores = pred_fixed
            for i in range(boxes.shape[0]):
                coor = boxes[i]
                coor[0] = coor[0] * image_h # y1
                coor[2] = coor[2] * image_h # y2
                coor[1] = coor[1] * image_w # x1
                coor[3] = coor[3] * image_w # x2

                try:
                    lbl = str(labels[classes[i]]) if labels else str(classes[i])
                except KeyError:
                    self.logger.error('Custom labels not provide name for {}. Using default'.format(classes[i]))
                    lbl = str(classes[i])

                new_obj = Object(
                    BoundBox(int(coor[1]), int(coor[0]), int(coor[3]), int(coor[2])),
                    lbl,
                    float(scores[i]),
                    subobject=None)
                res.append(new_obj)
        return res
    
    def non_max_suppression(self, boxes, score, width, height, overlapThresh=0.4):
        """Apply Non Maximum Suppression on detector boxes based on Malisiewicz et al."""
        classes = np.argmax(score, -1)
        scores_max = score.max(axis=-1)

        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes	
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,1] * width
        y1 = boxes[:,0] * height
        x2 = boxes[:,3] * width
        y2 = boxes[:,2] * height

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Idx with same class
            same_class = classes[idxs[:last]] == classes[i]
            # Delete only same class overlap
            overlap[same_class==False] = 0

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        if pick:
            return (boxes[pick], classes[pick], scores_max[pick]) #.astype("int")
        else:
            return None
    @classmethod
    def read_class_names(cls, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def tf_filter_boxes(self, box_xywh, scores, score_threshold=0.4, input_shape=[416, 416]):
        input_shape = tf.constant(input_shape)
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        input_shape = tf.cast(input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # return tf.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)

    def tf_predict(self, im, custom_labels=None, *args, **kwargs):
        """Predict using tensorflow backend"""
        if self.labels:
            labels = dict(self.labels)
            if custom_labels:
                labels.update(custom_labels)
                self.logger.info('Using custom labels for prediction')
        elif not self.labels and custom_labels:
            labels = custom_labels
        else:
            labels = None

        # Pack images
        images_data = []
        im = [im] if not isinstance(im, list) else im
        for i in im:
            original_image = cv.cvtColor(i, cv.COLOR_BGR2RGB)
            image_data = cv.resize(original_image, (self.input_size, self.input_size))
            image_data = image_data / 255.
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        
        if self.model:
            pred = self.model.predict(images_data)
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], images_data)
            self.interpreter.invoke()

            pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]

        boxes, pred_conf = self.tf_filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                             input_shape=tf.constant([self.input_size, self.input_size]))

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.5,
            score_threshold=0.25
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        return self.tf_cast_bbox(im, pred_bbox, labels)

    def tf_cast_bbox(self, image, bboxes, labels, show_label=True):
        """Cast to ML geometries"""
        res = []
        for j in range(len(image)):
            im_res = []
            # num_classes = len(labels)
            image_h, image_w, _ = image[j].shape
            out_boxes, out_scores, out_classes, num_boxes = bboxes
            for i in range(num_boxes[j]):
                # if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
                coor = out_boxes[j][i]
                coor[0] = int(coor[0] * image_h)
                coor[2] = int(coor[2] * image_h)
                coor[1] = int(coor[1] * image_w)
                coor[3] = int(coor[3] * image_w)

                score = out_scores[j][i]
                class_ind = int(out_classes[j][i])

                try:
                    lbl = str(labels[class_ind]) if labels else str(class_ind)
                except KeyError:
                    self.logger.error('Custom labels not provide name for {}. Using default'.format(class_ind[i]))
                    lbl = str(class_ind[i])

                new_obj = Object(
                    BoundBox(int(coor[1]), int(coor[0]), int(coor[3]), int(coor[2])),
                    lbl,
                    float(score),
                    subobject=None)
                im_res.append(new_obj)
            res.append(im_res)

        return res






if __name__ == '__main__':
    """Prediction example"""
    # import sys
    # sys.path.append('/misdoc/vaico/mldrawer/')
    # from MLdrawer.drawer import draw

    # img_path = '/home/juanc/Pictures/zap_4.png'
    # im = cv.imread(img_path)
    # labels =Yolo4.read_class_names('test/data/coco.nombres')
    # model = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_sota.tflite',
    #                        labels=None, input_size=416)

    # res = model.predict(im)
    # print(res)

    # draw(res, im)
    # cv.imshow('Prediction',im)
    #
    # cv.waitKey(0) # waits until a key is pressed
    # cv.destroyAllWindows() # destroys the window showing image


    # Load from .pb
    img_path = '/misdoc/datasets/baluarte/02:42:ac:11:00:02/2020-09-03_22:01:48'
    im = cv.imread(img_path)
    model = Yolo4.load(
        '/home/juanc/Downloads/yolov4_personas_nov_lite-20210706T145752Z-001/yolov4_personas_nov_lite',
        labels=None, 
        input_size=608)

    res = model.predict(im)
    print(res)
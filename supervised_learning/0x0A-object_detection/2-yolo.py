#!/usr/bin/env python3
"""
2-yolo.py
Module that defines a class called Yolo
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Class that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo class constructor

        Args:
            self: this
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names used
                          for the Darknet model, listed in order of index,
                          can be found
            class_t: float representing the box score threshold
                     for the initial filtering step
            nms_t: float representing the IOU threshold
                   for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all of the anchor boxes:
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Function that calculates sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Function that process outputs

        Args:
            self: this
            outputs: list of numpy.ndarrays containing the predictions
                     from the Darknet model for a single image:
            image_size (numpy.ndarray) containing the image’s original size
        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """

        image_height, image_width = image_size[0], image_size[1]

        boxes = [output[..., 0:4] for output in outputs]

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            indexes_y = np.arange(grid_height)
            indexes_y = indexes_y.reshape(grid_height, 1, 1)
            cy = c + indexes_y

            indexes_x = np.arange(grid_width)
            indexes_x = indexes_x.reshape(1, grid_width, 1)
            cx = c + indexes_x

            tx = (box[..., 0])
            ty = (box[..., 1])

            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)

            bx = tx_n + cx
            by = ty_n + cy

            bx /= grid_width
            by /= grid_height

            tw = (box[..., 2])
            th = (box[..., 3])

            tw_t = np.exp(tw)
            th_t = np.exp(th)

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = pw * tw_t
            bh = ph * th_t

            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            box[..., 0] = x1 * image_width
            box[..., 1] = y1 * image_height
            box[..., 2] = x2 * image_width
            box[..., 3] = y2 * image_height

        box_confidences = \
            [self.sigmoid(output[..., 4, np.newaxis]) for output in outputs]
        box_class_probs = \
            [self.sigmoid(output[..., 5:]) for output in outputs]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Function that filter boxes

        Args:
            boxes: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output
            box_confidences: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output
            box_class_probs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the processed box class probabilities for each output

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores):
        """
        obj_thresh = self.class_t

        box_scores_full = []
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            box_scores_full.append(box_conf * box_class_prob)

        box_scores_list = [score.max(axis=3) for score in box_scores_full]
        box_scores_list = [score.reshape(-1) for score in box_scores_list]
        box_scores = np.concatenate(box_scores_list)

        index_to_delete = np.where(box_scores < obj_thresh)

        box_scores = np.delete(box_scores, index_to_delete)

        box_classes_list = [box.argmax(axis=3) for box in box_scores_full]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, index_to_delete)

        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, index_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores

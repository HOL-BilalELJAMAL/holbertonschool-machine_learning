#!/usr/bin/env python3
"""
7-yolo.py
Module that defines a class called Yolo
"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os


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

    @staticmethod
    def iou(box1, box2):
        """Function that calculates intersection over union (x1, y1, x2, y2)"""
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Function that performs a non max suppression

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes
            box_classes: numpy.ndarray of shape (?,)
            containing the class number for the class that
            filtered_boxes predicts
            box_scores: numpy.ndarray of shape (?)
            containing the box scores for each box in filtered_boxes

        Returns:
            tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predicted_box_classes = np.array([box_classes[i] for i in index])
        predicted_box_scores = np.array([box_scores[i] for i in index])

        _, class_counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        accumulated_count = 0

        for class_count in class_counts:
            while i < accumulated_count + class_count:
                j = i + 1
                while j < accumulated_count + class_count:
                    tmp = self.iou(box_predictions[i],
                                   box_predictions[j])
                    if tmp > self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accumulated_count += class_count

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Function that load images

        Args:
            folder_path: string representing the path to the folder
            holding all the images to load

        Returns:
            Tuple of (images, image_paths):
                images: a list of images as numpy.ndarrays
                image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(image) for image in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        Function that preprocess the images

        Args:
            images: list of images as numpy.ndarray

        Returns:
            Tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                input_w: the input width for the Darknet model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2)
                containing the original height and width of the images
                2 => (image_height, image_width)
        """
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        pimages_list = []
        image_shapes_list = []

        for img in images:
            img_shape = img.shape[0], img.shape[1]
            image_shapes_list.append(img_shape)

            dim = (input_w, input_h)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

            pimage = resized / 255
            pimages_list.append(pimage)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Function that displays the image with all boundary boxes, class names,
        and box scores

        Args:
            image: numpy.ndarray containing an unprocessed image
            boxes: numpy.ndarray containing the boundary boxes for the image
            box_classes: numpy.ndarray containing class indices for each box
            box_scores: numpy.ndarray containing the box scores for each box
            file_name: file path where the original image is stored
        """
        for i in range(len(boxes)):
            score = "{:.2f}".format(box_scores[i])

            start_point = (int(boxes[i, 0]), int(boxes[i, 1]))
            end_point = (int(boxes[i, 2]), int(boxes[i, 3]))
            color = (255, 0, 0)

            thickness = 2

            image = cv2.rectangle(image,
                                  start_point, end_point,
                                  color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(boxes[i, 0]), int(boxes[i, 1] - 5))
            fontScale = 0.5

            color = (0, 0, 255)

            thickness = 1
            image = cv2.putText(image,
                                self.class_names[box_classes[i]] + score,
                                org, font, fontScale, color, thickness,
                                cv2.LINE_AA)

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)
        if key == ord('s'):
            os.mkdir('detections') if not os.path.isdir('detections') else None

            os.chdir('detections')

            cv2.imwrite(file_name, image)

            os.chdir('../')
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Function that predicts the image

        Args:
            folder_path: string representing the path
            to the folder holding all the images to predict

        Returns
            Tuple of (predictions, image_paths):
            predictions: a list of tuples for each image of
            (boxes, box_classes, box_scores)
            image_paths: a list of image paths corresponding to each
            prediction in predictions
        """
        predictions = []

        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        for i in range(pimages.shape[0]):
            current_out = [out[i] for out in outputs]

            boxes, box_confidences, box_class_probs = \
                self.process_outputs(current_out, image_shapes[i])

            filtered_boxes, box_classes, box_scores = \
                self.filter_boxes(boxes, box_confidences, box_class_probs)

            box_predictions, predicted_box_classes, predicted_box_scores = \
                self.non_max_suppression(filtered_boxes,
                                         box_classes,
                                         box_scores)

            file_name = image_paths[i].split('/')[-1]
            self.show_boxes(images[i], box_predictions,
                            predicted_box_classes,
                            predicted_box_scores,
                            file_name)

            predictions.append((box_predictions,
                                predicted_box_classes,
                                predicted_box_scores))

        return predictions, image_paths

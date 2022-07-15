from cv2 import imshow
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import collections
import json

import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetectionSystem:
    global category_index
    global detection_model
    def __init__(self) -> None:
        utils_ops.tf = tf.compat.v1
        tf.gfile = tf.io.gfile
        PATH_TO_LABELS = os.path.join(pathlib.Path(__file__).parent.resolve(), 'models/research/object_detection/data/mscoco_label_map.pbtxt')
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.detection_model = self.load_model(model_name)

    def load_model(self, model_name):
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=model_name, 
            origin=base_url + model_file,
            untar=True)
        model_dir = pathlib.Path(model_dir)/"saved_model"
        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']
        return model

    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
        return output_dict

    def visualize_labels_on_image_array(
        self,
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_scores=False,
        skip_labels=False):

        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_instance_boundaries_map = {}
        box_to_keypoints_map = collections.defaultdict(list)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if instance_boundaries is not None:
                    box_to_instance_boundaries_map[box] = instance_boundaries[i]
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ''
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in category_index.keys():
                                class_name = category_index[classes[i]]['name']
                            else:
                                class_name = 'N/A'
                            display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = '{}%'.format(int(100*scores[i]))
                        else:
                            display_str = dict(
                                {
                                    "object": display_str,
                                    "percentage": int(100*scores[i])
                                }
                             )
                            # display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                            box_to_display_str_map["detected"].append(display_str)
        convert_to_dict = dict(box_to_display_str_map)
        return convert_to_dict


    def get_array_result_of_detection(self, model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
        image_np = image_path
        image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        visualize_data_array = self.visualize_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=2)

        return visualize_data_array

    def show_inference(self, model, image_path):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = image_path
        image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=2)
        image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
        return image_np

    def DetectObjects(self, imagepath, Loadjson=False):
        img=cv2.imread(str(imagepath))
        detection_result= self.get_array_result_of_detection(self.detection_model,img)
        if Loadjson != False:
            detection_result = json.dumps(detection_result, indent=2)
        return detection_result

    def DetectObjectsAndSaveImage(self, imagepath, imagePathforSave, Loadjson=False):
        img=cv2.imread(str(imagepath))
        detection_result= self.get_array_result_of_detection(self.detection_model,img)
        detected_img= self.show_inference(self.detection_model,img)
        status = cv2.imwrite(imagePathforSave, detected_img)
        if Loadjson != False:
            detection_result = json.dumps(detection_result, indent=2)
        return detection_result, status
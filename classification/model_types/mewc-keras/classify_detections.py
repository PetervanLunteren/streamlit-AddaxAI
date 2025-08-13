# Script to further identify MD animal detections using keras classification models trained via MEWC
# MEWC - Mega Efficient Wildlife Classifier - University of Tasmania
# https://github.com/zaandahl/mewc

# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI
# Written by Peter van Lunteren
# Latest edit by Peter van Lunteren on 13 May 2025

#############################################
############### MODEL GENERIC ###############
#############################################
# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Classification inference script for AddaxAI models')
parser.add_argument('--model-path', required=True, help='Path to the classification model file')
parser.add_argument('--json-path', required=True, help='Path to the JSON file with detection results')

args = parser.parse_args()
cls_model_fpath = args.model_path
json_path = args.json_path

# lets not freak out over truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

##############################################
############### MODEL SPECIFIC ###############
##############################################
# imports
import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
from keras import saving
os.environ["KERAS_BACKEND"] = "jax"

# load model
animal_model = saving.load_model(cls_model_fpath, compile=False)
img_size = 384

# check GPU availability (tensorflow does support the GPU on Windows Native)
GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False

# read label map
def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
class_map = read_yaml(os.path.join(os.path.dirname(cls_model_fpath), "class_list.yaml"))
inv_class = {v: k for k, v in class_map.items()}
class_ids = sorted(inv_class.values())

# apparently the class_list.yaml can be formatted differently
def can_all_keys_be_converted_to_int(d):
    for key in d.keys():
        try:
            int(key)
        except ValueError:
            return False
    return True

# take note how it is formatted
if not can_all_keys_be_converted_to_int(class_map):
    formatted_int_label = False
else:
    formatted_int_label = True

# extra processing step if the class_list is formatted as int:label
if formatted_int_label:    
    class_ids = [class_map[i] for i in sorted(inv_class.values())]

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    img = np.array(PIL_crop)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    pred = animal_model.predict(img, verbose=0)[0]
    classifications = []
    for i in range(len(pred)):
        classifications.append([class_ids[i], float(pred[i])])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# mewc snipping method taken from https://github.com/zaandahl/mewc-snip/blob/main/src/mewc_snip.py#L29
# which points to this MD util https://github.com/agentmorris/MegaDetector/blob/main/megadetector/visualization/visualization_utils.py#L352
# the function below is rewritten for a single image input without expansion
def get_crop(image, bbox): 
    x1, y1, w_box, h_box = bbox
    ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    left = max(left,0); right = max(right,0)
    top = max(top,0); bottom = max(bottom,0)
    left = min(left,im_width-1); right = min(right,im_width-1)
    top = min(top,im_height-1); bottom = min(bottom,im_height-1)
    image_cropped = image.crop((left, top, right, bottom))
    # resizing will be done in get_classification()
    return image_cropped


#############################################
############### MODEL GENERIC ###############
#############################################
# run main function
import classification.cls_inference as ea

ea.create_raw_classifications(json_path= json_path,
                               GPU_availability= GPU_availability,
                               crop_function=get_crop,
                               inference_function=get_classification,)

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
parser.add_argument('--country', default=None, help='Country code for geofencing (e.g., "USA", "KEN")')
parser.add_argument('--state', default=None, help='State code for geofencing (e.g., "CA", "TX" - US only)')

args = parser.parse_args()
cls_model_fpath = args.model_path
json_path = args.json_path
country = args.country
state = args.state

# lets not freak out over truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

##############################################
############### MODEL SPECIFIC ###############
##############################################
# imports
# import os
# import cv2
# import yaml
# import numpy as np
# import tensorflow as tf
# from keras import saving
# os.environ["KERAS_BACKEND"] = "jax"


# Disable GPU and Metal backend on macOS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_MLIR_ENABLE_TRACING"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# os.environ["TF_XLA_FLAGS"] = "--xla_disable_hlo_passes=memory_space_assignment"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=memory_space_assignment"

os.environ["TF_DISABLE_MLIR_BRIDGE"] = "1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_METAL_ENABLE"] = "0"
os.environ["TF_DEVICE_ALLOCATION_POLICY"] = "0"



# Libraries

import json
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
from tqdm import tqdm


import tensorflow as tf
from tensorflow import keras


# cls_model_dir_fpath = os.path.dirname(cls_model_fpath)


# model = tf.keras.models.load_model(cls_model_dir_fpath)



model = tf.keras.models.load_model("/Applications/AddaxAI_files/models/cls/ANDREA/ANDREA_model.keras")

model.summary()

exit()


import tensorflow as tf

pb_model_path = "/Applications/AddaxAI_files/models/cls/ANDREA/TropiCam_AI_model"

# model_path = "/Applications/AddaxAI_files/models/cls/ANDREA"
model = tf.keras.models.load_model(pb_model_path)

# Test: print model summary
model.summary()


# model.save(os.path.join("/Applications/AddaxAI_files/models/cls/ANDREA/TropiCam_AI_model.h5"), save_format='h5')



model.save("/Applications/AddaxAI_files/models/cls/ANDREA_model.keras")

exit()
# EXECUTED: classify_detections({'json_fpath': '/Users/peter/Desktop/Arboreal MODEL/test-imgs/image_recognition_file.json', 'data_type': 'img', 'simple_mode': False})

# export PYTORCH_ENABLE_MPS_FALLBACK=1 && '/Applications/AddaxAI_files/envs/env-tensorflow-v1/bin/python' '/Applications/AddaxAI_files/AddaxAI/classification_utils/model_types/andrea-anpc/classify_detections.py' '/Applications/AddaxAI_files' '/Applications/AddaxAI_files/models/cls/ANDREA/saved_model.pb' '0.2' '0.5' 'False' '/Users/peter/Desktop/Arboreal MODEL/test-imgs/image_recognition_file.json' 'None' 'False' '0'
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  convnext_base (Functional)  (None, 7, 7, 1024)        87566464  
                                                                 
#  global_average_pooling2d (  (None, 1024)              0         
#  GlobalAveragePooling2D)                                         
                                                                 
#  dense (Dense)               (None, 2048)              2099200   
                                                                 
#  dropout (Dropout)           (None, 2048)              0         
                                                                 
#  dense_1 (Dense)             (None, 84)                172116    
                                                                 
# =================================================================
# Total params: 89837780 (342.70 MB)
# Trainable params: 38223956 (145.81 MB)
# Non-trainable params: 51613824 (196.89 MB)
# _________________________________________________________________









exit()


# exit()



# keras_model = os.path.join(cls_model_dir_fpath, "model_without_xla.keras")
# h5_model = os.path.join(cls_model_dir_fpath, "model_without_xla.h5")

# model = tf.keras.models.load_model(keras_model)
# model.save(h5_model)

# exit()

# model.save(keras_model, save_format="keras")

# model = load_model(keras_model)

# exit()


# from keras.layers import TFSMLayer

# model = TFSMLayer(cls_model_fpath, call_endpoint="serving_default")


# Load model and taxonomy
taxonomy_path = os.path.join(os.path.dirname(cls_model_fpath), "taxon-mapping.csv")
# model = load_model(cls_model_fpath)
taxonomy_df = pd.read_csv(taxonomy_path)
class_names = taxonomy_df['model_class'].tolist()



# ValueError: File format not supported: filepath=/Applications/AddaxAI_files/models/cls/ANDREA/saved_model.pb. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/Applications/AddaxAI_files/models/cls/ANDREA/saved_model.pb, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

# I have the foloowing files:
    
# /Applications/AddaxAI_files/models/cls/ANDREA/keras_metadata.pb
# /Applications/AddaxAI_files/models/cls/ANDREA/saved_model.pb
# /Applications/AddaxAI_files/models/cls/ANDREA/variables/variables.data-00000-of-00001
# /Applications/AddaxAI_files/models/cls/ANDREA/variables/variables.index



# # load model
# animal_model = saving.load_model(cls_model_fpath, compile=False)
# img_size = 384

# check GPU availability (tensorflow does support the GPU on Windows Native)
GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False

# # read label map
# def read_yaml(file_path):
#     with open(file_path, 'r') as f:
#         return yaml.safe_load(f)
# class_map = read_yaml(os.path.join(os.path.dirname(cls_model_fpath), "class_list.yaml"))
# inv_class = {v: k for k, v in class_map.items()}
# class_ids = sorted(inv_class.values())

# # apparently the class_list.yaml can be formatted differently
# def can_all_keys_be_converted_to_int(d):
#     for key in d.keys():
#         try:
#             int(key)
#         except ValueError:
#             return False
#     return True

# # take note how it is formatted
# if not can_all_keys_be_converted_to_int(class_map):
#     formatted_int_label = False
# else:
#     formatted_int_label = True

# # extra processing step if the class_list is formatted as int:label
# if formatted_int_label:    
#     class_ids = [class_map[i] for i in sorted(inv_class.values())]

def preprocess_crop(cropped_img, target_size=(224, 224)):
    img_resized = cropped_img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.keras.applications.convnext.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# def classify_crop(model, img_array, taxonomy_df, class_names, tropicam_conf_threshold=0.75):
#     prediction = model.predict(img_array, verbose=0)[0]
#     result = {}
#     taxonomic_levels = ['species', 'genus', 'family', 'order', 'class']
    
#     for level in taxonomic_levels:
#         unique_labels = sorted(taxonomy_df[level].unique())
#         aggregated = np.zeros(len(unique_labels))
        
#         for _, row in taxonomy_df.iterrows():
#             species_idx = class_names.index(row['species'])
#             level_idx = unique_labels.index(row[level])
#             aggregated[level_idx] += prediction[species_idx]
        
#         pred_idx = np.argmax(aggregated)
#         confidence = np.clip(aggregated[pred_idx], 0.0, 1.0)
#         result[f'pred_{level}'] = unique_labels[pred_idx]
#         result[f'conf_{level}'] = confidence
    
#     for level in taxonomic_levels:
#         if result[f'conf_{level}'] >= tropicam_conf_threshold:
#             result['best_prediction'] = result[f'pred_{level}']
#             result['best_confidence'] = result[f'conf_{level}']
#             result['taxo_level'] = level
#             break
#     else:
#         result['best_prediction'] = 'Uncertain'
#         result['best_confidence'] = 0.0
#         result['taxo_level'] = 'None'
    
#     return result



# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    # img = np.array(PIL_crop)
    # img = cv2.resize(img, (img_size, img_size))
    # img = np.expand_dims(img, axis=0)
    # pred = animal_model.predict(img, verbose=0)[0]
    # classifications = []
    # for i in range(len(pred)):
    #     classifications.append([class_ids[i], float(pred[i])])
    # return classifications



    img_array = preprocess_crop(PIL_crop)
    prediction = model.predict(img_array, verbose=0)[0]
    # classification = classify_crop(model, img_array, taxonomy_df, class_names, tropicam_conf_threshold)
    print(f"Prediction shape: {prediction.shape}")
    print(f"prediction: {prediction}")


# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# mewc snipping method taken from https://github.com/zaandahl/mewc-snip/blob/main/src/mewc_snip.py#L29
# which points to this MD util https://github.com/agentmorris/MegaDetector/blob/main/megadetector/visualization/visualization_utils.py#L352
# the function below is rewritten for a single image input without expansion

# def get_crop(image, bbox): 
#     x1, y1, w_box, h_box = bbox
#     ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box
#     im_width, im_height = image.size
#     (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                                     ymin * im_height, ymax * im_height)
#     left = max(left,0); right = max(right,0)
#     top = max(top,0); bottom = max(bottom,0)
#     left = min(left,im_width-1); right = min(right,im_width-1)
#     top = min(top,im_height-1); bottom = min(bottom,im_height-1)
#     image_cropped = image.crop((left, top, right, bottom))
#     # resizing will be done in get_classification()
#     return image_cropped

def get_crop(img, bbox):
    # try:
    # with Image.open(image_path) as img:
    w, h = img.size
    left, top = int(bbox[0] * w), int(bbox[1] * h)
    right, bottom = int((bbox[0] + bbox[2]) * w), int((bbox[1] + bbox[3]) * h)
    return img.crop((max(0, left), max(0, top), min(w, right), min(h, bottom)))
    # except Exception as e:
    #     print(f"Error cropping {image_path}: {e}")
    #     return None


#############################################
############### MODEL GENERIC ###############
#############################################
# run main function
import classification.cls_inference as ea

model_id = os.path.basename(os.path.dirname(cls_model_fpath))

ea.create_raw_classifications(json_path= json_path,
                               GPU_availability= GPU_availability,
                               crop_function=get_crop,
                               inference_function=get_classification,
                               model_id=model_id)

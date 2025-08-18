# Script to further identify MD animal detections using a pytorch-wildlife classification model
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# Script by Peter van Lunteren
# Model by Pytorch-Wildlife (https://github.com/microsoft/CameraTraps)
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

# set working directory to file location
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# import packages
from PytorchWildlife.models import classification as pw_classification
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import json
import supervision as sv
import numpy as np
import torch
from PytorchWildlife.data import transforms as pw_trans
trans_clf = pw_trans.Classification_Inference_Transform(target_size=224)

# load model
classification_model = pw_classification.AI4GAmazonRainforest(weights = cls_model_fpath)

# check GPU availability
GPU_availability = False
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
except:
    pass
if not GPU_availability:
    GPU_availability = torch.cuda.is_available()

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    preprocessed_crop = trans_clf(PIL_crop)
    
    if isinstance(preprocessed_crop, torch.Tensor): # Convert tensor to numpy array for PIL compatibility
        preprocessed_crop = preprocessed_crop.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
    classifications = classification_model.single_image_classification(preprocessed_crop)['all_confidences']
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img, bbox_norm):

    # convert bbox to int
    img_width, img_height = img.size
    left = int(round(bbox_norm[0] * img_width))
    top = int(round(bbox_norm[1] * img_height))
    right = int(round(bbox_norm[2] * img_width)) + left
    bottom = int(round(bbox_norm[3] * img_height)) + top

    # crop using supervision method
    crop = Image.fromarray(sv.crop_image(np.array(img.convert("RGB")), xyxy = [left, top, right, bottom]))
    return crop

#############################################
############### MODEL GENERIC ###############
#############################################
# run main function
import classification.cls_inference as ea

ea.create_raw_classifications(json_path= json_path,
                               GPU_availability= GPU_availability,
                               crop_function=get_crop,
                               inference_function=get_classification,)

# Script to further identify MD animal detections using a yolov8 classification model
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# Written by Peter van Lunteren
# Latest edit by Peter van Lunteren on 13 May 2025

#############################################
############### MODEL GENERIC ###############
#############################################
# Parse command line arguments
import sys
import argparse

# Global variables for arguments
cls_model_fpath = None
json_path = None
country = None
state = None

# Only parse args if running as main script or if args are provided
if __name__ == '__main__' or len(sys.argv) > 1:
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
from ultralytics import YOLO
import torch
from PIL import ImageOps

# make sure windows trained models work on unix too
import pathlib
import platform
plt = platform.system()
if plt != 'Windows': pathlib.WindowsPath = pathlib.PosixPath

# Global variables for model (will be initialized when needed)
animal_model = None

# check GPU availability
GPU_availability = False
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
except:
    pass
if not GPU_availability:
    GPU_availability = torch.cuda.is_available()

def load_model():
    """Load the model if not already loaded."""
    global animal_model, cls_model_fpath
    
    if animal_model is not None:
        return  # Model already loaded
    
    # Ensure we have a model path
    if cls_model_fpath is None:
        raise ValueError("Model path not set. Cannot load model.")
    
    # load model
    animal_model = YOLO(cls_model_fpath)

# read label map
# # not neccesary for yolov8 models to retrieve label map exernally, as it is incorporated into the model itself

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    try:
        load_model()  # Ensure model is loaded
    except Exception as e:
        return []
    
    try:
        results = animal_model(PIL_crop, verbose=False)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        classifications = []
        for idx, v in names_dict.items():
            classifications.append([v, float(probs[idx])])  # Convert numpy float32 to Python float
        return classifications
    except Exception as e:
        return []

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img, bbox_norm): # created by Dan Morris
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    box_size = max(box_w, box_h)
    box_size = pad_crop(box_size)
    xmin = max(0, min(
        xmin - int((box_size - box_w) / 2),
        img_w - box_w))
    ymin = max(0, min(
        ymin - int((box_size - box_h) / 2),
        img_h - box_h))
    box_w = min(img_w, box_size)
    box_h = min(img_h, box_size)
    if box_w == 0 or box_h == 0:
        return
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
    crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
    return crop

# make sure small animals are not overly enlarged
def pad_crop(box_size):
    input_size_network = 224
    default_padding = 30
    diff_size = input_size_network - box_size
    if box_size >= input_size_network:
        box_size = box_size + default_padding
    else:
        if diff_size < default_padding:
            box_size = box_size + default_padding
        else:
            box_size = input_size_network    
    return box_size

#############################################
############### MODEL GENERIC ###############
#############################################
# run main function only when script is executed directly
if __name__ == '__main__':
    load_model()  # Load model for direct execution
    import classification.cls_inference as ea
    
    ea.create_raw_classifications(json_path= json_path,
                                   GPU_availability= GPU_availability,
                                   crop_function=get_crop,
                                   inference_function=get_classification,)

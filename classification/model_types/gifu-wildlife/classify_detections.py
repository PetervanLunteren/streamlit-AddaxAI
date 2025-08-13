# Script to further identify MD animal detections using a Resnet50 classification models for Japan - Gifu area
# https://github.com/gifu-wildlife/TrainingMdetClassifire

# It constsist of code that is specific for this kind of model architechture, and
# code that is generic for all model architectures that will be run via AddaxAI.

# Written by Peter van Lunteren, modified by Masaki Ando
# Latest edit by Peter van Lunteren on 26 May 2025

#############################################
############### MODEL GENERIC ###############
#############################################

# import packages
import AddaxAI.classification_utils.inference_lib as ea
import platform
import pathlib
from torchvision.models import resnet
from torchvision import transforms
import omegaconf
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch
import os
from PIL import ImageFile
import sys

# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Classification inference script for AddaxAI models')
parser.add_argument('--model-path', required=True, help='Path to the classification model file')
parser.add_argument('--json-path', required=True, help='Path to the JSON file with detection results')

args = parser.parse_args()
cls_model_fpath = args.model_path
json_path = args.json_path

# lets not freak out over truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

##############################################
############### MODEL SPECIFIC ###############
##############################################

# class to load custom ResNet50 model
class CustomResNet50(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, device_str='cpu'):
        '''
        Constructor of the model. Loads ResNet50, optionally with weights from a local file.
        '''
        super(CustomResNet50, self).__init__()

        # Load model without downloading weights
        self.model = resnet.resnet50(weights=None)

        # If path is provided, load weights from local file
        if pretrained_path is not None:
            state_dict = torch.load(
                pretrained_path, map_location=torch.device(device_str))
            self.model.load_state_dict(state_dict)

        # Replace final classification layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# make sure windows trained models work on unix too
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# check GPU availability
GPU_availability = False
device_str = 'cpu'
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
        device_str = 'mps'
except:
    pass
if not GPU_availability:
    if torch.cuda.is_available():
        GPU_availability = True
        device_str = 'cuda'

# load model
class_csv_fpath = os.path.join(os.path.dirname(cls_model_fpath), 'classes.csv')
classes = pd.read_csv(class_csv_fpath)
pretrained_weights_path = os.path.join(
    os.path.dirname(cls_model_fpath), 'resnet50-11ad3fa6.pth')
model = CustomResNet50(num_classes=len(
    classes), pretrained_path=pretrained_weights_path, device_str=device_str)
checkpoint = torch.load(cls_model_fpath, map_location=torch.device(device_str))
model.load_state_dict(checkpoint['state_dict'])
model.to(torch.device(device_str))
model.eval()
device = torch.device(device_str)

# image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
def get_classification(PIL_crop):
    input_tensor = preprocess(PIL_crop)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    output = model(input_batch)
    probabilities = F.softmax(output, dim=1)
    probabilities_np = probabilities.cpu().detach().numpy()
    confidence_scores = probabilities_np[0]
    classifications = []
    for i in range(len(confidence_scores)):
        pred_class = classes.iloc[i].values[1]
        pred_conf = confidence_scores[i]
        classifications.append([pred_class, pred_conf])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img, bbox_norm):
    buffer = 0
    width, height = img.size
    bbox1, bbox2, bbox3, bbox4 = bbox_norm
    left = width * bbox1
    top = height * bbox2
    right = width * (bbox1 + bbox3)
    bottom = height * (bbox2 + bbox4)
    left = max(0, int(left) - buffer)
    top = max(0, int(top) - buffer)
    right = min(width, int(right) + buffer)
    bottom = min(height, int(bottom) + buffer)
    image_cropped = img.crop((left, top, right, bottom))
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

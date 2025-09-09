# Script to further identify MD animal detections using PT classification models
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# It is designed for models developed by the San Diego Zoo Wildlife Alliance fork 
# https://github.com/PetervanLunteren/animl-py

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
import os
import torch
import pandas as pd
import torch.nn as nn
from PIL import ImageOps
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet, convnext_base, ConvNeXt_Base_Weights

# make sure windows trained models work on unix too
import pathlib
import platform
plt = platform.system()
if plt != 'Windows': pathlib.WindowsPath = pathlib.PosixPath

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

# init model architecture
class EfficientNetV2M(nn.Module):
    def __init__(self, num_classes, tune=True):
        super(EfficientNetV2M, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = efficientnet.efficientnet_v2_m(weights=efficientnet.EfficientNet_V2_M_Weights.DEFAULT)
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        prediction = self.model.classifier(x) 
        return prediction
    
class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes, tune=True):
        super(EfficientNetV2S, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = efficientnet.efficientnet_v2_s(weights=efficientnet.EfficientNet_V2_S_Weights.DEFAULT)
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        prediction = self.model.classifier(x)
        return prediction
    
class ConvNeXtBase(nn.Module):
    def __init__(self, num_classes, tune=True):
        super(ConvNeXtBase, self).__init__()
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        if not tune:
            for param in self.model.parameters():
                param.requires_grad = False
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    def forward(self, x):
        '''
        Forward pass (prediction).
        '''
        return self.model(x)

# load model
checkpoint = torch.load(cls_model_fpath, map_location=torch.device(device_str))
image_size = tuple(checkpoint['image_size'])
architecture = checkpoint['architecture']
categories = checkpoint['categories']
classes = list(categories.keys())
if architecture == 'efficientnet_v2_m':
    model = EfficientNetV2M(len(classes), tune=False)
elif architecture == 'efficientnet_v2_s':
    model = EfficientNetV2S(len(classes), tune=False)
elif architecture == 'convnext_base':
    model = ConvNeXtBase(len(classes), tune=False)
model.load_state_dict(checkpoint['model'])
model.to(torch.device(device_str))
model.eval()
device = torch.device(device_str)

# image preprocessing 
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
def get_classification(PIL_crop):
    try:
        load_model()  # Ensure model is loaded
    except Exception as e:
        return []
    
    try:
        if PIL_crop.mode == "RGBA":
            PIL_crop = PIL_crop.convert("RGB")
        input_tensor = preprocess(PIL_crop)
        input_batch = input_tensor.unsqueeze(0)  
        input_batch = input_batch.to(device)
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)
        probabilities_np = probabilities.cpu().detach().numpy()
        confidence_scores = probabilities_np[0]
        classifications = []
        for i in range(len(confidence_scores)):
            pred_class = classes[i]
            pred_conf = float(confidence_scores[i])  # Convert numpy float32 to Python float
            classifications.append([pred_class, pred_conf])
        return classifications
    except Exception as e:
        return []

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img, bbox_norm):
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    box_size = max(box_w, box_h)
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

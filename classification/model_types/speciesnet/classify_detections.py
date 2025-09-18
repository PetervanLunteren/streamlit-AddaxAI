# Script to further identify MD animal detections using SpeciesNet classification models
# SpeciesNet is Google's image classifier for camera trap images that runs as an ensemble with MegaDetector
# It consists of code that is specific for SpeciesNet architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# Script by Peter van Lunteren
# Model by Google (https://github.com/google/cameratrapai)
# Latest edit by Peter van Lunteren on 18 August 2025

#############################################
############### MODEL GENERIC ###############
#############################################
# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='SpeciesNet classification inference script for AddaxAI models')
parser.add_argument('--model-path', required=True, help='Path to the SpeciesNet classification model directory')
parser.add_argument('--json-path', required=True, help='Path to the JSON file with detection results')
parser.add_argument('--country', default=None, help='Country code for geofencing (e.g., "USA", "KEN")')
parser.add_argument('--state', default=None, help='State code for geofencing (e.g., "CA", "TX" - US only)')

args = parser.parse_args()
cls_model_path = args.model_path
json_path = args.json_path
country = args.country
state = args.state

##############################################
############### MODEL SPECIFIC ###############
##############################################
import os

# SpeciesNet expects a model directory, but the standard interface passes a model file path
# Convert file path to directory path if needed
if os.path.isfile(cls_model_path):
    cls_model_dir = os.path.dirname(cls_model_path)
else:
    cls_model_dir = cls_model_path
import sys
import subprocess

# Get the AddaxAI root directory (assuming this script is in classification/model_types/speciesnet/)
ADDAXAI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_speciesnet_classification():
    """
    Run SpeciesNet classifier using the MegaDetector + SpeciesNet script.
    This adapts the original run_speciesnet function to work with the standard interface.
    """
    # Create output file path (replace input file with speciesnet output)
    output_file = json_path.replace("-in-progress.json", "-speciesnet-output.json")
    
    # Get the folder containing images (parent of the json file)
    deployment_folder = os.path.dirname(json_path)
    
    # Build command for run_md_and_speciesnet
    command = [
        f"{ADDAXAI_ROOT}/envs/env-addaxai-base/bin/python",
        "-m", "megadetector.detection.run_md_and_speciesnet",
        deployment_folder,  # source folder
        output_file,        # output file  
        "--detections_file", json_path,  # skip detection, use existing results
        "--classification_model", cls_model_dir,  # local model directory
        "--loader_workers", "1",  # Reduce workers to avoid multiprocessing issues
        "--classifier_batch_size", "4",  # Smaller batch size for stability
        "--verbose"  # Enable verbose output for debugging
    ]
    
    # Add geofencing parameters if specified
    if country:
        command.extend(["--country", country])
        print(f"SpeciesNet geofencing enabled for country: {country}", flush=True)
        
        if state and country == "USA":
            command.extend(["--admin1_region", state])
            print(f"SpeciesNet geofencing enabled for US state: {state}", flush=True)

    # Log the command for debugging/audit purposes
    print(f"\n\nRunning SpeciesNet command:\n{' '.join(command)}\n", flush=True)

    # Set working directory to project root
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
        universal_newlines=True,
        cwd=ADDAXAI_ROOT  # Set working directory to project root
    )

    # Stream output to console with immediate flushing
    for line in process.stdout:
        line = line.strip()
        print(line, flush=True)

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        print(f"SpeciesNet classification failed with exit code {return_code}.", flush=True)
        sys.exit(return_code)
    
    # Replace the original file with the SpeciesNet output
    if os.path.exists(output_file):
        if os.path.exists(json_path):
            os.remove(json_path)
        os.rename(output_file, json_path)
        print("SpeciesNet classification completed successfully.", flush=True)
    else:
        print("SpeciesNet output file not created.", flush=True)
        sys.exit(1)

#############################################
############### MODEL GENERIC ###############
#############################################
# Run main function
if __name__ == "__main__":
    run_speciesnet_classification()
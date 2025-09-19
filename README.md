# AddaxAI Streamlit Application

A temporary repository to build a new AddaxAI version. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and don’t have to be gentle.

## Prerequisites

**macOS Only**: This application is currently designed for macOS systems only.

No external dependencies required! The application includes its own micromamba binary for managing Python environments.

## Installation

### 1. Clone Repository
Clone this repository to your desired location:
```bash
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
cd streamlit-AddaxAI
```

### 2. Create Environment and Install Packages

Create environment using included micromamba (installs in `./envs/env-addaxai-base`):
```bash
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y
```
Install SpeciesNet with the required flag for macOS (ignore package conflict about `protobuf`):
```bash
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install --use-pep517 speciesnet==5.0.2
```

### 3. Launch Application
Run the application using the created environment:
```bash
./bin/macos/micromamba run -p ./envs/env-addaxai-base streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```plaintext
streamlit-AddaxAI/
├── main.py                  # Application entry point
├── .streamlit/              # Streamlit configuration
│   └── config.toml
├── assets/                  # Static assets
│   ├── css/                 # CSS styles
│   ├── dicts/               # Dictionaries
│   ├── images/              # Logos and icons
│   ├── language/            # Localization files
│   ├── loaders/             # Loading animations
│   ├── logs/                # Log files
│   ├── model_meta/          # Model metadata
│   └── test_images/         # Sample/testing images
├── bin/                     # Executables
│   └── macos/
│       └── micromamba
├── classification/          # ML inference system
│   ├── cls_inference.py     # Generic classification engine
│   └── model_types/         # Model-specific implementations
├── components/              # Reusable UI components
├── docs/                    # Documentation sources
├── envs/                    # Environment definitions
│   └── ymls/
├── pages/                   # Streamlit pages
│   ├── analysis_advanced.py
│   ├── analysis_quick.py
│   ├── camera_management.py
│   ├── depth_estimation.py
│   ├── explore_results.py
│   ├── human_verification.py
│   ├── post_processing.py
│   ├── remove_duplicates.py
│   └── settings.py
├── tests/                   # Unit tests
│   ├── test_components.py
│   └── test_utils.py
├── utils/                   # Utility modules
│   ├── analysis_utils.py
│   ├── common.py
│   ├── config.py
│   ├── data_verification.py
│   ├── folder_selector.py
│   └── huggingface_downloader.py
├── mkdocs.yml               # MkDocs configuration
├── PROJECT_STRUCTURE.md     # Project structure documentation
├── README.md                # Overview and installation
└── DEVELOPERS.md            # Developer guidelines
```
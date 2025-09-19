# AddaxAI Streamlit Application

A temporary repository to build a new AddaxAI version. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can experiment without affecting the original.

## Prerequisites

**macOS Only**: This application is currently designed for macOS systems only.

This project requires the following Python packages, which are installed via the provided environment.yml:

- opencv-python
- pillow
- tqdm

The application includes its own micromamba binary for managing Python environments.

## Installation

### 1. Clone Repository

Clone this repository to your desired location
```bash
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
cd streamlit-AddaxAI
```

### 2. Create Environment and Install Packages

Create environment using included micromamba (installs in `./envs/env-addaxai-base`):
```bash
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y
```

Install SpeciesNet and additional dependencies:
```bash
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install --use-pep517 speciesnet==5.0.2
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install opencv-python pillow tqdm
```

### 3. Launch Application

Run the application using the created environment:
```bash
./bin/macos/micromamba run -p ./envs/env-addaxai-base streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
streamlit-AddaxAI/
├── main.py                    # Application entry point
├── pages/                     # Streamlit pages
│   ├── analysis_advanced.py   # Advanced analysis workflow
│   ├── analysis_quick.py      # Quick analysis interface
│   ├── human_verification.py  # Manual review interface
│   ├── remove_duplicates.py   # Duplicate detection
│   ├── explore_results.py     # Results visualization
│   ├── post_processing.py     # Post-processing tools
│   ├── camera_management.py   # Metadata management
│   └── settings.py            # Application settings
├── components/                # Reusable UI components
├── utils/                     # Core utilities and business logic (includes video utilities)
├── config/                    # Application configuration
├── assets/                    # Static assets (CSS, images, etc.)
├── classification/            # Classification inference system
└── envs/                      # Conda environments
```

## Documentation

- **[DEVELOPERS.md](DEVELOPERS.md)**: Development guidelines and architecture
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure documentation

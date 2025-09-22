# AddaxAI Streamlit Application

A temporary repository to build a new AddaxAI version. Completely separate from its original repo [addaxai](https://github.com/PetervanLunteren/addaxai) so that we can experiment freely without impacting the original codebase.

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

Create the environment using the included micromamba (installs in `./envs/env-addaxai-base`):
```bash
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y
```
Install SpeciesNet with the required macOS flag (ignore any `protobuf` conflict warnings):
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

```
streamlit-AddaxAI/
├── .streamlit/                # Streamlit configuration (page settings, theming)
├── assets/                    # Static assets (CSS, images, dictionaries, loaders)
├── bin/                       # Bundled executables (micromamba)
├── classification/            # Classification inference scripts by model type
├── components/                # Reusable UI components (modals, steppers, helpers)
├── config/                    # Default configuration files (general_settings, explore_results)
├── docs/                      # Documentation (Markdown files for user/dev guides)
├── envs/                      # Environment YAML definitions for micromamba
├── main.py                    # Streamlit application entry point
├── pages/                     # Individual Streamlit pages for each tool
├── requirements.txt           # Consolidated Python dependencies
├── utils/                     # Core utilities and business logic modules
├── tests/                     # Unit tests for components and utilities
└── README.md                  # This file
```

## Documentation

- **[DEVELOPERS.md](DEVELOPERS.md)**: Development guidelines and architecture
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure documentation

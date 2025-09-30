# AddaxAI Streamlit Application

A temporary repository to build a new AddaxAI version. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can experiment without constraints.

## Prerequisites

**macOS Only**: This application is currently designed for macOS systems only.

No external dependencies required! The application includes its own micromamba binary for managing Python environments.

## Installation

### 1. Clone Repository
Clone this repository to your desired location
```bash
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
cd streamlit-AddaxAI
```

### 2. Create Environment and Install Packages

Create environment using included micromamba (installs in `./envs/env-addaxai-base`)

```bash
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y
```
Install SpeciesNet with the required flag for macOS (ignore package conflict about `protobuf`)
```bash
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install --use-pep517 speciesnet==5.0.2
```

### 3. Launch Application
Run the application using the created environment
```bash
./bin/macos/micromamba run -p ./envs/env-addaxai-base streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
streamlit-AddaxAI/
├── .streamlit/                 # Streamlit configuration
├── assets/                     # Static assets (CSS, images, etc.)
├── bin/                        # Bundled binaries (e.g., micromamba)
├── classification/             # Classification inference system
├── components/                 # Reusable UI components
├── utils/                      # Core utilities and business logic
├── envs/                       # Environment YAMLs and environments
├── pages/                      # Streamlit pages
├── tests/                      # Test suite
├── main.py                     # Application entry point
├── mkdocs.yml                  # Documentation configuration
├── DEVELOPERS.md               # Development guidelines
└── PROJECT_STRUCTURE.md        # Detailed project structure documentation
```

## Documentation

- **[DEVELOPERS.md](DEVELOPERS.md)**: Development guidelines and architecture
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure documentation

# Installation

## Prerequisites

**macOS Only**: This application is currently designed for macOS systems only.

No external dependencies required! The application includes its own micromamba binary for managing Python environments.

## Installation Steps

### 1. Clone Repository
Clone this repository to your desired location
```bash
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
cd streamlit-AddaxAI
```

### 2. Create Environment and Install Packages

Create environment using included micromamba (installs in ./envs/env-addaxai-base)

```bash
./bin/darwin/micromamba env create -f envs/ymls/addaxai-base/darwin/environment.yml --prefix ./envs/env-addaxai-base -y
```
Install SpeciesNet with the required flag for macOS (ignore package conflict about `protobuf`)
```bash
./bin/darwin/micromamba run -p ./envs/env-addaxai-base pip install --use-pep517 speciesnet==5.0.2
```

### 3. Launch Application
Run the application using the created environment
```bash
./bin/darwin/micromamba run -p ./envs/env-addaxai-base streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

*For detailed information, see the [README](https://github.com/PetervanLunteren/streamlit-AddaxAI/blob/main/README.md).*
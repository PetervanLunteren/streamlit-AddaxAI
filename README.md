# AddaxAI Streamlit Application

A multi-page Streamlit app for wildlife camera trap analysis. This repo is independent of the original AddaxAI repository.

## Prerequisites

The application supports macOS and Windows. It includes its own micromamba binary for managing Python environments; no external conda installation is required.

## Installation

### 1. Clone Repository

```bash
# Clone this repository to your desired location
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
cd streamlit-AddaxAI
```

### 2. Create Environment and Install Packages

Use the included micromamba to create the base environment:

```bash
# macOS / Linux
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y

# Windows (use .exe)
./bin/macos/micromamba.exe env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y
```

### 3. Launch Application

```bash
# macOS / Linux
./bin/macos/micromamba run -p ./envs/env-addaxai-base streamlit run main.py

# Windows
./bin/macos/micromamba.exe run -p ./envs/env-addaxai-base streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

For an overview of the folder layout and where to find key files, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Documentation

- **[DEVELOPERS.md](DEVELOPERS.md)**: Development guidelines and architecture
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure documentation
- **[CLAUDE.md](CLAUDE.md)**: Context for AI-assisted tasks

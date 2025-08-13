# AddaxAI Streamlit Application

A temporary repository to buid a new AddaxAI version. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle. 

## Prerequisites

**macOS Only**: This application is currently designed for macOS systems only.

No external dependencies required! The application includes its own micromamba binary for managing Python environments.

## Installation

### 1. Clone Repository

```bash
# Clone this repository to your desired location
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
cd streamlit-AddaxAI
```

### 2. Create Environment and Install Packages

The application includes its own micromamba binary, so no external conda installation is needed:

```bash
# Create environment using included micromamba (installs in ./envs/env-addaxai-base)
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base
```

### 3. Launch Application

```bash
# Run the application using the created environment
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
├── utils/                     # Core utilities and business logic
├── config/                    # Application configuration
├── data/                      # Test data and samples
├── assets/                    # Static assets (CSS, images, etc.)
├── models/                    # AI model files
├── classification/            # Classification inference system
└── envs/                      # Conda environments
```

## Documentation

- **[DEVELOPERS.md](DEVELOPERS.md)**: Development guidelines and architecture
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure documentation

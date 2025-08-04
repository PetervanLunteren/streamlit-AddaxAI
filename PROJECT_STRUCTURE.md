# AddaxAI Streamlit Application - Project Structure

This document describes the refactored project structure for the AddaxAI multi-page Streamlit application.

## 📁 Directory Structure

```
streamlit-AddaxAI/
├── main.py                    # Main application entry point
├── PROJECT_STRUCTURE.md       # This documentation file
├── 
├── components/                # Reusable UI components
│   ├── __init__.py           # Component exports
│   ├── progress.py           # MultiProgressBars class
│   ├── stepper.py            # StepperBar class
│   └── ui_helpers.py         # UI helper functions
│
├── pages/                    # Streamlit pages (st.Page navigation)
│   ├── analysis_advanced.py  # Advanced analysis workflow
│   ├── analysis_quick.py     # Quick analysis interface
│   ├── camera_management.py  # Camera metadata management
│   ├── depth_estimation.py   # Depth estimation tool
│   ├── explore_results.py    # Results exploration
│   ├── post_processing.py    # Post-processing operations
│   ├── remove_duplicates.py  # Remove duplicate detections
│   ├── settings.py           # Global settings management
│   └── human_verification.py # Human verification interface
│
├── utils/                    # Utility functions and core logic
│   ├── __init__.py
│   ├── analysis_utils.py     # Advanced analysis utilities
│   ├── common.py             # Shared utility functions
│   ├── config.py             # Global configuration and paths
│   ├── data_verification.py  # Data validation utilities
│   ├── folder_selector.py    # Tkinter folder selection
│   ├── huggingface_downloader.py # HuggingFace model downloading
│   └── init_paths.py         # Path initialization
│
├── .streamlit/               # Streamlit configuration (standard location)
│   └── config.toml           # Theme and UI settings
├── config/                   # Application configuration files
│   ├── general_settings.json # App-wide settings
│   └── analyse_advanced.json # Analysis workflow state
│
├── data/                     # Data files and test data
│   ├── test-images/          # Test images for development
│   ├── sample-data/          # Sample datasets
│   └── results/              # Analysis results
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_components.py    # UI component tests
│   └── test_utils.py         # Utility function tests
│
├── assets/                   # Static assets (unchanged)
│   ├── css/                  # Custom CSS styles
│   ├── images/               # Logos and icons
│   ├── language/             # Internationalization files
│   ├── loaders/              # Loading animations
│   ├── logs/                 # Application logs
│   └── model_meta/           # AI model metadata
│
├── classification/           # ML inference system (unchanged)
│   ├── cls_inference.py      # Generic classification engine
│   └── model_types/          # Model-specific implementations
│
├── models/                   # AI model files (unchanged)
│   ├── cls/                  # Classification models
│   └── det/                  # Detection models
│
├── envs/                     # Conda environments (unchanged)
├── bin/                      # Binaries (unchanged)
└── DEVELOPERS.md             # Developer documentation
```

## 🔧 Key Changes Made

### ✅ New Components Package
- **`components/`**: Extracted reusable UI components from `utils/common.py`
- **`MultiProgressBars`**: Advanced progress tracking with tqdm integration
- **`StepperBar`**: Visual step indicators for workflows
- **UI Helpers**: `print_widget_label`, `info_box`, `warning_box`, `radio_buttons_with_captions`

### ✅ Reorganized Configuration
- **`config/`**: Centralized all configuration files
- **Moved**: `vars/` → `config/`
- **Moved**: `.streamlit/` → `config/streamlit/`
- **Updated**: All path references throughout codebase

### ✅ Improved Data Structure
- **`data/`**: Organized data files by purpose
- **Moved**: `test-imgs/` → `data/test-images/`
- **Added**: `data/sample-data/` and `data/results/` directories

### ✅ Added Testing Infrastructure
- **`tests/`**: Unit test structure for future development
- **Placeholder tests**: For components and utilities
- **Ready for**: pytest, unittest, or other testing frameworks

### ✅ Updated Import Structure
- **Components**: Now imported from `components` package
- **Clean imports**: Removed wildcard imports, organized dependencies
- **Backwards compatible**: Core functionality preserved

## 📦 Import Guide

### Using UI Components
```python
from components import MultiProgressBars, StepperBar, print_widget_label, info_box, warning_box
```

### Using Utilities
```python
from utils.common import load_vars, update_vars, get_session_var, set_session_var
from utils.config import ADDAXAI_FILES, log
```

### Using Analysis Functions
```python
from utils.analysis_utils import load_known_projects, load_model_metadata
```

## 🚀 Benefits of New Structure

1. **Modularity**: UI components separated from business logic
2. **Maintainability**: Clear separation of concerns
3. **Testability**: Dedicated tests directory with proper structure
4. **Collaboration**: Easier for new developers to understand codebase
5. **Scalability**: Ready for future expansion and features

## 🔄 Migration Notes

- **No functionality changes**: All existing features work identically
- **Import updates**: Files updated to use new component structure
- **Path updates**: Configuration paths updated to use `config/` directory
- **Session state**: Improved handling for imports outside Streamlit context

## 🏃‍♂️ Getting Started

The application runs exactly as before:

```bash
cd /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI
conda activate env-streamlit-addaxai
streamlit run main.py
```

All functionality remains unchanged - this refactoring focused purely on code organization and maintainability.
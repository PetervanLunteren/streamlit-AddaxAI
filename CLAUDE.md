# AddaxAI Wildlife Camera Analysis Application - Claude Context

## CRITICAL: Task Workflow Requirements
**ALWAYS ask at least 3 clarifying questions before starting ANY task and WAIT for answers before proceeding.** This is essential for proper task execution. Do not proceed with work until you receive responses to your questions.

## Project Overview
AddaxAI is a Streamlit-based wildlife camera trap analysis platform that automates animal detection and species classification using AI models. It processes camera trap deployments (SD card image collections) through a complete AI-powered workflow, replacing manual review of thousands of images.

**Core Value**: Automates the bottleneck in wildlife research - manually reviewing camera trap images - enabling researchers to focus on conservation decisions rather than image sorting.

## Application Architecture

### Entry Point: `main.py`
- Streamlit app with optimized startup pattern using session state caching
- Three-tier storage: session state → local JSON files → global config files
- Built-in micromamba executable for environment management
- Dual mode: Simple (single tool) vs Advanced (full toolkit)

### Key Workflows

**Primary: Add Deployment (`pages/analysis_advanced.py`)**
4-step wizard for processing camera trap deployments:
1. Folder Selection: Choose deployment folder
2. Deployment Metadata: Project, location, capture dates
3. Model Selection: Detection + classification AI models
4. Species Selection: Target species for analysis area

**Additional Tools**:
- `human_verification.py`: Manual review of AI predictions
- `remove_duplicates.py`: Eliminate redundant images
- `explore_results.py`: Data visualization and analysis
- `post_processing.py`: Result refinement tools
- `camera_management.py`: Metadata and deployment tracking
- `settings.py`: Application configuration

### AI Model System

**Model Types**:
- Detection: Animal detection (MegaDetector variants in `models/det/`)
- Classification: Species identification (regional models in `models/cls/`)
- Support: PyTorch, TensorFlow, SpeciesNet

**Model Organization**:
```
models/cls/[MODEL-ID]/
├── variables.json      # Model configuration
├── taxonomy.csv        # Species taxonomies
└── [model weights]     # Actual model files
```

**Classification Inference (`classification/`)**:
- Model-specific scripts in `model_types/[model-name]/classify_detections.py`
- Standardized interface: crop animals → classify → filter results
- GPU/CPU auto-detection, batch processing

### Configuration & Data

**Storage Hierarchy**:
- Session state: Temporary runtime cache
- Local config: `config/*.json` for tool settings
- Global map: `~/.config/AddaxAI/map.json` for projects/deployments

**Project Structure**:
- Projects → Locations → Deployments → Results
- Each deployment links to AI analysis outputs

## Environment Management

**Built-in Micromamba System**:
- MacOS: `./bin/macos/micromamba`
- Windows: `./bin/macos/micromamba.exe` (note: still macos path)
- Environment files: `envs/ymls/[env-name]/macos/environment.yml`
- Environment names always start with "env-"

**Key Environments**:
- `env-addaxai-base`: Main Streamlit application, MegaDetector, and SpeciesNet
- `env-pytorch`: PyTorch-based models
- `env-tensorflow-v1/v2`: TensorFlow models

**Common Commands**:
```bash
# Create environment
./bin/macos/micromamba env create -f envs/ymls/addaxai-base/macos/environment.yml --prefix ./envs/env-addaxai-base -y

# Install SpeciesNet (requires special flag on macOS)
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install --use-pep517 speciesnet==5.0.1

# Run in environment
./bin/macos/micromamba run -p ./envs/env-addaxai-base streamlit run main.py
./bin/macos/micromamba run -p ./envs/env-addaxai-base python -m py_compile utils/analysis_utils.py
```

## Key Files & Components

**Core Logic**:
- `utils/analysis_utils.py`: Main processing pipeline and AI model integration
- `utils/common.py`: Shared utilities, caching, configuration loading
- `utils/config.py`: Configuration management and validation
- `components/`: Reusable UI components (stepper, progress bars, etc.)

**Data Processing Flow**:
1. Load images from deployment folder
2. Run detection model → locate animals
3. Crop animal regions → classify species
4. Apply filtering/confidence thresholds
5. Generate standardized JSON results

## Performance Optimizations

**Caching Strategy**:
- Session state caching reduces file I/O by 80-90%
- Model metadata cached to eliminate repeated reads
- Conditional modal/component creation
- Write-through caching for immediate UI updates

**Technical Features**:
- Automatic model downloading and environment setup
- Version compatibility checking between models
- Queue-based batch processing
- Comprehensive error handling and logging

## Common Task Patterns

When working on this project:
1. **Model Integration**: Add new models to `models/cls/` or `models/det/` with proper `variables.json`
2. **UI Changes**: Modify pages in `pages/` or components in `components/`
3. **Processing Logic**: Update `utils/analysis_utils.py` for core AI workflows
4. **Configuration**: Modify JSON files in `config/` or global settings
5. **Environment Issues**: Check micromamba setup and environment yml files

## Testing & Quality
- Test files in `tests/` directory
- Manual testing through Streamlit interface
- Model validation through sample deployments
- Environment isolation ensures reproducible results

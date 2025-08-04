# AddaxAI Streamlit Application

A modern, multi-page Streamlit application for camera trap image analysis using AI models. This application provides an intuitive interface for wildlife researchers to analyze camera trap data with multiple detection and classification models.

## 🚀 Features

- **Multi-page Interface**: Organized tools for different analysis workflows
- **Advanced Analysis**: Complete camera trap processing pipeline
- **Model Management**: Support for multiple detection and classification models
- **Project Management**: Organize deployments by projects and locations
- **Interactive Maps**: GPS coordinate selection and visualization
- **Species Classification**: Automated species identification with multiple regional models
- **Duplicate Detection**: Remove duplicate images from analysis
- **Human Verification**: Review and correct AI predictions
- **Results Export**: Export results in various formats
- **Multi-language Support**: Interface available in 9 languages

## 📋 Prerequisites

1. **Conda Distribution**: Install [Miniforge](https://github.com/conda-forge/miniforge) or similar
2. **AddaxAI Core**: Install the base AddaxAI system from [addaxdatascience.com](https://addaxdatascience.com/addaxai/#install)

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n env-streamlit-addaxai python=3.11 -y
conda activate env-streamlit-addaxai
```

### 2. Install Required Packages

```bash
pip install streamlit streamlit-extras streamlit-aggrid
pip install folium streamlit-folium streamlit-tree-select
pip install streamlit-antd-components st-flexible-callout-elements
pip install exifread piexif gpsphoto hachoir
pip install appdirs requests pillow
pip install streamlit-modal
```

### 3. Clone Repository

Navigate to your AddaxAI installation directory:

```bash
# macOS
cd /Applications/AddaxAI_files/AddaxAI

# Linux  
cd ~/AddaxAI_files/AddaxAI

# Clone this repository
git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
```

### 4. Launch Application

```bash
cd streamlit-AddaxAI
conda activate env-streamlit-addaxai
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

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

## 🎯 Usage

### Quick Start

1. **Launch the application** using the installation steps above
2. **Select mode**: Choose between Simple or Advanced mode in the sidebar
3. **Choose analysis tool**: Navigate to "Advanced Analysis" or "Quick Analysis"
4. **Select data**: Browse and select your camera trap image folder
5. **Configure models**: Choose detection and classification models
6. **Run analysis**: Process your images and review results

### Advanced Workflow

The advanced analysis tool provides a complete pipeline:

1. **📁 Data Selection**: Browse and select image folders
2. **📍 Project Setup**: Create or select projects and camera locations
3. **🤖 Model Configuration**: Choose AI models for detection and classification
4. **🔍 Species Filtering**: Select target species for analysis
5. **⚙️ Processing**: Run the complete analysis pipeline
6. **📊 Results**: View and export results

### Available Models

#### Detection Models
- **MegaDetector 5a/5b**: General wildlife detection

#### Classification Models
- **Sub-Saharan Drylands**: African wildlife (328+ species)
- **European Models**: DeepFaune v1.1-v1.3 for European wildlife
- **Regional Specialists**: Tasmania, New Zealand, Iran, Namibia, Peru, USA Southwest
- **And many more**: See the model selection interface for full list

## ⚙️ Configuration

### Application Settings
- **Language**: Choose from 9 supported languages
- **Mode**: Switch between Simple and Advanced interfaces
- **Theme**: Custom green color scheme optimized for wildlife research

### Project Management
- **Projects**: Organize your camera trap studies
- **Locations**: Define camera positions with GPS coordinates
- **Deployments**: Track individual camera deployment periods

## 🔧 Development

This application follows modern Streamlit best practices:

- **Component-based architecture**: Reusable UI components
- **Clean separation of concerns**: Pages, components, utils structure
- **Persistent configuration**: Settings survive app restarts
- **Session state management**: Optimized for performance
- **Multi-language support**: Internationalization ready

See `DEVELOPERS.md` for detailed development guidelines and architecture documentation.

## 📖 Documentation

- **[DEVELOPERS.md](DEVELOPERS.md)**: Development guidelines and architecture
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the coding standards in `DEVELOPERS.md`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is part of the AddaxAI ecosystem. Please refer to the main AddaxAI license terms.

## 🆘 Support

- **Documentation**: Check `DEVELOPERS.md` and `PROJECT_STRUCTURE.md`
- **Issues**: Report bugs and request features via GitHub Issues
- **Website**: Visit [addaxdatascience.com](https://addaxdatascience.com) for more information

## 🏆 Acknowledgments

- **MegaDetector**: Microsoft AI for Conservation team
- **Model Contributors**: Various research institutions and wildlife organizations
- **Streamlit Community**: For the excellent framework and components
- **Camera Trap Researchers**: For feedback and testing

---

**Note**: This Streamlit application is designed to eventually replace the legacy tkinter version, providing a more modern, web-based interface for camera trap analysis workflows.
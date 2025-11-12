
# UniRare Framework 🎯

**UniRare: Framework Reconciling Bottom-Up and Top-Down Attention in Visual Saliency**

A comprehensive framework that combines state-of-the-art saliency detection models with rarity enhancement techniques to improve visual attention prediction. UniRare integrates multiple deep learning models and provides a unified 
interface for enhanced saliency detection.

- **Paper Link**: [To be provided - Current Overleaf](https://www.overleaf.com/2811315353kzttqhrnygxh#b7a9a8)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/numediart/UniRare)

## 🔍 Overview

UniRare is a novel framework that enhances traditional saliency detection by incorporating rarity information. The framework supports multiple state-of-the-art saliency detection models and provides various fusion strategies to combine saliency maps with rarity information.

**Key Innovations:**
- **Rarity Network**: A specialized network that computes rarity maps from intermediate feature representations
- **Multiple Fusion Strategies**: Addition, multiplication, subtraction, and Itti-based fusion methods
- **Unified Interface**: Support for multiple saliency detection architectures through a common API
- **Real-time Processing**: Optimized for both research and practical applications

## ✨ Features

- 🧠 **Multiple Saliency Models**: Integration of Unisal, TempSal, and TranSalNet architectures
- 🎯 **Rarity Enhancement**: Novel rarity network for attention refinement
- 🔄 **Fusion Strategies**: Multiple methods to combine saliency and rarity information
- 🚀 **GPU Acceleration**: CUDA, MPS, and CPU support with automatic device selection

## 🤖 Supported Models

### Saliency Detection Models

1. **Unisal** - Unified Saliency Detection Model
   - Architecture: CNN + RNN hybrid
   - Strengths: Temporal consistency, real-time performance
   - Paper: [Link to be added]

2. **TempSal** - Temporal Saliency Network
   - Architecture: Multi-level temporal feature extraction
   - Strengths: Video saliency, temporal dynamics
   - Paper: [Link to be added]

3. **TranSalNet (Dense)** - Transformer-based Saliency Network with DenseNet
   - Architecture: Vision Transformer + DenseNet backbone
   - Strengths: Global attention, feature reuse
   - Paper: [Link to be added]

4. **TranSalNet (Res)** - Transformer-based Saliency Network with ResNet
   - Architecture: Vision Transformer + ResNet backbone
   - Strengths: Deep feature extraction, skip connections
   - Paper: [Link to be added]

### Rarity Enhancement

- **RarityNetwork**: Computes rarity maps from intermediate feature layers
- **Fusion Methods**:
  - **Addition**: `S_final = S_sal + S_rarity`
  - **Multiplication**: `S_final = S_sal × S_rarity`
  - **Subtraction**: `S_final = S_sal - S_rarity`
  - **Itti Fusion**: `S_final = (S_sal + S_rarity) / 2`

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration, optional)

### Quick Install

1. **Clone the repository:**
```bash
git clone https://github.com/numediart/UniRare.git
cd UniRare
```

2. **Create a virtual environment:**
```bash
python -m venv unirare_env python=3.11
source unirare_env/bin/activate  # On Windows: unirare_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

The framework requires the following main packages:
- **PyTorch** 2.5.1+ (with CUDA support if available)
- **OpenCV** 4.10.0+ for image processing
- **NumPy** 2.2.1+ for numerical computations
- **Matplotlib** 3.10.0+ for visualization
- **scikit-image** 0.25.0+ for image utilities

See `requirements.txt` for the complete list of dependencies.

## 🚀 Usage

### Basic Usage

Process images with all available models:

```bash
python run_images.py --directory ./images/
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--directory` | str | `./images/` | Input directory containing images |
| `--layers` | str | `3,4,5` | Layer indices for rarity computation |
| `--threshold` | float | `None` | Threshold for rarity network |



## 📥 Model Downloads

### Pre-trained Weights

Download the pre-trained model weights and place them in the appropriate directories:

#### 1. Unisal Model
- **File**: `weights_best.pth`
- **Location**: `src/model/Unisal/weights/`
- **Download**: [Unisal Weights - To be provided]
- **Size**: ~45MB
- **Description**: Unified saliency detection model with RNN temporal processing

#### 2. TranSalNet Dense Model
- **File**: `TranSalNet_Dense.pth`
- **Location**: `src/model/TranSalNet/weights/`
- **Download**: [TranSalNet Dense Weights - To be provided]
- **Size**: ~120MB
- **Description**: Transformer-based saliency with DenseNet backbone

#### 3. TranSalNet ResNet Model
- **File**: `TranSalNet_Res.pth`
- **Location**: `src/model/TranSalNet/weights/`
- **Download**: [TranSalNet ResNet Weights - To be provided]
- **Size**: ~95MB
- **Description**: Transformer-based saliency with ResNet backbone

#### 4. TempSal Model
- **File**: `multilevel_tempsal.pt`
- **Location**: `src/model/TempSal/weights/`
- **Download**: [TempSal Weights - To be provided]
- **Size**: ~180MB
- **Description**: Multi-level temporal saliency detection model

### Quick Download Script

```python
cd script
python downloader_weights.py
```

## 📖 Citation

If you use UniRare in your research, please cite our paper:

```bibtex
@article{unirare2024,
  title={UniRare: Framework Reconciling Bottom-Up and Top-Down Attention in Visual Saliency},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  url={[Paper URL]}
}
```

### Related Publications

- **Paper Link**: [To be provided - Current Overleaf](https://www.overleaf.com/2811315353kzttqhrnygxh#b7a9a8)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


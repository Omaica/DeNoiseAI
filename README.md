# DeNoiseAI

A deep learning-based audio denoising system using **DCCRN** (Deep Complex Convolution Recurrent Network) for speech enhancement.

## Features
- Complex-valued neural network for phase-aware speech enhancement
- SI-SNR + Multi-resolution STFT loss for high-quality denoising
- Support for various SNR levels (-5dB to 20dB)
- Dynamic noise mixing for data augmentation
- Evaluation with PESQ, STOI, SI-SNR metrics

## Project Structure
```
DeNoiseAI/
├── configs/
│   └── dccrn_config.yaml        # Hyperparameters
├── src/
│   ├── models/
│   │   ├── complex_layers.py    # Complex-valued layers
│   │   └── dccrn.py             # DCCRN model
│   ├── datasets/
│   │   └── audio_dataset.py     # Dataset & DataLoader
│   ├── losses/
│   │   └── loss.py              # SI-SNR, STFT loss
│   ├── utils/
│   │   ├── audio_utils.py       # Audio processing utilities
│   │   └── metrics.py           # PESQ, STOI evaluation
│   └── train.py                 # Training loop
├── scripts/
│   ├── download_data.py         # Download datasets
│   ├── prepare_data.py          # Preprocess audio
│   └── mix_audio.py             # Mix clean + noise
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
```bash
python scripts/download_data.py --output_dir data/raw
```

### 3. Prepare data
```bash
python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed
python scripts/mix_audio.py --clean_dir data/processed/clean --noise_dir data/processed/noise --output_dir data/train
```

### 4. Train
```bash
python src/train.py --config configs/dccrn_config.yaml
```

### 5. Inference
```bash
python scripts/inference.py --model_path checkpoints/best_model.pth --input noisy.wav --output clean.wav
```

## Requirements
- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU with 6GB+ VRAM (recommended)

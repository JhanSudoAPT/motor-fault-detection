# Motor Fault Detection System

A machine learning system for detecting motor faults using vibration signal analysis with time and frequency domain features.

## Features

- Signal processing with 24 time and frequency domain features
- Noise addition at multiple SNR levels
- Neural network model training with n → 2n → n → m architecture
- Comprehensive feature analysis (Random Forest + ANOVA)
- Performance evaluation across noise conditions

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your .mat files in the `data/input/` directory

## Usage

1. **Process data**: `python data_processor.py`
   - Extracts features from clean signals
   - Adds noise at different SNR levels
   - Creates training datasets

2. **Train model**: `python model_trainer.py`
   - Trains neural network model
   - Evaluates performance across noise levels
   - Saves best model and performance metrics

3. **Analyze features**: `python feature_analyzer.py`
   - Performs feature importance analysis
   - Creates visualizations and reports
   - Identifies most discriminative features


## Configuration

Edit `config.yaml` to customize:
- Signal processing parameters
- Noise levels (SNR)
- Model architecture and training
- Feature selection

## Results

The system generates:
- Processed datasets with 24 features
- Trained models (Keras + TFLite)
- Performance metrics across noise levels
- Feature importance rankings
- Visualization plots



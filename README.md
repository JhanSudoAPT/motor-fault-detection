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

The .zip is the library to import in Arduino IDE; after importing, open the library, then go to examples and static buffer; the following input vector is placed into the network for inference testing:

0.08955888519527402,139.4578910111803,0.020632957356597113,0.09749101564431141,0.01538461536422601,0.1424357820097787,0.14093343488135823,0.019862233067458043,0.9104411134794165,0.038078514760063985,-0.40391761307511576,1.2432533737436242,557.5953957384407,0.45224761434639804,-0.18753220003744708,9.707071159030331e-05,2.5448304156512505,3.4844127274080985,0.19491333627434648,3.9474517173294554,574.7705012664185,3.175098335299972,0.09755698016342909,0.00031489454116133723


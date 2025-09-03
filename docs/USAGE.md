# NetConfigMaster Usage Guide

## Overview
NetConfigMaster is a machine learning project that translates natural language descriptions into Nmstate YAML configurations for network management.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Training a Model
```bash
python src/pretrain.py  # Pretrain the model
python src/train.py     # Train on your dataset
```

#### 2. Evaluating a Model
```bash
python src/model_evaluation.py
```

#### 3. Running the Web UI
```bash
python src/ui.py
# or
flask --app src/ui.py run
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

## Data Management

### Dataset Structure
- `data/training/training.yaml`: Training dataset configuration
- `data/evaluation/evaluation.yaml`: Evaluation dataset configuration
- `data/raw/`: Place raw, unprocessed datasets here
- `data/processed/`: Cleaned and preprocessed datasets
- `data/evaluation_results/`: Model evaluation results

### Adding Your Own Data
1. Place raw datasets in `data/raw/`
2. Update configuration files in `data/training/` and `data/evaluation/`
3. Run preprocessing scripts as needed

## Model Workflows

### 1. Pretraining Process
- Configure model settings (codegen-350M compatible)
- Load and tokenize YAML dataset (substratusai/the-stack-yaml-k8s)
- Pretrain until convergence
- Save pretrained model

### 2. Training Process  
- Prepare training dataset with natural language + YAML pairs
- Load and preprocess data
- Initialize with pretrained weights
- Train until convergence
- Save trained model and tokenizer

### 3. Evaluation Process
- Prepare evaluation dataset
- Load and preprocess data
- Initialize pretrained model
- Define metrics (nmstate_correct, exact_match, yaml_correct, levenshtein_distance)
- Generate predictions and calculate scores

## Metrics

### Available Metrics
- **Nmstate Correct**: Validates Nmstate schema compliance
- **Exact Match (EM)**: Percentage of exact predictions
- **YAML Correct**: Validates YAML schema compliance  
- **Levenshtein Distance**: Edit distance between prediction and ground truth

### Custom Metrics
Add custom evaluation metrics in the `metrics/` directory. Each metric should:
1. Have its own subdirectory
2. Include an `app.py` file
3. Include the metric implementation file
4. Follow the existing pattern

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running from the project root
2. **CUDA Issues**: Check PyTorch CUDA compatibility
3. **Memory Issues**: Reduce batch size in training configurations

### Getting Help
- Check the main README.md for project overview
- Review individual directory README files
- Examine the evaluation results in `data/evaluation_results/`

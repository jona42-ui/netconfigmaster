# NetConfigMaster Architecture

## System Overview

NetConfigMaster is a machine learning pipeline that uses transformer models to translate natural language network configuration requests into Nmstate YAML specifications.

## Architecture Diagram

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Natural Language  │    │   Transformer    │    │  Nmstate YAML   │
│   Input             │───▶│   Model          │───▶│  Output         │
│   "Configure eth0   │    │   (CodeGen-350M  │    │  ---            │
│    with IP..."      │    │    based)        │    │  interfaces:... │
└─────────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. Data Pipeline (`data/`)
- **Raw Data**: Unprocessed datasets
- **Processed Data**: Cleaned and tokenized datasets
- **Configuration**: Training and evaluation YAML files
- **Results**: Model evaluation outputs

### 2. Source Code (`src/`)
- **pretrain.py**: Pretraining pipeline using YAML datasets
- **train.py**: Fine-tuning on network configuration tasks
- **model_evaluation.py**: Evaluation framework with custom metrics
- **ui.py**: Flask web interface
- **utils.py**: Shared utility functions

### 3. Metrics System (`metrics/`)
Custom evaluation metrics for network configuration tasks:
- **Nmstate Validation**: Ensures output conforms to Nmstate schema
- **YAML Validation**: Ensures syntactically correct YAML
- **Levenshtein Distance**: Character-level edit distance
- **Exact Match**: Binary exact string matching

### 4. Testing (`tests/`)
Unit and integration tests for reliability

## Data Flow

### Training Flow
```
Raw Dataset → Preprocessing → Tokenization → Model Training → Saved Model
     ↓              ↓             ↓              ↓              ↓
data/raw/    data/processed/  src/utils.py   src/train.py  models/
```

### Inference Flow  
```
Natural Language → Tokenization → Model Inference → YAML Generation → Validation
       ↓               ↓              ↓                ↓             ↓
   User Input      src/utils.py   Trained Model    Generated      metrics/
                                                    Output
```

### Evaluation Flow
```
Test Dataset → Model Inference → Predictions → Metric Calculation → Results
     ↓              ↓              ↓              ↓                ↓
data/evaluation/ Trained Model  Generated    metrics/         data/evaluation_results/
                                Outputs
```

## Model Architecture

### Base Model
- **Foundation**: CodeGen-350M transformer architecture
- **Pretraining**: YAML syntax learning on substratusai/the-stack-yaml-k8s
- **Fine-tuning**: Network configuration task-specific training

### Input/Output Format
- **Input**: Natural language description (tokenized)
- **Output**: Nmstate YAML configuration (generated tokens → detokenized)

## Key Design Decisions

### 1. Two-Stage Training
- **Pretraining**: General YAML understanding
- **Fine-tuning**: Domain-specific network configuration

### 2. Custom Metrics
- Standard metrics (BLEU, ROUGE) insufficient for configuration validation
- Custom metrics ensure syntactic and semantic correctness

### 3. Modular Architecture
- Separate metrics allow easy extension
- Clear separation between training, evaluation, and inference

## Scalability Considerations

### Performance
- Model size: 350M parameters (balance of capability vs. resource usage)
- Inference: Single GPU sufficient for real-time generation
- Batch processing: Configurable batch sizes for training

### Extensibility
- New metrics: Add to `metrics/` directory
- New datasets: Update `data/` configurations
- Model variants: Modify `src/train.py` and `src/pretrain.py`

## Technology Stack

- **ML Framework**: PyTorch + Transformers (Hugging Face)
- **Web Interface**: Flask
- **Data Processing**: PyYAML, Datasets library
- **Evaluation**: Custom Python implementations
- **Code Quality**: Pylint (configuration in `pylintrc`)

## Future Enhancements

1. **Model Improvements**
   - Larger model variants
   - Multi-task learning
   - Reinforcement learning from human feedback

2. **System Improvements**
   - REST API endpoints
   - Model versioning
   - Distributed training support

3. **Evaluation Improvements**
   - Integration tests with actual Nmstate
   - Network topology validation
   - Performance benchmarking

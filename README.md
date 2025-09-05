
# NetConfigMaster

NetConfigMaster is a comprehensive multi-task network configuration automation system that leverages advanced natural language processing and machine learning to transform network configuration management across multiple vendors and use cases.

## 🎯 Overview

NetConfigMaster represents a significant advancement in network automation, implementing a sophisticated multi-task learning approach inspired by the PreConfig methodology. The system supports three core tasks: **Configuration Generation** (natural language → configuration), **Configuration Analysis** (configuration → natural language), and **Configuration Translation** (vendor A → vendor B).

### 🚀 Phase 1 Implementation Complete

Our **comprehensive Phase 1 implementation** transforms NetConfigMaster from a single-task system into a full multi-task network configuration automation platform:

- **Multi-Task Architecture**: Shared encoder with task-specific decoders supporting all three core tasks
- **Multi-Vendor Support**: Native support for Cisco IOS, Juniper JUNOS, and Nmstate configurations
- **Advanced Evaluation**: Comprehensive metrics including BLEU, ROUGE, Exact Match, syntax validation, and semantic correctness
- **Intelligent Sampling**: Advanced sampling strategies for handling dataset imbalances across vendors and tasks
- **Professional Monitoring**: Complete training monitoring with metrics tracking, checkpointing, and visualization

## ✨ Key Features

### Multi-Task Capabilities
- **Configuration Generation**: Convert natural language descriptions to network configurations
- **Configuration Analysis**: Extract natural language descriptions from network configurations  
- **Configuration Translation**: Convert configurations between different vendor formats

### Advanced ML Pipeline
- **CodeT5-based Architecture**: Leverages Salesforce's CodeT5 model optimized for network configurations
- **Balanced Multi-Task Training**: Intelligent sampling and task-specific loss weighting
- **Curriculum Learning**: Progressive training from basic to advanced configurations
- **Performance Monitoring**: Real-time training metrics and convergence analysis

### Vendor Ecosystem
- **Cisco IOS/IOS-XE**: Complete command syntax support with validation
- **Juniper JUNOS**: Full configuration hierarchy parsing and generation
- **Nmstate**: Linux network state configuration with YAML processing
- **Cross-Vendor Translation**: Seamless configuration migration between vendors

### Professional-Grade Evaluation
- **Syntax Validation**: Vendor-specific configuration syntax checking
- **Semantic Correctness**: Network logic and consistency validation
- **Comprehensive Metrics**: BLEU, ROUGE, Exact Match scoring
- **Performance Analytics**: Task and vendor-specific accuracy tracking

## 🏗️ Architecture

NetConfigMaster employs a sophisticated multi-task learning architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    NetConfigMaster Architecture             │
├─────────────────────────────────────────────────────────────┤
│  Input Processing                                           │
│  ├── Natural Language Parser                               │
│  ├── Configuration Parser (Multi-Vendor)                   │
│  └── Task Type Detection                                    │
│                                                             │
│  Multi-Task Model (CodeT5-based)                          │
│  ├── Shared Encoder                                        │
│  ├── Task-Specific Decoders                               │
│  │   ├── Generation Decoder                               │
│  │   ├── Analysis Decoder                                 │
│  │   └── Translation Decoder                              │
│  └── Vendor-Aware Tokenization                            │
│                                                             │
│  Advanced Evaluation System                                 │
│  ├── BLEU/ROUGE/Exact Match Metrics                       │
│  ├── Syntax Validation Engine                             │
│  ├── Semantic Correctness Checker                         │
│  └── Performance Analytics                                 │
│                                                             │
│  Training & Monitoring                                      │
│  ├── Advanced Sampling Strategies                          │
│  ├── Curriculum Learning                                   │
│  ├── Training Monitoring & Checkpointing                   │
│  └── Visualization & Reporting                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Multi-Task Training Framework** (`src/multitask_train.py`): Comprehensive training system with balanced sampling
- **Evaluation Metrics System** (`src/evaluation_metrics.py`): Advanced multi-dimensional evaluation
- **Data Processing Pipeline** (`src/data_processing.py`): Unified data handling with vendor detection
- **Vendor Support Module** (`src/vendor_support.py`): Multi-vendor parsing and generation
- **Advanced Sampling** (`src/advanced_sampling.py`): Sophisticated dataset balancing
- **Training Monitor** (`src/training_monitor.py`): Professional training tracking and visualization

---

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Poetry (recommended) or pip

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/jona42-ui/netconfigmaster.git
cd netconfigmaster
```

2. **Install with Poetry** (recommended):
```bash
poetry install
poetry shell
```

3. **Or install with pip**:
```bash
pip install -r model\ training/requirements.txt
```

4. **Configure the system**:
```bash
cp configs/multitask_config.json configs/local_config.json
# Edit local_config.json with your settings
```

### Quick Start - Multi-Task Training

```bash
# Test Phase 1 implementation
./test_phase1.sh

# Start multi-task training
python src/multitask_train.py --config configs/multitask_config.json

# Monitor training progress
tail -f outputs/training.log
```

## 📖 Usage Examples

### 1. Configuration Generation
```python
from src.multitask_train import MultiTaskModel

# Load trained model
model = MultiTaskModel.load_pretrained("outputs/checkpoints/best_model.pt")

# Generate configuration
result = model.generate(
    task="generation",
    input_text="Configure GigabitEthernet0/1 with IP 192.168.1.1/24 and enable the port",
    vendor="cisco"
)
print(result.generated_config)
```

### 2. Configuration Analysis
```python
# Analyze existing configuration
config = """
interface GigabitEthernet0/1
 description LAN Interface
 ip address 192.168.1.1 255.255.255.0
 no shutdown
"""

result = model.analyze(
    task="analysis", 
    config=config,
    vendor="cisco"
)
print(result.description)  # "Configure GigabitEthernet0/1 with IP 192.168.1.1/24..."
```

### 3. Configuration Translation
```python
# Translate Cisco to Juniper
cisco_config = "interface GigabitEthernet0/1
 ip address 192.168.1.1 255.255.255.0
 no shutdown"

result = model.translate(
    task="translation",
    source_config=cisco_config,
    source_vendor="cisco",
    target_vendor="juniper"
)
print(result.translated_config)
```

## 🔧 Configuration

The system supports extensive configuration through `configs/multitask_config.json`:

```json
{
  "model": {
    "name": "Salesforce/codet5-base",
    "max_length": 512,
    "num_beams": 4
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "task_weights": {
      "generation": 1.0,
      "analysis": 1.1,
      "translation": 1.2
    }
  },
  "sampling": {
    "strategy": "balanced",
    "vendor_weights": {
      "cisco": 1.0,
      "juniper": 1.2,
      "nmstate": 0.8
    }
  }
}
```


## Project Structure

```
netconfigmaster/
├── data/                    # Datasets and configurations
│   ├── raw/                # Unprocessed datasets
│   ├── processed/          # Cleaned datasets
│   ├── training/           # Training configurations
│   ├── evaluation/         # Evaluation configurations
│   └── evaluation_results/ # Model evaluation results
├── src/                     # Source code
│   ├── pretrain.py         # Pretraining pipeline
│   ├── train.py            # Training pipeline
│   ├── model_evaluation.py # Evaluation framework
│   ├── ui.py               # Web interface
│   └── utils.py            # Utility functions
├── metrics/                 # Custom evaluation metrics
│   ├── nmstate_correct/    # Nmstate schema validation
│   ├── yaml_correct/       # YAML syntax validation
│   └── levenshtein_distance/ # Edit distance metric
├── docs/                    # Extended documentation
├── tests/                   # Unit and integration tests
├── requirements.txt         # Python dependencies
└── pylintrc                # Code quality configuration
```

## Workflows

### Pretraining Process Overview

* Model Configuration: Define the model configuration, using the same settings as for
  codegen-350M.
* Load and Tokenize Dataset: Load and tokenize the YAML dataset
  substratusai/the-stack-yaml-k8s.
* Pretraining: Pretrain the model to enhance its understanding of YAML syntax and
  semantics until the training loss converges.
* Save Model: Save the pretrained model.

### Training Process Overview

* Prepare Training Dataset: Prepare the dataset for training, ensuring each training
  sample consists of a natural language description and a YAML state.
* Load, Preprocess, and Tokenize Dataset: Load the dataset, preprocess the data, and
  tokenize it for training.
* Load Pretrained Model: Initialize the model using pretrained weights.
* Train the Model: Train the model using the prepared dataset until the training loss
  converges.
* Save Model and Tokenizer: Save the trained model and tokenizer for future use.

### Evaluation Process Overview

* Prepare Evaluation Dataset: Set up the dataset for evaluation, ensuring each
  sample includes a natural language description and the corresponding expected
  YAML state.
* Load, Preprocess, and Tokenize Dataset: Load the dataset and preprocess the
  data for evalution.
* Load Pretrained Model: Initialize the model with pretrained weights.
* Define Evaluation Metrics: Specify the metrics (nmstate_correct, exact_match,
  yaml_correct, levenshtein_distance) to be used for evaluation.
* Evaluate the Model: Use the model to perform inference on the natural
  language descriptions. Compare the generated YAML with the expected YAML
  based on the defined evaluation metrics to calculate the metric scores. Use
  cached data if available, or generate new predictions for evaluation.


## Quick Start

### Prerequisites
- Python 3.8+ 
- [Poetry](https://python-poetry.org/docs/#installation) (recommended)
- [Docker](https://docs.docker.com/get-docker/) (optional, for containerized development)
- [VS Code](https://code.visualstudio.com/) with Dev Containers extension (optional)

### Installation Options

#### Option 1: Using Poetry (Recommended)
```bash
# Clone the repository
git clone https://github.com/jona42-ui/netconfigmaster.git
cd netconfigmaster

# Run setup script
chmod +x scripts/setup.sh && ./scripts/setup.sh

# Or manually:
poetry install --with dev,docs
poetry shell
```

#### Option 2: Using Docker
```bash
# Development environment
docker-compose up dev

# Production web UI
docker-compose up web
```

#### Option 3: Using VS Code Dev Containers
1. Open the project in VS Code
2. Click "Reopen in Container" when prompted
3. Or use Command Palette: "Dev Containers: Reopen in Container"

### Usage

#### 1. Train a model
```bash
# Using Poetry
poetry run python src/pretrain.py  # Pretrain on YAML data
poetry run python src/train.py     # Fine-tune on network config data

# Using Docker
docker-compose --profile training up train
```

#### 2. Evaluate the model
```bash
# Using Poetry
poetry run python src/model_evaluation.py

# Using Docker
docker-compose --profile evaluation up evaluate
```

#### 3. Run the web interface
```bash
# Using Poetry
poetry run python src/ui.py

# Using Docker
docker-compose up web
```

#### 4. Development Tools
```bash
# Format and lint code
./scripts/lint.sh

# Run tests
./scripts/test.sh

# Build Docker images
./scripts/build.sh
```

Open your browser and navigate to [http://localhost:5000](http://localhost:5000) to use the web interface.

## Documentation

- 📖 [Usage Guide](docs/USAGE.md) - Detailed usage instructions
- 🏗️ [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- 📊 [Evaluation Results](data/evaluation_results/) - Latest model performance

## Evaluation Results and Metrics

The evaluation of the YAMLsmith model is based on several key metrics which are crucial for assessing the model's performance in translating the natural language into Nmstate states.

### Metrics Explained

- **Nmstate Correct**: Assesses whether the generated output correctly follows the Nmstate schema without any structural or syntactic errors.
- **EM (Exact Match)**: Measures the percentage of predictions that exactly match any one of the ground truth answers.
- **YAML Correct**: Assesses whether the generated output correctly follows a predefined YAML schema without any structural or syntactic errors.
- **Levenshtein Distance**: Quantifies the minimum number of single-character edits (insertions, deletions, or substitutions) required to change the prediction into the ground truth answer.

### Evaluation Results

Here are the results from our latest model evaluation:

| Metric                   | Score  |
|--------------------------|--------|
| Nmstate Correct Predictions | 91.18   |
| Nmstate Correct References | 94.12   |
| Exact Match (EM)          | 85.29   |
| YAML Correct              | 94.12  |
| Levenshtein Distance      | 14.12    |

### Interpretation of Results

- **Nmstate Correct Predictions**: A score of 91.18 indicates that the model's predictions matched the correct Nmstate schema 91.18% of the time.
- **Nmstate Correct References**: A score of 94.12 indicates that the model's references matched the correct Nmstate schema 94.12% of the time.
- **Exact Match (EM)**: A score of 85.29 indicates that the model's predictions exactly matched the ground truth answers 85.29% of the time.
- **YAML Correct**: A score of 94.12 signifies that all generated outputs correctly adhere to the required YAML schema.
- **Levenshtein Distance**: The average minimal edit distance of 14.12 indicates that, on average, a relatively higher number of edits are required to align the model's predictions with the ground truth.

### Reproducing the Evaluation Results

To reproduce the evaluation results, please follow these steps:

1. Pretraining Process Overview:
   - Model Configuration: Define the model configuration, using the same settings as for codegen-350M.
   - Load and Tokenize Dataset: Load and tokenize the YAML dataset substratusai/the-stack-yaml-k8s.
   - Pretraining: Pretrain the model to enhance its understanding of YAML syntax and semantics until the training loss converges.
   - Save Model: Save the pretrained model.

2. Training Process Overview:
   - Prepare Training Dataset: Prepare the dataset for training, ensuring each training sample consists of a natural language description and a YAML state.
   - Load, Preprocess, and Tokenize Dataset: Load the dataset, preprocess the data, and tokenize it for training.
   - Load Pretrained Model: Initialize the model using pretrained weights.
   - Train the Model: Train the model using the prepared dataset until the training loss converges.
   - Save Model and Tokenizer: Save the trained model and tokenizer for future use.

3. Evaluation Process Overview:
   - Prepare Evaluation Dataset: Set up the dataset for evaluation, ensuring each sample includes a natural language description and the corresponding expected YAML state.
   - Load, Preprocess, and Tokenize Dataset: Load the dataset and preprocess the data for evalution.
   - Load Pretrained Model: Initialize the model with pretrained weights.
   - Define Evaluation Metrics: Specify the metrics (nmstate_correct, exact_match, yaml_correct, levenshtein_distance) to be used for evaluation.
   - Evaluate the Model: Use the model to perform inference on the natural language descriptions. Compare the generated YAML with the expected YAML based on the defined evaluation metrics to calculate the metric scores. Use cached data if available, or generate new predictions for evaluation.

## License and Credits

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Credits

- Developed by [thembo jonathan](https://github.com/jona42-ui)
- Nmstate library: [Nmstate ](https://github.com/nmstate/nmstate)
- NetVisor tool: [NetVisor](https://github.com/ffmancera/NetVisor)



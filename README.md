````markdown
---

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

# 🚀 Contributing

We welcome contributions from the community! Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guidelines](CONTRIBUTING.md) before opening issues or pull requests.

- To report bugs or request features, use the GitHub Issues tab.
- For code contributions, fork the repo and submit a pull request.
- See `.github/ISSUE_TEMPLATE.md` and `.github/PULL_REQUEST_TEMPLATE.md` for templates.

---

# Translating Natural Language into Nmstate States

This repository contains a project that utilizes the `transformers` library to train a
model for generating Nmstate YAML states.

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
````


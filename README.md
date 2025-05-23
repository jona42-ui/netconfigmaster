
# Translating Natural Language into Nmstate States

This repository contains a project that utilizes the `transformers` library to train a
model for generating Nmstate YAML states.

## Project Structure

* `./dataset/`: Contains the dataset for traning and evaluation.
* `./evaluation_results/`: Contains the evaluation result of the most recent
    YAMLsmith model.
* `./metrics/`: Contains the customized evaluation metrics (`nmstate_correct`,
  `yaml_correct` , `levenshtein_distance` ) for evaluating the model.
* `./src/`: Contains the pipeline for pretraing, training, and evaluating the model,
    as well as the UI for hosting the model.

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


## Installation

Before running the script, ensure you have Python 3.8+ installed. You can then install
the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```bash
python app.py

or

flask --app app.py run
```

2. Open your web browser and go to [http://localhost:5000](http://127.0.0.1:5000).

3. Enter natural language commands related to network configuration in the input field and click "Submit" to see the generated network state.

4. Click on "Show Image" to visualize the network topology image.

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


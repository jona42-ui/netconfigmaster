#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Multi-Task Learning Framework for Network Configuration Automation

This module implements a comprehensive multi-task learning system supporting:
1. Configuration Generation (Natural Language → Network Configuration)
2. Configuration Analysis (Network Configuration → Natural Language)
3. Configuration Translation (Vendor A Config → Vendor B Config)

Key Features:
- Unified transformer architecture with task-specific heads
- Balanced sampling for imbalanced multi-task datasets
- Comprehensive evaluation metrics (BLEU, ROUGE, Exact Match)
- Support for multiple network vendors (Cisco, Juniper, etc.)
- Advanced training strategies and monitoring

Architecture:
    Shared Encoder → Task-Specific Decoders → Multi-Task Loss

Usage:
    python -m src.multitask_train --config configs/multitask_config.yaml
"""

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .evaluation_metrics import MultiTaskEvaluator
from .utils import load_yaml_dataset, setup_logging


# Task definitions and constants
TASK_TYPES = {
    'generation': 'nl_to_config',
    'analysis': 'config_to_nl', 
    'translation': 'config_to_config'
}

VENDOR_TAGS = {
    'cisco': '<cisco>',
    'juniper': '<juniper>',
    'nmstate': '<nmstate>',
    'natural_language': '<nl>'
}

TASK_PREFIXES = {
    'generation': 'Generate configuration: ',
    'analysis': 'Analyze configuration: ',
    'translation': 'Translate configuration: '
}


@dataclass
class MultiTaskConfig:
    """Configuration class for multi-task training"""
    
    # Model configuration
    model_name_or_path: str = "Salesforce/codet5-base"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    use_auth_token: Optional[str] = None
    
    # Data configuration
    train_files: Dict[str, str] = field(default_factory=dict)
    validation_files: Dict[str, str] = field(default_factory=dict)
    test_files: Dict[str, str] = field(default_factory=dict)
    max_source_length: int = 512
    max_target_length: int = 512
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    
    # Training configuration
    output_dir: str = "./results"
    overwrite_output_dir: bool = True
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.1
    logging_dir: str = "./logs"
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    
    # Multi-task specific configuration
    task_sampling_alpha: float = 0.5  # For balanced sampling
    task_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'generation': 1.0,
        'analysis': 1.0, 
        'translation': 1.0
    })
    curriculum_learning: bool = False
    curriculum_schedule: str = "linear"  # linear, exponential, step
    
    # Evaluation configuration
    predict_with_generate: bool = True
    generation_max_length: int = 512
    generation_num_beams: int = 4
    metric_for_best_model: str = "eval_combined_score"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True


class MultiTaskDataProcessor:
    """Handles data loading and preprocessing for multi-task learning"""
    
    def __init__(self, config: MultiTaskConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
    def load_datasets(self) -> Dict[str, DatasetDict]:
        """Load datasets for all tasks"""
        datasets = {}
        
        for task_name, task_type in TASK_TYPES.items():
            if task_name not in self.config.train_files:
                self.logger.warning(f"No training file specified for task {task_name}")
                continue
                
            task_datasets = {}
            
            # Load training data
            if self.config.train_files.get(task_name):
                train_data = load_yaml_dataset(self.config.train_files[task_name])
                task_datasets['train'] = Dataset.from_dict(
                    self._process_task_data(train_data, task_name, task_type)
                )
                
            # Load validation data
            if self.config.validation_files.get(task_name):
                val_data = load_yaml_dataset(self.config.validation_files[task_name])
                task_datasets['validation'] = Dataset.from_dict(
                    self._process_task_data(val_data, task_name, task_type)
                )
                
            # Load test data
            if self.config.test_files.get(task_name):
                test_data = load_yaml_dataset(self.config.test_files[task_name])
                task_datasets['test'] = Dataset.from_dict(
                    self._process_task_data(test_data, task_name, task_type)
                )
                
            if task_datasets:
                datasets[task_name] = DatasetDict(task_datasets)
                self.logger.info(f"Loaded {task_name} dataset with {len(task_datasets)} splits")
        
        return datasets
    
    def _process_task_data(self, raw_data: List[Dict], task_name: str, task_type: str) -> Dict[str, List]:
        """Process raw data for specific task type"""
        processed = {
            'input_text': [],
            'target_text': [],
            'task_name': [],
            'task_type': [],
            'source_vendor': [],
            'target_vendor': []
        }
        
        for item in raw_data:
            if task_type == 'nl_to_config':
                # Configuration Generation
                input_text = f"{TASK_PREFIXES['generation']}{item['question']}"
                target_text = f"{self._get_vendor_tag(item)}{item['answer']}"
                source_vendor = 'natural_language'
                target_vendor = item.get('vendor', 'unknown')
                
            elif task_type == 'config_to_nl':
                # Configuration Analysis
                input_text = f"{TASK_PREFIXES['analysis']}{self._get_vendor_tag(item)}{item['answer']}"
                target_text = f"{VENDOR_TAGS['natural_language']}{item['question']}"
                source_vendor = item.get('vendor', 'unknown')
                target_vendor = 'natural_language'
                
            elif task_type == 'config_to_config':
                # Configuration Translation
                source_vendor = item.get('source_vendor', 'cisco')
                target_vendor = item.get('target_vendor', 'juniper')
                input_text = f"{TASK_PREFIXES['translation']}{VENDOR_TAGS[source_vendor]}{item['source_config']}"
                target_text = f"{VENDOR_TAGS[target_vendor]}{item['target_config']}"
                
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            processed['input_text'].append(input_text)
            processed['target_text'].append(target_text)
            processed['task_name'].append(task_name)
            processed['task_type'].append(task_type)
            processed['source_vendor'].append(source_vendor)
            processed['target_vendor'].append(target_vendor)
            
        return processed
    
    def _get_vendor_tag(self, item: Dict) -> str:
        """Extract appropriate vendor tag from data item"""
        vendor = item.get('vendor', 'unknown')
        return VENDOR_TAGS.get(vendor, '<unknown>')
    
    def tokenize_datasets(self, datasets: Dict[str, DatasetDict]) -> Dict[str, DatasetDict]:
        """Tokenize all datasets"""
        tokenized_datasets = {}
        
        for task_name, task_dataset in datasets.items():
            tokenized_task = task_dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=self.config.preprocessing_num_workers,
                remove_columns=task_dataset["train"].column_names,
                load_from_cache_file=not self.config.overwrite_cache,
                desc=f"Tokenizing {task_name}",
            )
            tokenized_datasets[task_name] = tokenized_task
            
        return tokenized_datasets
    
    def _tokenize_function(self, examples):
        """Tokenization function for sequence-to-sequence tasks"""
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.config.max_source_length,
            padding=False,
            truncation=True,
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target_text"],
                max_length=self.config.max_target_length,
                padding=False,
                truncation=True,
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        # Add task metadata
        model_inputs["task_name"] = examples["task_name"]
        model_inputs["task_type"] = examples["task_type"]
        model_inputs["source_vendor"] = examples["source_vendor"]
        model_inputs["target_vendor"] = examples["target_vendor"]
        
        return model_inputs


class MultiTaskSampler:
    """Implements balanced sampling strategies for multi-task learning"""
    
    def __init__(self, datasets: Dict[str, DatasetDict], config: MultiTaskConfig):
        self.datasets = datasets
        self.config = config
        self.task_sizes = {task: len(dataset['train']) for task, dataset in datasets.items()}
        self.total_size = sum(self.task_sizes.values())
        self.sampling_probs = self._calculate_sampling_probabilities()
        
    def _calculate_sampling_probabilities(self) -> Dict[str, float]:
        """Calculate balanced sampling probabilities using temperature scaling"""
        # Calculate base proportions
        base_probs = {task: size / self.total_size for task, size in self.task_sizes.items()}
        
        # Apply temperature scaling (alpha parameter)
        alpha = self.config.task_sampling_alpha
        scaled_probs = {}
        
        for task, prob in base_probs.items():
            scaled_probs[task] = prob ** alpha
            
        # Normalize
        total_scaled = sum(scaled_probs.values())
        normalized_probs = {task: prob / total_scaled for task, prob in scaled_probs.items()}
        
        logging.info(f"Task sampling probabilities: {normalized_probs}")
        return normalized_probs
    
    def create_combined_dataset(self) -> Dataset:
        """Create a combined dataset with balanced sampling"""
        combined_data = []
        
        for task_name, dataset_dict in self.datasets.items():
            train_dataset = dataset_dict['train']
            task_prob = self.sampling_probs[task_name]
            
            # Calculate number of samples for this task
            n_samples = max(1, int(task_prob * self.total_size))
            
            # Sample with replacement if needed
            if n_samples > len(train_dataset):
                indices = np.random.choice(len(train_dataset), n_samples, replace=True)
            else:
                indices = np.random.choice(len(train_dataset), n_samples, replace=False)
            
            # Add sampled data
            for idx in indices:
                item = train_dataset[idx]
                item['task_id'] = task_name
                combined_data.append(item)
        
        # Shuffle combined dataset
        random.shuffle(combined_data)
        
        # Convert to Hugging Face Dataset
        combined_dict = defaultdict(list)
        for item in combined_data:
            for key, value in item.items():
                combined_dict[key].append(value)
        
        return Dataset.from_dict(dict(combined_dict))


class MultiTaskModel(nn.Module):
    """Multi-task model with shared encoder and task-specific components"""
    
    def __init__(self, config: MultiTaskConfig, base_model_config):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForSeq2SeqLM.from_config(base_model_config)
        
        # Task-specific loss weights
        self.task_weights = torch.tensor([
            config.task_loss_weights.get(task, 1.0) 
            for task in TASK_TYPES.keys()
        ])
        
    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None, **kwargs):
        """Forward pass with task-aware loss computation"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        if labels is not None and task_name is not None:
            # Apply task-specific loss weighting
            task_indices = torch.tensor([
                list(TASK_TYPES.keys()).index(task) 
                for task in task_name
            ], device=outputs.loss.device)
            
            weights = self.task_weights[task_indices].to(outputs.loss.device)
            weighted_loss = outputs.loss * weights.mean()
            
            outputs.loss = weighted_loss
            
        return outputs


class MultiTaskTrainer(Seq2SeqTrainer):
    """Custom trainer for multi-task learning"""
    
    def __init__(self, config: MultiTaskConfig, evaluator: 'MultiTaskEvaluator', **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute multi-task loss"""
        labels = inputs.pop("labels")
        task_names = inputs.pop("task_name", None)
        
        outputs = model(**inputs, labels=labels, task_name=task_names)
        loss = outputs.loss
        
        # Log task-specific losses if available
        if hasattr(outputs, 'task_losses'):
            for task, task_loss in outputs.task_losses.items():
                self.log({f"train_loss_{task}": task_loss})
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with multi-task metrics"""
        # Run standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add custom multi-task evaluation
        if self.eval_dataset is not None:
            custom_metrics = self.evaluator.evaluate_all_tasks(
                self.model, 
                self.eval_dataset, 
                self.tokenizer
            )
            
            # Add custom metrics to results
            for metric_name, value in custom_metrics.items():
                eval_results[f"{metric_key_prefix}_{metric_name}"] = value
        
        return eval_results


def setup_model_and_tokenizer(config: MultiTaskConfig) -> Tuple[MultiTaskModel, PreTrainedTokenizer]:
    """Setup model and tokenizer with multi-task capabilities"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=config.use_fast_tokenizer,
        revision=config.model_revision,
        use_auth_token=config.use_auth_token,
    )
    
    # Add special tokens for vendors and tasks
    special_tokens = list(VENDOR_TAGS.values())
    tokenizer.add_tokens(special_tokens)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model configuration
    base_config = AutoConfig.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        revision=config.model_revision,
        use_auth_token=config.use_auth_token,
    )
    
    # Adjust vocab size for new tokens
    base_config.vocab_size = len(tokenizer)
    
    # Create multi-task model
    model = MultiTaskModel(config, base_config)
    
    # Resize token embeddings
    model.base_model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Multi-Task Network Configuration Training")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = MultiTaskConfig(**config_dict)
    
    # Setup logging
    setup_logging(config.logging_dir)
    logger = logging.getLogger(__name__)
    
    # Set seeds for reproducibility
    set_seed(config.seed)
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup data processor
    logger.info("Loading and processing datasets...")
    data_processor = MultiTaskDataProcessor(config, tokenizer)
    raw_datasets = data_processor.load_datasets()
    tokenized_datasets = data_processor.tokenize_datasets(raw_datasets)
    
    # Create combined training dataset with balanced sampling
    logger.info("Creating balanced multi-task dataset...")
    sampler = MultiTaskSampler(tokenized_datasets, config)
    train_dataset = sampler.create_combined_dataset()
    
    # Combine validation datasets
    eval_datasets = {}
    for task_name, task_dataset in tokenized_datasets.items():
        if 'validation' in task_dataset:
            eval_datasets[task_name] = task_dataset['validation']
    
    # Create combined evaluation dataset
    eval_data_combined = []
    for task_name, eval_dataset in eval_datasets.items():
        for item in eval_dataset:
            item['task_id'] = task_name
            eval_data_combined.append(item)
    
    eval_dataset = Dataset.from_dict({
        key: [item[key] for item in eval_data_combined]
        for key in eval_data_combined[0].keys()
    }) if eval_data_combined else None
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.base_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,
        do_train=config.do_train,
        do_eval=config.do_eval,
        do_predict=config.do_predict,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        load_best_model_at_end=config.load_best_model_at_end,
        report_to="tensorboard",
        run_name=f"multitask-{config.model_name_or_path.replace('/', '-')}",
    )
    
    # Setup evaluator
    from .evaluation_metrics import MultiTaskEvaluator
    evaluator = MultiTaskEvaluator(tokenizer)
    
    # Setup trainer
    trainer = MultiTaskTrainer(
        config=config,
        evaluator=evaluator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Check for checkpoints
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif Path(config.output_dir).exists():
        checkpoint = get_last_checkpoint(config.output_dir)
        if checkpoint is not None:
            logger.info(f"Found checkpoint at {checkpoint}")
    
    # Train model
    if config.do_train:
        logger.info("Starting multi-task training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save model
        trainer.save_model()
        trainer.save_state()
        
        # Save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    # Evaluate model
    if config.do_eval:
        logger.info("Starting evaluation...")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)
    
    logger.info("Multi-task training completed successfully!")


if __name__ == "__main__":
    main()

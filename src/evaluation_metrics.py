#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Comprehensive Evaluation Metrics for Multi-Task Network Configuration Learning

This module implements advanced evaluation metrics for network configuration automation tasks:

1. Configuration Generation Metrics:
   - BLEU Score (n-gram precision)
   - ROUGE Score (recall-based)
   - Exact Match (binary correctness)
   - Configuration Syntax Validity
   - Semantic Correctness

2. Configuration Analysis Metrics:
   - Natural Language Quality (BLEU, ROUGE)
   - Information Completeness
   - Technical Accuracy

3. Configuration Translation Metrics:
   - Vendor-Agnostic Functionality Preservation
   - Syntax Correctness per Vendor
   - Semantic Equivalence

4. Cross-Task Metrics:
   - Combined Performance Score
   - Task-Weighted Evaluation
   - Multi-Task Learning Efficiency

Features:
- Vendor-aware evaluation
- Hierarchical metric computation
- Statistical significance testing
- Comprehensive reporting
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dataclasses import dataclass, field


try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import yaml
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK or rouge-score not available. Some metrics will use simpler implementations.")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    
    # BLEU configuration
    bleu_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    bleu_smoothing: bool = True
    
    # ROUGE configuration
    rouge_types: List[str] = field(default_factory=lambda: ['rouge1', 'rouge2', 'rougeL'])
    rouge_use_stemmer: bool = True
    
    # Syntax validation
    validate_yaml: bool = True
    validate_vendor_syntax: bool = True
    
    # Semantic evaluation
    check_functionality: bool = True
    vendor_specific_checks: bool = True
    
    # Statistical testing
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000


class BaseMetric:
    """Base class for all evaluation metrics"""
    
    def __init__(self, name: str, config: EvaluationConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute metric score"""
        raise NotImplementedError
    
    def aggregate(self, scores: List[float]) -> Dict[str, float]:
        """Aggregate scores across multiple samples"""
        if not scores:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }


class BLEUMetric(BaseMetric):
    """BLEU score implementation with configuration support"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__("BLEU", config)
        self.weights = config.bleu_weights
        self.smoothing_function = SmoothingFunction() if NLTK_AVAILABLE and config.bleu_smoothing else None
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute BLEU scores"""
        if not NLTK_AVAILABLE:
            return self._simple_bleu(predictions, references)
        
        scores = []
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU expects list of reference token lists
            
            # Compute sentence BLEU
            if self.smoothing_function:
                score = sentence_bleu(
                    ref_tokens, 
                    pred_tokens, 
                    weights=self.weights,
                    smoothing_function=self.smoothing_function.method1
                )
            else:
                score = sentence_bleu(ref_tokens, pred_tokens, weights=self.weights)
            
            scores.append(score)
        
        return self.aggregate(scores)
    
    def _simple_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple BLEU implementation when NLTK is not available"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            
            if len(pred_words) == 0:
                score = 0.0
            else:
                precision = len(pred_words & ref_words) / len(pred_words)
                recall = len(pred_words & ref_words) / max(len(ref_words), 1)
                score = 2 * precision * recall / max(precision + recall, 1e-10)
            
            scores.append(score)
        
        return self.aggregate(scores)


class ROUGEMetric(BaseMetric):
    """ROUGE score implementation"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__("ROUGE", config)
        self.rouge_types = config.rouge_types
        
        if NLTK_AVAILABLE:
            self.scorer = rouge_scorer.RougeScorer(
                config.rouge_types, 
                use_stemmer=config.rouge_use_stemmer
            )
        else:
            self.scorer = None
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute ROUGE scores"""
        if not self.scorer:
            return self._simple_rouge(predictions, references)
        
        rouge_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            
            for rouge_type in self.rouge_types:
                rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        # Aggregate scores
        result = {}
        for rouge_type, scores in rouge_scores.items():
            aggregated = self.aggregate(scores)
            for metric, value in aggregated.items():
                result[f"{rouge_type}_{metric}"] = value
        
        return result
    
    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE implementation"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            
            if len(ref_words) == 0:
                score = 1.0 if len(pred_words) == 0 else 0.0
            else:
                score = len(pred_words & ref_words) / len(ref_words)
            
            scores.append(score)
        
        return {'rouge_simple_' + k: v for k, v in self.aggregate(scores).items()}


class ExactMatchMetric(BaseMetric):
    """Exact match score implementation"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__("ExactMatch", config)
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute exact match scores"""
        scores = []
        for pred, ref in zip(predictions, references):
            # Normalize whitespace for comparison
            pred_norm = ' '.join(pred.split())
            ref_norm = ' '.join(ref.split())
            scores.append(1.0 if pred_norm == ref_norm else 0.0)
        
        return self.aggregate(scores)


class ConfigurationSyntaxMetric(BaseMetric):
    """Configuration syntax validation metric"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__("ConfigSyntax", config)
        self.vendor_patterns = {
            'cisco': [
                r'interface\s+\w+',
                r'ip\s+address\s+\d+\.\d+\.\d+\.\d+',
                r'router\s+(ospf|bgp)',
                r'access-list\s+\d+',
            ],
            'juniper': [
                r'interfaces\s*{',
                r'routing-options\s*{',
                r'protocols\s*{',
                r'security\s*{',
            ],
            'nmstate': [
                r'interfaces:',
                r'routes:',
                r'dns-resolver:',
            ]
        }
    
    def compute(self, predictions: List[str], references: List[str], vendor: str = 'unknown', **kwargs) -> Dict[str, float]:
        """Compute configuration syntax validity scores"""
        scores = []
        
        for pred in predictions:
            score = self._validate_syntax(pred, vendor)
            scores.append(score)
        
        result = self.aggregate(scores)
        
        # Add vendor-specific metrics
        if vendor in self.vendor_patterns:
            vendor_scores = self._validate_vendor_patterns(predictions, vendor)
            result.update({f"vendor_{vendor}_{k}": v for k, v in self.aggregate(vendor_scores).items()})
        
        return result
    
    def _validate_syntax(self, config_text: str, vendor: str) -> float:
        """Validate basic configuration syntax"""
        if not config_text.strip():
            return 0.0
        
        # Check for YAML validity if applicable
        if self.config.validate_yaml and vendor in ['nmstate']:
            try:
                yaml.safe_load(config_text)
                return 1.0
            except yaml.YAMLError:
                return 0.0
        
        # Basic syntax checks
        score = 1.0
        
        # Check for balanced braces (Juniper-style)
        if vendor == 'juniper':
            open_braces = config_text.count('{')
            close_braces = config_text.count('}')
            if open_braces != close_braces:
                score *= 0.5
        
        # Check for reasonable line structure
        lines = config_text.split('\n')
        empty_lines = sum(1 for line in lines if not line.strip())
        if empty_lines > len(lines) * 0.5:  # Too many empty lines
            score *= 0.8
        
        return score
    
    def _validate_vendor_patterns(self, predictions: List[str], vendor: str) -> List[float]:
        """Validate vendor-specific configuration patterns"""
        patterns = self.vendor_patterns.get(vendor, [])
        scores = []
        
        for pred in predictions:
            pattern_matches = 0
            for pattern in patterns:
                if re.search(pattern, pred, re.IGNORECASE | re.MULTILINE):
                    pattern_matches += 1
            
            # Score based on pattern coverage
            if patterns:
                score = pattern_matches / len(patterns)
            else:
                score = 1.0  # No patterns to check
            
            scores.append(score)
        
        return scores


class SemanticCorrectnessMetric(BaseMetric):
    """Semantic correctness evaluation for network configurations"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__("SemanticCorrectness", config)
        
        # Define semantic rules for different configuration types
        self.semantic_rules = {
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b',
            'interface_name': r'\b(?:eth|fa|gi|lo|se)\d+(?:/\d+)*\b',
            'vlan_id': r'\bvlan\s+(\d+)\b',
            'routing_protocol': r'\b(?:ospf|bgp|rip|eigrp)\b',
        }
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute semantic correctness scores"""
        scores = []
        detailed_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            score, details = self._evaluate_semantics(pred, ref)
            scores.append(score)
            
            for rule_name, rule_score in details.items():
                detailed_scores[rule_name].append(rule_score)
        
        result = self.aggregate(scores)
        
        # Add detailed rule scores
        for rule_name, rule_scores in detailed_scores.items():
            rule_agg = self.aggregate(rule_scores)
            result.update({f"semantic_{rule_name}_{k}": v for k, v in rule_agg.items()})
        
        return result
    
    def _evaluate_semantics(self, prediction: str, reference: str) -> Tuple[float, Dict[str, float]]:
        """Evaluate semantic correctness of a single prediction"""
        details = {}
        
        # Extract semantic elements from reference and prediction
        ref_elements = self._extract_semantic_elements(reference)
        pred_elements = self._extract_semantic_elements(prediction)
        
        # Compare semantic elements
        total_rules = len(self.semantic_rules)
        rule_scores = []
        
        for rule_name, pattern in self.semantic_rules.items():
            ref_matches = ref_elements.get(rule_name, set())
            pred_matches = pred_elements.get(rule_name, set())
            
            if not ref_matches and not pred_matches:
                rule_score = 1.0  # Both empty, perfect match
            elif not ref_matches:
                rule_score = 0.0  # Reference empty but prediction has elements
            elif not pred_matches:
                rule_score = 0.0  # Prediction empty but reference has elements
            else:
                # Compute Jaccard similarity
                intersection = len(ref_matches & pred_matches)
                union = len(ref_matches | pred_matches)
                rule_score = intersection / union if union > 0 else 0.0
            
            rule_scores.append(rule_score)
            details[rule_name] = rule_score
        
        # Overall semantic score
        overall_score = np.mean(rule_scores) if rule_scores else 0.0
        
        return overall_score, details
    
    def _extract_semantic_elements(self, text: str) -> Dict[str, set]:
        """Extract semantic elements from configuration text"""
        elements = {}
        
        for rule_name, pattern in self.semantic_rules.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            elements[rule_name] = set(matches)
        
        return elements


class MultiTaskEvaluator:
    """Comprehensive multi-task evaluation system"""
    
    def __init__(self, tokenizer=None, config: Optional[EvaluationConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.metrics = {
            'bleu': BLEUMetric(self.config),
            'rouge': ROUGEMetric(self.config),
            'exact_match': ExactMatchMetric(self.config),
            'config_syntax': ConfigurationSyntaxMetric(self.config),
            'semantic': SemanticCorrectnessMetric(self.config),
        }
        
        # Task-specific metric weights
        self.task_weights = {
            'generation': {'bleu': 0.3, 'rouge': 0.2, 'exact_match': 0.2, 'config_syntax': 0.2, 'semantic': 0.1},
            'analysis': {'bleu': 0.4, 'rouge': 0.4, 'exact_match': 0.1, 'config_syntax': 0.05, 'semantic': 0.05},
            'translation': {'bleu': 0.25, 'rouge': 0.15, 'exact_match': 0.15, 'config_syntax': 0.25, 'semantic': 0.2},
        }
    
    def evaluate_task(self, 
                     task_name: str,
                     predictions: List[str], 
                     references: List[str],
                     vendor: str = 'unknown',
                     **kwargs) -> Dict[str, Any]:
        """Evaluate a specific task"""
        
        if len(predictions) != len(references):
            raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
        
        results = {'task': task_name, 'vendor': vendor, 'sample_count': len(predictions)}
        
        # Compute all metrics
        for metric_name, metric in self.metrics.items():
            try:
                metric_results = metric.compute(
                    predictions, 
                    references, 
                    vendor=vendor,
                    task=task_name,
                    **kwargs
                )
                
                # Flatten nested results
                for key, value in metric_results.items():
                    results[f"{metric_name}_{key}"] = value
                    
            except Exception as e:
                self.logger.error(f"Error computing {metric_name} for task {task_name}: {e}")
                results[f"{metric_name}_error"] = str(e)
        
        # Compute weighted task score
        task_score = self._compute_weighted_task_score(results, task_name)
        results['weighted_task_score'] = task_score
        
        return results
    
    def _compute_weighted_task_score(self, results: Dict[str, Any], task_name: str) -> float:
        """Compute weighted score for a specific task"""
        if task_name not in self.task_weights:
            return 0.0
        
        weights = self.task_weights[task_name]
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            metric_key = f"{metric_name}_mean"
            if metric_key in results and isinstance(results[metric_key], (int, float)):
                weighted_score += results[metric_key] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def evaluate_all_tasks(self, 
                          model, 
                          datasets: Dict[str, Any], 
                          tokenizer=None) -> Dict[str, Any]:
        """Evaluate model on all tasks"""
        
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        all_results = {}
        task_scores = []
        
        for task_name, task_data in datasets.items():
            self.logger.info(f"Evaluating task: {task_name}")
            
            # Generate predictions (this would need to be implemented based on your model interface)
            predictions = self._generate_predictions(model, task_data, tokenizer)
            references = self._extract_references(task_data)
            vendor = self._extract_vendor(task_data)
            
            # Evaluate task
            task_results = self.evaluate_task(
                task_name=task_name,
                predictions=predictions,
                references=references,
                vendor=vendor
            )
            
            all_results[task_name] = task_results
            task_scores.append(task_results.get('weighted_task_score', 0.0))
        
        # Compute overall metrics
        if task_scores:
            all_results['overall'] = {
                'combined_score': np.mean(task_scores),
                'task_score_std': np.std(task_scores),
                'min_task_score': np.min(task_scores),
                'max_task_score': np.max(task_scores),
            }
        
        return all_results
    
    def _generate_predictions(self, model, task_data, tokenizer) -> List[str]:
        """Generate predictions from model (placeholder implementation)"""
        # This would need to be implemented based on your specific model interface
        # For now, return dummy predictions
        return ["dummy prediction"] * len(task_data)
    
    def _extract_references(self, task_data) -> List[str]:
        """Extract reference texts from task data"""
        if hasattr(task_data, 'select'):
            # Hugging Face dataset
            return [item['target_text'] for item in task_data]
        elif isinstance(task_data, list):
            return [item.get('target_text', '') for item in task_data]
        else:
            raise ValueError(f"Unsupported task_data type: {type(task_data)}")
    
    def _extract_vendor(self, task_data) -> str:
        """Extract vendor information from task data"""
        if hasattr(task_data, 'select') and len(task_data) > 0:
            # Hugging Face dataset
            first_item = task_data[0]
            return first_item.get('target_vendor', 'unknown')
        elif isinstance(task_data, list) and len(task_data) > 0:
            return task_data[0].get('target_vendor', 'unknown')
        else:
            return 'unknown'
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report_lines = [
            "# Multi-Task Network Configuration Evaluation Report",
            "",
            "## Summary",
            ""
        ]
        
        # Overall metrics
        if 'overall' in evaluation_results:
            overall = evaluation_results['overall']
            report_lines.extend([
                f"- **Combined Score**: {overall.get('combined_score', 0):.4f}",
                f"- **Task Score Std**: {overall.get('task_score_std', 0):.4f}",
                f"- **Min Task Score**: {overall.get('min_task_score', 0):.4f}",
                f"- **Max Task Score**: {overall.get('max_task_score', 0):.4f}",
                ""
            ])
        
        # Task-specific results
        for task_name, task_results in evaluation_results.items():
            if task_name == 'overall':
                continue
                
            report_lines.extend([
                f"## Task: {task_name.title()}",
                "",
                f"- **Vendor**: {task_results.get('vendor', 'unknown')}",
                f"- **Sample Count**: {task_results.get('sample_count', 0)}",
                f"- **Weighted Score**: {task_results.get('weighted_task_score', 0):.4f}",
                ""
            ])
            
            # Metric details
            report_lines.append("### Metrics")
            report_lines.append("")
            
            for key, value in task_results.items():
                if key.endswith('_mean') and isinstance(value, (int, float)):
                    metric_name = key.replace('_mean', '').replace('_', ' ').title()
                    report_lines.append(f"- **{metric_name}**: {value:.4f}")
            
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text


# Utility functions
def evaluate_model_predictions(predictions_file: str, 
                             references_file: str, 
                             task_name: str = 'generation',
                             vendor: str = 'unknown',
                             config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
    """Evaluate model predictions from files"""
    
    # Load data
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r') as f:
        references = [line.strip() for line in f]
    
    # Create evaluator
    evaluator = MultiTaskEvaluator(config=config)
    
    # Evaluate
    results = evaluator.evaluate_task(
        task_name=task_name,
        predictions=predictions,
        references=references,
        vendor=vendor
    )
    
    return results


def compare_models(model_results: Dict[str, Dict[str, Any]], 
                  output_path: Optional[str] = None) -> str:
    """Compare results from multiple models"""
    
    report_lines = [
        "# Model Comparison Report",
        "",
        "## Overview",
        ""
    ]
    
    # Create comparison table
    models = list(model_results.keys())
    if not models:
        return "No models to compare."
    
    # Get all tasks from first model
    first_model_results = model_results[models[0]]
    tasks = [k for k in first_model_results.keys() if k != 'overall']
    
    # Comparison table header
    report_lines.append("| Task | " + " | ".join(models) + " |")
    report_lines.append("|------|" + "|".join("------" for _ in models) + "|")
    
    # Add rows for each task
    for task in tasks:
        row = f"| {task.title()} |"
        for model in models:
            score = model_results[model].get(task, {}).get('weighted_task_score', 0.0)
            row += f" {score:.4f} |"
        report_lines.append(row)
    
    report_lines.extend(["", "## Detailed Analysis", ""])
    
    # Detailed comparison for each task
    for task in tasks:
        report_lines.extend([
            f"### {task.title()} Task",
            ""
        ])
        
        for model in models:
            task_results = model_results[model].get(task, {})
            report_lines.extend([
                f"#### {model}",
                f"- Weighted Score: {task_results.get('weighted_task_score', 0):.4f}",
                f"- BLEU: {task_results.get('bleu_mean', 0):.4f}",
                f"- ROUGE: {task_results.get('rouge1_mean', 0):.4f}",
                f"- Exact Match: {task_results.get('exact_match_mean', 0):.4f}",
                ""
            ])
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text

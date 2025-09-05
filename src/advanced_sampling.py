"""
Advanced Sampling Strategies for Multi-Task Network Configuration Training
=======================================================================

This module provides sophisticated sampling techniques to handle imbalanced datasets
across multiple tasks, vendors, and configuration types in network automation.

Key Features:
- Stratified sampling across vendors and task types
- Dynamic sample weighting based on difficulty and frequency
- Curriculum learning support with progressive difficulty
- Adaptive resampling based on model performance
- Minority class oversampling and majority class undersampling
"""

import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class SampleMetadata:
    """Metadata for training samples used in advanced sampling strategies."""
    vendor: str
    task_type: str
    difficulty: str
    category: str
    sample_id: str
    performance_score: Optional[float] = None
    error_count: int = 0
    selection_count: int = 0

class AdvancedSampler:
    """
    Advanced sampling strategies for multi-task network configuration training.
    
    Implements multiple sampling techniques to ensure balanced and effective training
    across different vendors, task types, and difficulty levels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_metadata: Dict[str, SampleMetadata] = {}
        self.vendor_weights = config.get('vendor_weights', {})
        self.task_weights = config.get('task_weights', {})
        self.difficulty_weights = config.get('difficulty_weights', {})
        self.sampling_strategy = config.get('sampling_strategy', 'balanced')
        self.curriculum_enabled = config.get('curriculum_learning', False)
        self.current_epoch = 0
        self.performance_history = defaultdict(list)
        
    def register_samples(self, samples: List[Dict[str, Any]]):
        """Register samples with metadata for advanced sampling."""
        for i, sample in enumerate(samples):
            sample_id = f"{sample.get('vendor', 'unknown')}_{sample.get('task_type', 'unknown')}_{i}"
            
            metadata = SampleMetadata(
                vendor=sample.get('vendor', 'unknown'),
                task_type=sample.get('task_type', 'generation'),
                difficulty=sample.get('metadata', {}).get('difficulty', 'basic'),
                category=sample.get('metadata', {}).get('category', 'general'),
                sample_id=sample_id
            )
            
            self.sample_metadata[sample_id] = metadata
            
        logger.info(f"Registered {len(samples)} samples for advanced sampling")
        
    def analyze_dataset_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of samples across different dimensions."""
        vendor_dist = Counter(meta.vendor for meta in self.sample_metadata.values())
        task_dist = Counter(meta.task_type for meta in self.sample_metadata.values())
        difficulty_dist = Counter(meta.difficulty for meta in self.sample_metadata.values())
        category_dist = Counter(meta.category for meta in self.sample_metadata.values())
        
        analysis = {
            'total_samples': len(self.sample_metadata),
            'vendor_distribution': dict(vendor_dist),
            'task_distribution': dict(task_dist),
            'difficulty_distribution': dict(difficulty_dist),
            'category_distribution': dict(category_dist),
            'imbalance_ratios': self._calculate_imbalance_ratios(vendor_dist, task_dist, difficulty_dist)
        }
        
        logger.info(f"Dataset analysis: {analysis}")
        return analysis
        
    def _calculate_imbalance_ratios(self, vendor_dist: Counter, task_dist: Counter, 
                                   difficulty_dist: Counter) -> Dict[str, float]:
        """Calculate imbalance ratios for different distributions."""
        ratios = {}
        
        # Vendor imbalance
        if vendor_dist:
            max_vendor = max(vendor_dist.values())
            min_vendor = min(vendor_dist.values())
            ratios['vendor_imbalance'] = max_vendor / min_vendor if min_vendor > 0 else float('inf')
            
        # Task imbalance
        if task_dist:
            max_task = max(task_dist.values())
            min_task = min(task_dist.values())
            ratios['task_imbalance'] = max_task / min_task if min_task > 0 else float('inf')
            
        # Difficulty imbalance
        if difficulty_dist:
            max_diff = max(difficulty_dist.values())
            min_diff = min(difficulty_dist.values())
            ratios['difficulty_imbalance'] = max_diff / min_diff if min_diff > 0 else float('inf')
            
        return ratios
        
    def stratified_sample(self, samples: List[Dict[str, Any]], 
                         batch_size: int) -> List[Dict[str, Any]]:
        """
        Perform stratified sampling to ensure representation across all strata.
        
        Ensures each batch contains samples from different vendors, tasks, and difficulties.
        """
        if not samples:
            return []
            
        # Group samples by strata (vendor + task + difficulty)
        strata = defaultdict(list)
        for sample in samples:
            vendor = sample.get('vendor', 'unknown')
            task_type = sample.get('task_type', 'generation')
            difficulty = sample.get('metadata', {}).get('difficulty', 'basic')
            stratum_key = f"{vendor}_{task_type}_{difficulty}"
            strata[stratum_key].append(sample)
            
        # Calculate samples per stratum
        num_strata = len(strata)
        if num_strata == 0:
            return random.sample(samples, min(batch_size, len(samples)))
            
        base_samples_per_stratum = batch_size // num_strata
        remainder = batch_size % num_strata
        
        stratified_batch = []
        strata_keys = list(strata.keys())
        
        for i, stratum_key in enumerate(strata_keys):
            stratum_samples = strata[stratum_key]
            samples_to_take = base_samples_per_stratum + (1 if i < remainder else 0)
            
            if samples_to_take > 0:
                selected = random.sample(stratum_samples, 
                                       min(samples_to_take, len(stratum_samples)))
                stratified_batch.extend(selected)
                
        # Fill remaining slots if needed
        while len(stratified_batch) < batch_size and len(stratified_batch) < len(samples):
            remaining_samples = [s for s in samples if s not in stratified_batch]
            if remaining_samples:
                stratified_batch.append(random.choice(remaining_samples))
                
        return stratified_batch[:batch_size]
        
    def weighted_sample(self, samples: List[Dict[str, Any]], 
                       batch_size: int) -> List[Dict[str, Any]]:
        """
        Perform weighted sampling based on vendor, task, and difficulty weights.
        
        Samples are selected with probability proportional to their composite weight.
        """
        if not samples:
            return []
            
        weights = []
        for sample in samples:
            vendor = sample.get('vendor', 'unknown')
            task_type = sample.get('task_type', 'generation')
            difficulty = sample.get('metadata', {}).get('difficulty', 'basic')
            
            vendor_weight = self.vendor_weights.get(vendor, 1.0)
            task_weight = self.task_weights.get(task_type, 1.0)
            difficulty_weight = self.difficulty_weights.get(difficulty, 1.0)
            
            # Composite weight with performance adjustment
            sample_id = f"{vendor}_{task_type}_{samples.index(sample)}"
            performance_factor = self._get_performance_factor(sample_id)
            
            composite_weight = vendor_weight * task_weight * difficulty_weight * performance_factor
            weights.append(composite_weight)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(samples)] * len(samples)
            
        # Weighted sampling without replacement
        selected_indices = []
        remaining_indices = list(range(len(samples)))
        remaining_weights = weights.copy()
        
        for _ in range(min(batch_size, len(samples))):
            if not remaining_indices:
                break
                
            # Renormalize remaining weights
            total_remaining = sum(remaining_weights)
            if total_remaining > 0:
                normalized_weights = [w / total_remaining for w in remaining_weights]
            else:
                normalized_weights = [1.0 / len(remaining_weights)] * len(remaining_weights)
                
            # Select index
            selected_idx = np.random.choice(len(remaining_indices), p=normalized_weights)
            actual_idx = remaining_indices[selected_idx]
            selected_indices.append(actual_idx)
            
            # Remove from remaining
            remaining_indices.pop(selected_idx)
            remaining_weights.pop(selected_idx)
            
        return [samples[i] for i in selected_indices]
        
    def curriculum_sample(self, samples: List[Dict[str, Any]], 
                         batch_size: int, epoch: int) -> List[Dict[str, Any]]:
        """
        Implement curriculum learning by gradually introducing harder samples.
        
        Starts with easy samples and progressively includes more difficult ones.
        """
        self.current_epoch = epoch
        
        # Define curriculum schedule
        curriculum_progress = min(1.0, epoch / self.config.get('curriculum_epochs', 10))
        
        # Difficulty ordering
        difficulty_order = ['basic', 'intermediate', 'advanced']
        
        # Filter samples based on curriculum progress
        available_samples = []
        max_difficulty_idx = int(curriculum_progress * len(difficulty_order))
        
        for sample in samples:
            difficulty = sample.get('metadata', {}).get('difficulty', 'basic')
            if difficulty in difficulty_order:
                difficulty_idx = difficulty_order.index(difficulty)
                if difficulty_idx <= max_difficulty_idx:
                    available_samples.append(sample)
            else:
                available_samples.append(sample)  # Include unknown difficulties
                
        if not available_samples:
            available_samples = samples  # Fallback to all samples
            
        # Apply stratified sampling to curriculum-filtered samples
        return self.stratified_sample(available_samples, batch_size)
        
    def adaptive_sample(self, samples: List[Dict[str, Any]], 
                       batch_size: int) -> List[Dict[str, Any]]:
        """
        Adaptive sampling based on model performance on different sample types.
        
        Increases sampling probability for poorly performing sample types.
        """
        if not samples:
            return []
            
        # Group samples by performance characteristics
        performance_groups = defaultdict(list)
        for sample in samples:
            vendor = sample.get('vendor', 'unknown')
            task_type = sample.get('task_type', 'generation')
            group_key = f"{vendor}_{task_type}"
            performance_groups[group_key].append(sample)
            
        # Calculate inverse performance weights
        group_weights = {}
        for group_key, group_samples in performance_groups.items():
            avg_performance = self._get_group_performance(group_key)
            # Higher weight for lower performance (needs more training)
            weight = 1.0 / (avg_performance + 0.1)  # Add small epsilon to avoid division by zero
            group_weights[group_key] = weight
            
        # Normalize group weights
        total_weight = sum(group_weights.values())
        if total_weight > 0:
            group_weights = {k: v / total_weight for k, v in group_weights.items()}
            
        # Sample from groups proportionally
        sampled_batch = []
        for group_key, weight in group_weights.items():
            group_samples = performance_groups[group_key]
            samples_from_group = max(1, int(batch_size * weight))
            
            if group_samples:
                selected = random.sample(group_samples, 
                                       min(samples_from_group, len(group_samples)))
                sampled_batch.extend(selected)
                
        # Adjust to exact batch size
        if len(sampled_batch) > batch_size:
            sampled_batch = random.sample(sampled_batch, batch_size)
        elif len(sampled_batch) < batch_size:
            remaining_samples = [s for s in samples if s not in sampled_batch]
            additional_needed = batch_size - len(sampled_batch)
            if remaining_samples:
                additional = random.sample(remaining_samples, 
                                         min(additional_needed, len(remaining_samples)))
                sampled_batch.extend(additional)
                
        return sampled_batch
        
    def get_batch(self, samples: List[Dict[str, Any]], 
                  batch_size: int, epoch: int = 0) -> List[Dict[str, Any]]:
        """
        Get a batch using the configured sampling strategy.
        
        Routes to the appropriate sampling method based on configuration.
        """
        if not samples:
            return []
            
        strategy = self.sampling_strategy.lower()
        
        if strategy == 'stratified':
            return self.stratified_sample(samples, batch_size)
        elif strategy == 'weighted':
            return self.weighted_sample(samples, batch_size)
        elif strategy == 'curriculum':
            return self.curriculum_sample(samples, batch_size, epoch)
        elif strategy == 'adaptive':
            return self.adaptive_sample(samples, batch_size)
        elif strategy == 'balanced':
            # Combination of stratified and weighted sampling
            stratified_batch = self.stratified_sample(samples, batch_size)
            return self.weighted_sample(stratified_batch, batch_size)
        else:
            # Default random sampling
            return random.sample(samples, min(batch_size, len(samples)))
            
    def update_performance(self, sample_id: str, performance_score: float, 
                          had_error: bool = False):
        """Update performance metrics for a sample."""
        if sample_id in self.sample_metadata:
            metadata = self.sample_metadata[sample_id]
            metadata.performance_score = performance_score
            if had_error:
                metadata.error_count += 1
            metadata.selection_count += 1
            
            # Update group performance history
            group_key = f"{metadata.vendor}_{metadata.task_type}"
            self.performance_history[group_key].append(performance_score)
            
    def _get_performance_factor(self, sample_id: str) -> float:
        """Get performance-based sampling factor for a sample."""
        if sample_id not in self.sample_metadata:
            return 1.0
            
        metadata = self.sample_metadata[sample_id]
        if metadata.performance_score is None:
            return 1.0
            
        # Lower performance = higher sampling probability
        return 2.0 - metadata.performance_score  # Assumes score is between 0 and 1
        
    def _get_group_performance(self, group_key: str) -> float:
        """Get average performance for a sample group."""
        if group_key not in self.performance_history:
            return 0.5  # Default moderate performance
            
        performance_scores = self.performance_history[group_key]
        return np.mean(performance_scores[-100:])  # Use recent performance
        
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about sampling behavior."""
        stats = {
            'total_samples': len(self.sample_metadata),
            'vendor_counts': Counter(meta.vendor for meta in self.sample_metadata.values()),
            'task_counts': Counter(meta.task_type for meta in self.sample_metadata.values()),
            'difficulty_counts': Counter(meta.difficulty for meta in self.sample_metadata.values()),
            'selection_counts': {
                sample_id: meta.selection_count 
                for sample_id, meta in self.sample_metadata.items()
            },
            'error_rates': {
                sample_id: meta.error_count / max(1, meta.selection_count)
                for sample_id, meta in self.sample_metadata.items()
            },
            'performance_averages': {
                group: np.mean(scores[-50:]) if scores else 0.0
                for group, scores in self.performance_history.items()
            }
        }
        
        return stats


class MinorityOversampler:
    """Oversampling techniques for minority classes in network configuration data."""
    
    def __init__(self, min_samples_per_class: int = 10):
        self.min_samples_per_class = min_samples_per_class
        
    def oversample_vendors(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Oversample minority vendor configurations."""
        vendor_groups = defaultdict(list)
        for sample in samples:
            vendor = sample.get('vendor', 'unknown')
            vendor_groups[vendor].append(sample)
            
        oversampled = []
        for vendor, vendor_samples in vendor_groups.items():
            if len(vendor_samples) < self.min_samples_per_class:
                # Duplicate samples to reach minimum
                multiplier = math.ceil(self.min_samples_per_class / len(vendor_samples))
                vendor_samples = vendor_samples * multiplier
                vendor_samples = vendor_samples[:self.min_samples_per_class]
                
            oversampled.extend(vendor_samples)
            
        return oversampled
        
    def oversample_tasks(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Oversample minority task types."""
        task_groups = defaultdict(list)
        for sample in samples:
            task_type = sample.get('task_type', 'generation')
            task_groups[task_type].append(sample)
            
        oversampled = []
        for task_type, task_samples in task_groups.items():
            if len(task_samples) < self.min_samples_per_class:
                # Duplicate samples to reach minimum
                multiplier = math.ceil(self.min_samples_per_class / len(task_samples))
                task_samples = task_samples * multiplier
                task_samples = task_samples[:self.min_samples_per_class]
                
            oversampled.extend(task_samples)
            
        return oversampled


def create_advanced_sampler(config: Dict[str, Any]) -> AdvancedSampler:
    """Factory function to create an advanced sampler with configuration."""
    return AdvancedSampler(config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'vendor_weights': {'cisco': 1.0, 'juniper': 1.2, 'nmstate': 0.8},
        'task_weights': {'generation': 1.0, 'analysis': 1.1, 'translation': 1.2},
        'difficulty_weights': {'basic': 0.8, 'intermediate': 1.0, 'advanced': 1.3},
        'sampling_strategy': 'balanced',
        'curriculum_learning': True,
        'curriculum_epochs': 10
    }
    
    # Create sampler
    sampler = create_advanced_sampler(config)
    
    # Test with sample data
    test_samples = [
        {'vendor': 'cisco', 'task_type': 'generation', 'metadata': {'difficulty': 'basic'}},
        {'vendor': 'juniper', 'task_type': 'analysis', 'metadata': {'difficulty': 'intermediate'}},
        {'vendor': 'nmstate', 'task_type': 'translation', 'metadata': {'difficulty': 'advanced'}},
    ]
    
    sampler.register_samples(test_samples)
    analysis = sampler.analyze_dataset_distribution()
    print(f"Dataset analysis: {analysis}")
    
    batch = sampler.get_batch(test_samples, batch_size=2, epoch=5)
    print(f"Sample batch: {len(batch)} samples")
    
    stats = sampler.get_sampling_statistics()
    print(f"Sampling statistics: {stats}")
    
    print("Advanced sampling system ready for multi-task training!")

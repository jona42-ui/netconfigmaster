"""
Comprehensive Training Monitoring System for Multi-Task Network Configuration
============================================================================

This module provides advanced monitoring, logging, and checkpointing capabilities
for the multi-task network configuration training system.

Features:
- Real-time training metrics tracking
- Comprehensive logging with structured output
- Model checkpointing and recovery
- Performance visualization
- Early stopping based on multiple criteria
- Training progress reporting
- Resource usage monitoring
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from collections import defaultdict, deque

# Try importing optional dependencies with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific point in time."""
    epoch: int
    step: int
    timestamp: float
    total_loss: float
    task_losses: Dict[str, float]
    learning_rate: float
    vendor_accuracies: Dict[str, float]
    task_accuracies: Dict[str, float]
    batch_size: int
    samples_processed: int
    training_time: float
    memory_usage_mb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    
@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    epoch: int
    timestamp: float
    bleu_scores: Dict[str, float]
    rouge_scores: Dict[str, float]
    exact_match_scores: Dict[str, float]
    syntax_accuracy: Dict[str, float]
    semantic_accuracy: Dict[str, float]
    vendor_performance: Dict[str, Dict[str, float]]
    task_performance: Dict[str, Dict[str, float]]
    overall_score: float

class TrainingMonitor:
    """
    Comprehensive monitoring system for multi-task network configuration training.
    
    Tracks metrics, manages checkpoints, provides early stopping, and generates reports.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring configuration
        self.save_checkpoint_every = config.get('save_checkpoint_every', 1000)
        self.evaluate_every = config.get('evaluate_every', 500)
        self.log_every = config.get('log_every', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)
        
        # Metrics storage
        self.training_metrics: List[TrainingMetrics] = []
        self.evaluation_metrics: List[EvaluationMetrics] = []
        self.best_metrics = {'loss': float('inf'), 'accuracy': 0.0}
        self.patience_counter = 0
        self.should_stop_early = False
        
        # Performance tracking
        self.step_times = deque(maxlen=100)  # Track last 100 step times
        self.memory_usage = deque(maxlen=100)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize checkpoint tracking
        self.checkpoint_history = []
        self.best_checkpoint_path = None
        
        logger.info(f"Training monitor initialized with output directory: {self.output_dir}")
        
    def _setup_logging(self):
        """Setup structured logging for training monitoring."""
        log_file = self.output_dir / "training.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    def log_training_step(self, epoch: int, step: int, metrics: Dict[str, Any],
                         model_state: Any = None, optimizer_state: Any = None):
        """Log metrics for a training step."""
        timestamp = time.time()
        step_start_time = getattr(self, '_step_start_time', timestamp)
        step_time = timestamp - step_start_time
        self.step_times.append(step_time)
        
        # Extract metrics
        total_loss = metrics.get('total_loss', 0.0)
        task_losses = metrics.get('task_losses', {})
        learning_rate = metrics.get('learning_rate', 0.0)
        vendor_accuracies = metrics.get('vendor_accuracies', {})
        task_accuracies = metrics.get('task_accuracies', {})
        batch_size = metrics.get('batch_size', 0)
        samples_processed = metrics.get('samples_processed', 0)
        
        # Get system metrics
        memory_usage = self._get_memory_usage()
        gpu_usage = self._get_gpu_usage()
        
        # Create training metrics object
        training_metric = TrainingMetrics(
            epoch=epoch,
            step=step,
            timestamp=timestamp,
            total_loss=total_loss,
            task_losses=task_losses,
            learning_rate=learning_rate,
            vendor_accuracies=vendor_accuracies,
            task_accuracies=task_accuracies,
            batch_size=batch_size,
            samples_processed=samples_processed,
            training_time=step_time,
            memory_usage_mb=memory_usage,
            gpu_usage_percent=gpu_usage
        )
        
        self.training_metrics.append(training_metric)
        
        # Log to console/file
        if step % self.log_every == 0:
            self._log_training_progress(training_metric)
            
        # Save checkpoint
        if step % self.save_checkpoint_every == 0:
            self._save_checkpoint(epoch, step, model_state, optimizer_state, metrics)
            
        # Update best metrics
        self._update_best_metrics(training_metric)
        
    def log_evaluation_step(self, epoch: int, eval_metrics: Dict[str, Any]):
        """Log evaluation metrics."""
        timestamp = time.time()
        
        evaluation_metric = EvaluationMetrics(
            epoch=epoch,
            timestamp=timestamp,
            bleu_scores=eval_metrics.get('bleu_scores', {}),
            rouge_scores=eval_metrics.get('rouge_scores', {}),
            exact_match_scores=eval_metrics.get('exact_match_scores', {}),
            syntax_accuracy=eval_metrics.get('syntax_accuracy', {}),
            semantic_accuracy=eval_metrics.get('semantic_accuracy', {}),
            vendor_performance=eval_metrics.get('vendor_performance', {}),
            task_performance=eval_metrics.get('task_performance', {}),
            overall_score=eval_metrics.get('overall_score', 0.0)
        )
        
        self.evaluation_metrics.append(evaluation_metric)
        
        # Log evaluation results
        self._log_evaluation_progress(evaluation_metric)
        
        # Check for early stopping
        self._check_early_stopping(evaluation_metric)
        
        # Save metrics to file
        self._save_metrics()
        
    def _log_training_progress(self, metric: TrainingMetrics):
        """Log training progress to console and file."""
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        log_message = (
            f"Epoch {metric.epoch}, Step {metric.step}: "
            f"Loss={metric.total_loss:.4f}, "
            f"LR={metric.learning_rate:.6f}, "
            f"Time={metric.training_time:.2f}s, "
            f"AvgTime={avg_step_time:.2f}s"
        )
        
        if metric.memory_usage_mb:
            log_message += f", Memory={metric.memory_usage_mb:.1f}MB"
            
        if metric.task_losses:
            task_loss_str = ", ".join([f"{k}={v:.4f}" for k, v in metric.task_losses.items()])
            log_message += f", TaskLosses=({task_loss_str})"
            
        logger.info(log_message)
        
    def _log_evaluation_progress(self, metric: EvaluationMetrics):
        """Log evaluation progress."""
        log_message = (
            f"Evaluation Epoch {metric.epoch}: "
            f"Overall={metric.overall_score:.4f}"
        )
        
        if metric.bleu_scores:
            avg_bleu = sum(metric.bleu_scores.values()) / len(metric.bleu_scores)
            log_message += f", BLEU={avg_bleu:.4f}"
            
        if metric.exact_match_scores:
            avg_em = sum(metric.exact_match_scores.values()) / len(metric.exact_match_scores)
            log_message += f", ExactMatch={avg_em:.4f}"
            
        logger.info(log_message)
        
    def _save_checkpoint(self, epoch: int, step: int, model_state: Any, 
                        optimizer_state: Any, metrics: Dict[str, Any]):
        """Save model checkpoint with metadata."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics,
            'timestamp': time.time(),
            'config': self.config
        }
        
        try:
            # Save checkpoint (would use torch.save in real implementation)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            self.checkpoint_history.append({
                'path': str(checkpoint_path),
                'epoch': epoch,
                'step': step,
                'timestamp': time.time(),
                'loss': metrics.get('total_loss', float('inf'))
            })
            
            # Update best checkpoint
            if metrics.get('total_loss', float('inf')) < self.best_metrics['loss']:
                self.best_checkpoint_path = str(checkpoint_path)
                
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def _update_best_metrics(self, metric: TrainingMetrics):
        """Update best metrics tracking."""
        if metric.total_loss < self.best_metrics['loss']:
            self.best_metrics['loss'] = metric.total_loss
            
        # Calculate overall accuracy if available
        if metric.task_accuracies:
            overall_accuracy = sum(metric.task_accuracies.values()) / len(metric.task_accuracies)
            if overall_accuracy > self.best_metrics['accuracy']:
                self.best_metrics['accuracy'] = overall_accuracy
                
    def _check_early_stopping(self, eval_metric: EvaluationMetrics):
        """Check if training should stop early."""
        if len(self.evaluation_metrics) < 2:
            return
            
        # Compare with previous evaluation
        prev_metric = self.evaluation_metrics[-2]
        improvement = eval_metric.overall_score - prev_metric.overall_score
        
        if improvement > self.early_stopping_min_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stopping_patience:
            self.should_stop_early = True
            logger.warning(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return None
            
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return None
            
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage (placeholder implementation)."""
        # In a real implementation, this would use nvidia-ml-py or similar
        return None
        
    def _save_metrics(self):
        """Save all metrics to JSON files."""
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        # Save training metrics
        training_file = metrics_dir / "training_metrics.json"
        training_data = [asdict(metric) for metric in self.training_metrics]
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
            
        # Save evaluation metrics
        eval_file = metrics_dir / "evaluation_metrics.json"
        eval_data = [asdict(metric) for metric in self.evaluation_metrics]
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
            
        # Save checkpoint history
        checkpoint_file = metrics_dir / "checkpoint_history.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
            
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        if not self.training_metrics:
            return {"error": "No training metrics available"}
            
        report = {
            'training_summary': {
                'total_steps': len(self.training_metrics),
                'total_epochs': max(m.epoch for m in self.training_metrics) if self.training_metrics else 0,
                'training_duration': self._calculate_training_duration(),
                'best_loss': self.best_metrics['loss'],
                'best_accuracy': self.best_metrics['accuracy'],
                'final_loss': self.training_metrics[-1].total_loss if self.training_metrics else None
            },
            'performance_metrics': {
                'average_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
                'steps_per_second': 1.0 / (sum(self.step_times) / len(self.step_times)) if self.step_times else 0,
                'memory_usage_mb': list(self.memory_usage) if self.memory_usage else []
            },
            'task_performance': self._analyze_task_performance(),
            'vendor_performance': self._analyze_vendor_performance(),
            'convergence_analysis': self._analyze_convergence(),
            'checkpoints': {
                'total_checkpoints': len(self.checkpoint_history),
                'best_checkpoint': self.best_checkpoint_path,
                'checkpoint_history': self.checkpoint_history[-10:]  # Last 10 checkpoints
            }
        }
        
        # Save report
        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
        
    def _calculate_training_duration(self) -> float:
        """Calculate total training duration in seconds."""
        if len(self.training_metrics) < 2:
            return 0.0
            
        start_time = self.training_metrics[0].timestamp
        end_time = self.training_metrics[-1].timestamp
        return end_time - start_time
        
    def _analyze_task_performance(self) -> Dict[str, Any]:
        """Analyze performance across different tasks."""
        task_analysis = defaultdict(list)
        
        for metric in self.training_metrics:
            for task, accuracy in metric.task_accuracies.items():
                task_analysis[task].append(accuracy)
                
        task_summary = {}
        for task, accuracies in task_analysis.items():
            if accuracies:
                task_summary[task] = {
                    'average_accuracy': sum(accuracies) / len(accuracies),
                    'best_accuracy': max(accuracies),
                    'final_accuracy': accuracies[-1],
                    'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
                }
                
        return task_summary
        
    def _analyze_vendor_performance(self) -> Dict[str, Any]:
        """Analyze performance across different vendors."""
        vendor_analysis = defaultdict(list)
        
        for metric in self.training_metrics:
            for vendor, accuracy in metric.vendor_accuracies.items():
                vendor_analysis[vendor].append(accuracy)
                
        vendor_summary = {}
        for vendor, accuracies in vendor_analysis.items():
            if accuracies:
                vendor_summary[vendor] = {
                    'average_accuracy': sum(accuracies) / len(accuracies),
                    'best_accuracy': max(accuracies),
                    'final_accuracy': accuracies[-1],
                    'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
                }
                
        return vendor_summary
        
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence patterns."""
        if len(self.training_metrics) < 10:
            return {"error": "Insufficient data for convergence analysis"}
            
        losses = [m.total_loss for m in self.training_metrics]
        
        # Calculate moving averages
        window_size = min(50, len(losses) // 4)
        moving_avg = []
        for i in range(len(losses) - window_size + 1):
            avg = sum(losses[i:i + window_size]) / window_size
            moving_avg.append(avg)
            
        convergence_analysis = {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'total_improvement': losses[0] - losses[-1],
            'relative_improvement': (losses[0] - losses[-1]) / losses[0] if losses[0] != 0 else 0,
            'is_converged': self._is_converged(moving_avg),
            'convergence_step': self._find_convergence_point(moving_avg),
            'loss_volatility': self._calculate_volatility(losses)
        }
        
        return convergence_analysis
        
    def _is_converged(self, moving_avg: List[float], threshold: float = 0.001) -> bool:
        """Check if training has converged."""
        if len(moving_avg) < 10:
            return False
            
        recent_changes = [abs(moving_avg[i] - moving_avg[i-1]) for i in range(-10, 0)]
        avg_change = sum(recent_changes) / len(recent_changes)
        return avg_change < threshold
        
    def _find_convergence_point(self, moving_avg: List[float]) -> Optional[int]:
        """Find the approximate convergence point."""
        if len(moving_avg) < 20:
            return None
            
        # Look for the point where improvement rate drops significantly
        improvements = [moving_avg[i-1] - moving_avg[i] for i in range(1, len(moving_avg))]
        
        # Find where improvements become consistently small
        for i in range(10, len(improvements) - 10):
            recent_improvements = improvements[i:i+10]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            if avg_improvement < 0.001:  # Very small improvements
                return i
                
        return None
        
    def _calculate_volatility(self, losses: List[float]) -> float:
        """Calculate loss volatility (standard deviation)."""
        if len(losses) < 2:
            return 0.0
            
        mean_loss = sum(losses) / len(losses)
        variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
        return variance ** 0.5
        
    def create_visualizations(self):
        """Create training visualizations if matplotlib is available."""
        if not HAS_MATPLOTLIB or not self.training_metrics:
            logger.warning("Cannot create visualizations: matplotlib not available or no data")
            return
            
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Loss over time
        self._plot_loss_curves(viz_dir)
        
        # Accuracy over time
        self._plot_accuracy_curves(viz_dir)
        
        # Performance comparison
        self._plot_performance_comparison(viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
        
    def _plot_loss_curves(self, viz_dir: Path):
        """Plot training loss curves."""
        steps = [m.step for m in self.training_metrics]
        losses = [m.total_loss for m in self.training_metrics]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses, label='Total Loss', linewidth=2)
        
        # Plot task-specific losses if available
        task_losses = defaultdict(list)
        for metric in self.training_metrics:
            for task, loss in metric.task_losses.items():
                task_losses[task].append(loss)
                
        for task, task_loss_values in task_losses.items():
            if len(task_loss_values) == len(steps):
                plt.plot(steps, task_loss_values, label=f'{task.title()} Loss', alpha=0.7)
                
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_accuracy_curves(self, viz_dir: Path):
        """Plot accuracy curves."""
        if not any(m.task_accuracies for m in self.training_metrics):
            return
            
        steps = [m.step for m in self.training_metrics]
        
        plt.figure(figsize=(12, 6))
        
        # Plot task accuracies
        task_accuracies = defaultdict(list)
        for metric in self.training_metrics:
            for task, acc in metric.task_accuracies.items():
                task_accuracies[task].append(acc)
                
        for task, acc_values in task_accuracies.items():
            if len(acc_values) == len(steps):
                plt.plot(steps, acc_values, label=f'{task.title()} Accuracy', linewidth=2)
                
        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy')
        plt.title('Task Accuracy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(viz_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_performance_comparison(self, viz_dir: Path):
        """Plot vendor and task performance comparison."""
        if not self.evaluation_metrics:
            return
            
        latest_eval = self.evaluation_metrics[-1]
        
        # Vendor performance
        if latest_eval.vendor_performance:
            vendors = list(latest_eval.vendor_performance.keys())
            
            plt.figure(figsize=(15, 5))
            
            # Subplot for different metrics
            metrics_to_plot = ['bleu', 'rouge', 'exact_match']
            
            for i, metric_name in enumerate(metrics_to_plot, 1):
                plt.subplot(1, 3, i)
                
                scores = []
                for vendor in vendors:
                    vendor_scores = latest_eval.vendor_performance[vendor]
                    score = vendor_scores.get(metric_name, 0.0)
                    scores.append(score)
                    
                plt.bar(vendors, scores)
                plt.title(f'{metric_name.upper()} by Vendor')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.savefig(viz_dir / 'vendor_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def start_step_timing(self):
        """Start timing for a training step."""
        self._step_start_time = time.time()
        
    def should_stop_training(self) -> bool:
        """Check if training should be stopped."""
        return self.should_stop_early
        
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        return self.best_checkpoint_path
        
    def cleanup_old_checkpoints(self, keep_best: int = 5):
        """Clean up old checkpoints, keeping only the best ones."""
        if len(self.checkpoint_history) <= keep_best:
            return
            
        # Sort by loss (ascending)
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['loss'])
        checkpoints_to_remove = sorted_checkpoints[keep_best:]
        
        for checkpoint in checkpoints_to_remove:
            try:
                os.remove(checkpoint['path'])
                logger.info(f"Removed old checkpoint: {checkpoint['path']}")
            except OSError:
                logger.warning(f"Could not remove checkpoint: {checkpoint['path']}")
                
        # Update history
        self.checkpoint_history = sorted_checkpoints[:keep_best]


def create_training_monitor(config: Dict[str, Any], output_dir: str) -> TrainingMonitor:
    """Factory function to create a training monitor."""
    return TrainingMonitor(config, output_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'save_checkpoint_every': 1000,
        'evaluate_every': 500,
        'log_every': 100,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001
    }
    
    monitor = create_training_monitor(config, "/tmp/training_output")
    
    # Simulate training steps
    for epoch in range(3):
        for step in range(1000):
            monitor.start_step_timing()
            
            # Simulate training metrics
            metrics = {
                'total_loss': 2.0 - (epoch * 1000 + step) * 0.001,
                'task_losses': {
                    'generation': 1.0 - (epoch * 1000 + step) * 0.0005,
                    'analysis': 0.8 - (epoch * 1000 + step) * 0.0003,
                    'translation': 1.2 - (epoch * 1000 + step) * 0.0007
                },
                'learning_rate': 0.001,
                'vendor_accuracies': {
                    'cisco': 0.3 + (epoch * 1000 + step) * 0.0003,
                    'juniper': 0.25 + (epoch * 1000 + step) * 0.0003,
                    'nmstate': 0.2 + (epoch * 1000 + step) * 0.0003
                },
                'task_accuracies': {
                    'generation': 0.3 + (epoch * 1000 + step) * 0.0002,
                    'analysis': 0.35 + (epoch * 1000 + step) * 0.0002,
                    'translation': 0.25 + (epoch * 1000 + step) * 0.0002
                },
                'batch_size': 16,
                'samples_processed': (epoch * 1000 + step) * 16
            }
            
            monitor.log_training_step(epoch, step, metrics)
            
            # Simulate evaluation
            if step % 500 == 0:
                eval_metrics = {
                    'bleu_scores': {'overall': 0.5 + step * 0.0001},
                    'rouge_scores': {'overall': 0.4 + step * 0.0001},
                    'exact_match_scores': {'overall': 0.3 + step * 0.0001},
                    'overall_score': 0.4 + step * 0.0001
                }
                monitor.log_evaluation_step(epoch, eval_metrics)
                
            time.sleep(0.001)  # Simulate training time
            
    # Generate final report
    report = monitor.generate_training_report()
    print(f"Training completed. Report: {report['training_summary']}")
    
    # Create visualizations
    monitor.create_visualizations()
    
    print("Training monitoring system ready!")

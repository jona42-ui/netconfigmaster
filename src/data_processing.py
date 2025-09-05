#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Unified Data Processing Pipeline for Multi-Task Network Configuration Learning

This module provides comprehensive data processing capabilities for:
1. Natural Language ↔ Configuration transformations
2. Multi-vendor configuration support (Cisco, Juniper, Nmstate)
3. Task-specific data formatting and validation
4. Advanced data augmentation and preprocessing

Key Features:
- Vendor-aware configuration parsing
- Automatic task type detection
- Data quality validation and cleaning
- Balanced dataset creation
- Format standardization across tasks

Data Flow:
    Raw Data → Validation → Vendor Detection → Task Formatting → Tokenization → Batching

Supported Formats:
- YAML configuration files
- JSON datasets
- Plain text configurations
- Structured vendor-specific formats
"""

import json
import logging
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class DataProcessingConfig:
    """Configuration for data processing pipeline"""
    
    # Input/Output settings
    input_formats: List[str] = field(default_factory=lambda: ['yaml', 'json', 'txt'])
    output_format: str = 'json'
    encoding: str = 'utf-8'
    
    # Validation settings
    validate_syntax: bool = True
    validate_semantics: bool = True
    min_text_length: int = 10
    max_text_length: int = 2048
    
    # Preprocessing settings
    normalize_whitespace: bool = True
    remove_comments: bool = False
    standardize_commands: bool = True
    
    # Vendor detection
    auto_detect_vendor: bool = True
    supported_vendors: List[str] = field(default_factory=lambda: ['cisco', 'juniper', 'nmstate'])
    vendor_confidence_threshold: float = 0.7
    
    # Task processing
    auto_detect_task: bool = True
    task_distribution: Dict[str, float] = field(default_factory=lambda: {
        'generation': 0.4,
        'analysis': 0.3,
        'translation': 0.3
    })
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_ratio: float = 0.2
    paraphrase_variations: int = 3


class VendorDetector:
    """Automatic vendor detection for network configurations"""
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.VendorDetector')
        
        # Define vendor-specific patterns
        self.vendor_patterns = {
            'cisco': {
                'keywords': [
                    'interface', 'ip address', 'router bgp', 'router ospf',
                    'access-list', 'route-map', 'ip route', 'line vty',
                    'hostname', 'enable secret', 'service password-encryption'
                ],
                'syntax': [
                    r'interface\s+(FastEthernet|GigabitEthernet|Serial)',
                    r'ip\s+address\s+\d+\.\d+\.\d+\.\d+\s+\d+\.\d+\.\d+\.\d+',
                    r'router\s+(ospf|bgp)\s+\d+',
                    r'access-list\s+\d+\s+(permit|deny)',
                    r'route-map\s+\w+\s+(permit|deny)\s+\d+'
                ],
                'structure': ['no_braces', 'indentation_based']
            },
            'juniper': {
                'keywords': [
                    'interfaces', 'routing-options', 'protocols', 'security',
                    'policy-options', 'firewall', 'snmp', 'system'
                ],
                'syntax': [
                    r'interfaces\s*\{',
                    r'routing-options\s*\{',
                    r'protocols\s*\{',
                    r'set\s+\w+',
                    r'\w+\s*\{[^}]*\}'
                ],
                'structure': ['braces', 'hierarchical']
            },
            'nmstate': {
                'keywords': [
                    'interfaces', 'routes', 'dns-resolver', 'route-rules',
                    'state', 'type', 'ipv4', 'ipv6'
                ],
                'syntax': [
                    r'interfaces:\s*$',
                    r'routes:\s*$',
                    r'dns-resolver:\s*$',
                    r'-\s+name:\s+\w+',
                    r'type:\s+(ethernet|bond|bridge)'
                ],
                'structure': ['yaml', 'list_based']
            }
        }
    
    def detect_vendor(self, config_text: str) -> Tuple[str, float]:
        """Detect vendor from configuration text with confidence score"""
        
        if not config_text or not config_text.strip():
            return 'unknown', 0.0
        
        vendor_scores = {}
        
        for vendor, patterns in self.vendor_patterns.items():
            score = self._calculate_vendor_score(config_text, patterns)
            vendor_scores[vendor] = score
        
        # Find best match
        if not vendor_scores:
            return 'unknown', 0.0
        
        best_vendor = max(vendor_scores, key=vendor_scores.get)
        confidence = vendor_scores[best_vendor]
        
        # Check confidence threshold
        if confidence < self.config.vendor_confidence_threshold:
            return 'unknown', confidence
        
        self.logger.debug(f"Detected vendor: {best_vendor} (confidence: {confidence:.3f})")
        return best_vendor, confidence
    
    def _calculate_vendor_score(self, config_text: str, patterns: Dict[str, List]) -> float:
        """Calculate vendor match score for given patterns"""
        text_lower = config_text.lower()
        total_score = 0.0
        max_possible_score = 0.0
        
        # Keyword matching (40% weight)
        keyword_score = 0.0
        for keyword in patterns['keywords']:
            if keyword.lower() in text_lower:
                keyword_score += 1
        
        if patterns['keywords']:
            keyword_score = keyword_score / len(patterns['keywords'])
            total_score += keyword_score * 0.4
        max_possible_score += 0.4
        
        # Syntax pattern matching (40% weight)
        syntax_score = 0.0
        for pattern in patterns['syntax']:
            if re.search(pattern, config_text, re.IGNORECASE | re.MULTILINE):
                syntax_score += 1
        
        if patterns['syntax']:
            syntax_score = syntax_score / len(patterns['syntax'])
            total_score += syntax_score * 0.4
        max_possible_score += 0.4
        
        # Structure analysis (20% weight)
        structure_score = self._analyze_structure(config_text, patterns['structure'])
        total_score += structure_score * 0.2
        max_possible_score += 0.2
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0
    
    def _analyze_structure(self, config_text: str, expected_structures: List[str]) -> float:
        """Analyze configuration structure"""
        score = 0.0
        
        for structure in expected_structures:
            if structure == 'braces':
                open_braces = config_text.count('{')
                close_braces = config_text.count('}')
                if open_braces > 0 and abs(open_braces - close_braces) <= 1:
                    score += 1
                    
            elif structure == 'yaml':
                try:
                    yaml.safe_load(config_text)
                    score += 1
                except yaml.YAMLError:
                    pass
                    
            elif structure == 'indentation_based':
                lines = config_text.split('\n')
                indented_lines = sum(1 for line in lines if line.startswith(' ') or line.startswith('\t'))
                if indented_lines > len(lines) * 0.3:  # At least 30% indented
                    score += 1
                    
            elif structure == 'hierarchical':
                # Check for hierarchical keywords
                hierarchical_keywords = ['set', 'edit', 'up', 'top']
                if any(keyword in config_text.lower() for keyword in hierarchical_keywords):
                    score += 1
                    
            elif structure == 'list_based':
                if config_text.count('- ') > 2:  # Multiple list items
                    score += 1
        
        return score / len(expected_structures) if expected_structures else 0.0


class ConfigurationParser:
    """Parse and normalize configurations from different vendors"""
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.ConfigurationParser')
        self.vendor_detector = VendorDetector(config)
    
    def parse_configuration(self, config_text: str, vendor: Optional[str] = None) -> Dict[str, Any]:
        """Parse configuration text and extract structured information"""
        
        if not vendor and self.config.auto_detect_vendor:
            vendor, confidence = self.vendor_detector.detect_vendor(config_text)
        
        # Normalize text
        normalized_text = self._normalize_text(config_text)
        
        # Parse based on vendor
        if vendor == 'cisco':
            parsed_data = self._parse_cisco_config(normalized_text)
        elif vendor == 'juniper':
            parsed_data = self._parse_juniper_config(normalized_text)
        elif vendor == 'nmstate':
            parsed_data = self._parse_nmstate_config(normalized_text)
        else:
            parsed_data = self._parse_generic_config(normalized_text)
        
        # Add metadata
        parsed_data['vendor'] = vendor
        parsed_data['original_text'] = config_text
        parsed_data['normalized_text'] = normalized_text
        
        return parsed_data
    
    def _normalize_text(self, text: str) -> str:
        """Normalize configuration text"""
        if not text:
            return ""
        
        normalized = text
        
        if self.config.normalize_whitespace:
            # Normalize whitespace while preserving structure
            normalized = re.sub(r'[ \t]+', ' ', normalized)  # Multiple spaces/tabs to single space
            normalized = re.sub(r'\n\s*\n', '\n', normalized)  # Multiple newlines to single
            normalized = normalized.strip()
        
        if self.config.remove_comments:
            # Remove comments (vendor-specific)
            normalized = re.sub(r'!\s*.*$', '', normalized, flags=re.MULTILINE)  # Cisco comments
            normalized = re.sub(r'#.*$', '', normalized, flags=re.MULTILINE)    # Unix-style comments
        
        return normalized
    
    def _parse_cisco_config(self, config_text: str) -> Dict[str, Any]:
        """Parse Cisco-style configuration"""
        parsed = {
            'interfaces': [],
            'routing': {},
            'access_lists': [],
            'route_maps': [],
            'other': []
        }
        
        lines = config_text.split('\n')
        current_section = None
        current_interface = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Interface detection
            if line.startswith('interface '):
                current_interface = {
                    'name': line.split()[1],
                    'config': []
                }
                parsed['interfaces'].append(current_interface)
                current_section = 'interface'
            
            # Router protocols
            elif line.startswith('router '):
                protocol = line.split()[1]
                parsed['routing'][protocol] = []
                current_section = f'router_{protocol}'
            
            # Access lists
            elif line.startswith('access-list '):
                parsed['access_lists'].append(line)
                current_section = None
            
            # Route maps
            elif line.startswith('route-map '):
                parsed['route_maps'].append(line)
                current_section = 'route_map'
            
            # Handle nested configuration
            elif current_section == 'interface' and current_interface:
                current_interface['config'].append(line)
            
            elif current_section and current_section.startswith('router_'):
                protocol = current_section.split('_')[1]
                parsed['routing'][protocol].append(line)
            
            else:
                parsed['other'].append(line)
        
        return parsed
    
    def _parse_juniper_config(self, config_text: str) -> Dict[str, Any]:
        """Parse Juniper-style configuration"""
        parsed = {
            'interfaces': {},
            'routing_options': {},
            'protocols': {},
            'security': {},
            'other': {}
        }
        
        # Simple hierarchical parsing
        try:
            # Try to parse as a structured format
            sections = self._extract_juniper_sections(config_text)
            parsed.update(sections)
        except Exception as e:
            self.logger.warning(f"Failed to parse Juniper config structure: {e}")
            parsed['raw'] = config_text
        
        return parsed
    
    def _parse_nmstate_config(self, config_text: str) -> Dict[str, Any]:
        """Parse Nmstate YAML configuration"""
        try:
            parsed = yaml.safe_load(config_text)
            if not isinstance(parsed, dict):
                parsed = {'raw': config_text}
        except yaml.YAMLError as e:
            self.logger.warning(f"Failed to parse Nmstate YAML: {e}")
            parsed = {'raw': config_text, 'error': str(e)}
        
        return parsed
    
    def _parse_generic_config(self, config_text: str) -> Dict[str, Any]:
        """Parse unknown/generic configuration format"""
        return {
            'raw': config_text,
            'lines': config_text.split('\n'),
            'line_count': len(config_text.split('\n')),
            'character_count': len(config_text)
        }
    
    def _extract_juniper_sections(self, config_text: str) -> Dict[str, Any]:
        """Extract sections from Juniper configuration"""
        sections = {}
        
        # Simple regex-based extraction
        patterns = {
            'interfaces': r'interfaces\s*\{([^}]*)\}',
            'routing_options': r'routing-options\s*\{([^}]*)\}',
            'protocols': r'protocols\s*\{([^}]*)\}',
            'security': r'security\s*\{([^}]*)\}'
        }
        
        for section_name, pattern in patterns.items():
            matches = re.finditer(pattern, config_text, re.MULTILINE | re.DOTALL)
            section_content = []
            for match in matches:
                section_content.append(match.group(1).strip())
            
            if section_content:
                sections[section_name] = section_content
        
        return sections


class TaskDataProcessor:
    """Process data for specific tasks (generation, analysis, translation)"""
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.TaskDataProcessor')
        self.parser = ConfigurationParser(config)
    
    def process_generation_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process data for configuration generation task"""
        processed = []
        
        for item in data:
            try:
                # Extract natural language question and configuration answer
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                if not question or not answer:
                    self.logger.warning("Missing question or answer, skipping item")
                    continue
                
                # Parse configuration to detect vendor and validate
                parsed_config = self.parser.parse_configuration(answer)
                vendor = parsed_config.get('vendor', 'unknown')
                
                # Validate lengths
                if (len(question) < self.config.min_text_length or 
                    len(question) > self.config.max_text_length or
                    len(answer) < self.config.min_text_length or
                    len(answer) > self.config.max_text_length):
                    continue
                
                processed_item = {
                    'task_type': 'generation',
                    'input_text': question,
                    'target_text': answer,
                    'vendor': vendor,
                    'source_type': 'natural_language',
                    'target_type': 'configuration',
                    'parsed_config': parsed_config,
                    'metadata': item.get('metadata', {})
                }
                
                processed.append(processed_item)
                
            except Exception as e:
                self.logger.error(f"Error processing generation item: {e}")
                continue
        
        self.logger.info(f"Processed {len(processed)} generation samples")
        return processed
    
    def process_analysis_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process data for configuration analysis task"""
        processed = []
        
        for item in data:
            try:
                # For analysis, we reverse the generation task
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                if not question or not answer:
                    continue
                
                # Parse configuration (now the input)
                parsed_config = self.parser.parse_configuration(answer)
                vendor = parsed_config.get('vendor', 'unknown')
                
                # Validate lengths
                if (len(question) < self.config.min_text_length or 
                    len(question) > self.config.max_text_length or
                    len(answer) < self.config.min_text_length or
                    len(answer) > self.config.max_text_length):
                    continue
                
                processed_item = {
                    'task_type': 'analysis',
                    'input_text': answer,  # Configuration is input
                    'target_text': question,  # Natural language is target
                    'vendor': vendor,
                    'source_type': 'configuration',
                    'target_type': 'natural_language',
                    'parsed_config': parsed_config,
                    'metadata': item.get('metadata', {})
                }
                
                processed.append(processed_item)
                
            except Exception as e:
                self.logger.error(f"Error processing analysis item: {e}")
                continue
        
        self.logger.info(f"Processed {len(processed)} analysis samples")
        return processed
    
    def process_translation_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process data for configuration translation task"""
        processed = []
        
        for item in data:
            try:
                source_config = item.get('source_config', '')
                target_config = item.get('target_config', '')
                
                if not source_config or not target_config:
                    self.logger.warning("Missing source or target config, skipping item")
                    continue
                
                # Parse both configurations
                source_parsed = self.parser.parse_configuration(source_config)
                target_parsed = self.parser.parse_configuration(target_config)
                
                source_vendor = source_parsed.get('vendor', 'unknown')
                target_vendor = target_parsed.get('vendor', 'unknown')
                
                # Validate lengths
                if (len(source_config) < self.config.min_text_length or 
                    len(source_config) > self.config.max_text_length or
                    len(target_config) < self.config.min_text_length or
                    len(target_config) > self.config.max_text_length):
                    continue
                
                processed_item = {
                    'task_type': 'translation',
                    'input_text': source_config,
                    'target_text': target_config,
                    'source_vendor': source_vendor,
                    'target_vendor': target_vendor,
                    'source_type': 'configuration',
                    'target_type': 'configuration',
                    'source_parsed': source_parsed,
                    'target_parsed': target_parsed,
                    'metadata': item.get('metadata', {})
                }
                
                processed.append(processed_item)
                
            except Exception as e:
                self.logger.error(f"Error processing translation item: {e}")
                continue
        
        self.logger.info(f"Processed {len(processed)} translation samples")
        return processed
    
    def detect_task_type(self, item: Dict[str, Any]) -> str:
        """Automatically detect task type from data item"""
        
        # Check explicit task type
        if 'task_type' in item:
            return item['task_type']
        
        # Infer from data structure
        if 'source_config' in item and 'target_config' in item:
            return 'translation'
        elif 'question' in item and 'answer' in item:
            # Check if answer looks like configuration
            answer = item['answer']
            parsed = self.parser.parse_configuration(answer)
            if parsed.get('vendor', 'unknown') != 'unknown':
                return 'generation'  # NL -> Config
            else:
                return 'analysis'    # Config -> NL
        
        return 'unknown'


class UnifiedDataLoader:
    """Unified data loader for all supported formats"""
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.UnifiedDataLoader')
        self.task_processor = TaskDataProcessor(config)
    
    def load_dataset(self, file_path: Union[str, Path], task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load dataset from file with automatic format detection"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Detect format
        file_format = self._detect_format(file_path)
        self.logger.info(f"Loading {file_format} dataset from {file_path}")
        
        # Load raw data
        if file_format == 'yaml':
            raw_data = self._load_yaml(file_path)
        elif file_format == 'json':
            raw_data = self._load_json(file_path)
        elif file_format == 'txt':
            raw_data = self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        # Process data based on task type
        if task_type == 'generation':
            processed_data = self.task_processor.process_generation_data(raw_data)
        elif task_type == 'analysis':
            processed_data = self.task_processor.process_analysis_data(raw_data)
        elif task_type == 'translation':
            processed_data = self.task_processor.process_translation_data(raw_data)
        elif task_type is None and self.config.auto_detect_task:
            processed_data = self._process_mixed_tasks(raw_data)
        else:
            # Generic processing
            processed_data = raw_data
        
        self.logger.info(f"Loaded {len(processed_data)} samples")
        return processed_data
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension and content"""
        extension = file_path.suffix.lower()
        
        if extension in ['.yaml', '.yml']:
            return 'yaml'
        elif extension == '.json':
            return 'json'
        elif extension in ['.txt', '.text']:
            return 'txt'
        else:
            # Try to detect from content
            try:
                with open(file_path, 'r', encoding=self.config.encoding) as f:
                    content = f.read(1024)  # Read first 1KB
                
                # Try JSON first
                try:
                    json.loads(content[:100] + '}' if not content.strip().endswith('}') else content[:100])
                    return 'json'
                except json.JSONDecodeError:
                    pass
                
                # Try YAML
                try:
                    yaml.safe_load(content)
                    return 'yaml'
                except yaml.YAMLError:
                    pass
                
                # Default to text
                return 'txt'
                
            except Exception:
                return 'txt'
    
    def _load_yaml(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load YAML dataset"""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If single dict, wrap in list
            return [data]
        else:
            raise ValueError(f"Unexpected YAML structure in {file_path}")
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON dataset"""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unexpected JSON structure in {file_path}")
    
    def _load_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text dataset"""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            lines = f.readlines()
        
        # Simple text processing - each line is a sample
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                data.append({
                    'text': line,
                    'index': i,
                    'source_file': str(file_path)
                })
        
        return data
    
    def _process_mixed_tasks(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process data with mixed task types"""
        processed_data = []
        task_counts = Counter()
        
        for item in raw_data:
            task_type = self.task_processor.detect_task_type(item)
            task_counts[task_type] += 1
            
            if task_type == 'generation':
                processed_items = self.task_processor.process_generation_data([item])
            elif task_type == 'analysis':
                processed_items = self.task_processor.process_analysis_data([item])
            elif task_type == 'translation':
                processed_items = self.task_processor.process_translation_data([item])
            else:
                processed_items = [item]  # Keep as-is
            
            processed_data.extend(processed_items)
        
        self.logger.info(f"Task distribution: {dict(task_counts)}")
        return processed_data
    
    def save_dataset(self, data: List[Dict[str, Any]], output_path: Union[str, Path], format_type: Optional[str] = None) -> None:
        """Save processed dataset"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type is None:
            format_type = self.config.output_format
        
        if format_type == 'json':
            with open(output_path, 'w', encoding=self.config.encoding) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format_type in ['yaml', 'yml']:
            with open(output_path, 'w', encoding=self.config.encoding) as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
        
        self.logger.info(f"Saved {len(data)} samples to {output_path}")


# Utility functions
def create_balanced_dataset(datasets: Dict[str, List[Dict[str, Any]]], 
                          target_size: int, 
                          task_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Create balanced dataset across tasks"""
    
    if task_weights is None:
        task_weights = {task: 1.0 for task in datasets.keys()}
    
    # Normalize weights
    total_weight = sum(task_weights.values())
    normalized_weights = {task: weight / total_weight for task, weight in task_weights.items()}
    
    balanced_data = []
    
    for task_name, task_data in datasets.items():
        if task_name not in normalized_weights:
            continue
        
        # Calculate target size for this task
        task_target = int(target_size * normalized_weights[task_name])
        
        if len(task_data) >= task_target:
            # Subsample
            import random
            selected_data = random.sample(task_data, task_target)
        else:
            # Oversample with replacement
            import random
            selected_data = random.choices(task_data, k=task_target)
        
        balanced_data.extend(selected_data)
    
    # Shuffle final dataset
    import random
    random.shuffle(balanced_data)
    
    return balanced_data


def validate_dataset_quality(data: List[Dict[str, Any]], config: DataProcessingConfig) -> Dict[str, Any]:
    """Validate dataset quality and provide statistics"""
    
    stats = {
        'total_samples': len(data),
        'task_distribution': Counter(),
        'vendor_distribution': Counter(),
        'text_length_stats': {
            'input_lengths': [],
            'target_lengths': []
        },
        'quality_issues': []
    }
    
    for item in data:
        # Task distribution
        task_type = item.get('task_type', 'unknown')
        stats['task_distribution'][task_type] += 1
        
        # Vendor distribution
        vendor = item.get('vendor', item.get('source_vendor', 'unknown'))
        stats['vendor_distribution'][vendor] += 1
        
        # Text lengths
        input_text = item.get('input_text', '')
        target_text = item.get('target_text', '')
        
        stats['text_length_stats']['input_lengths'].append(len(input_text))
        stats['text_length_stats']['target_lengths'].append(len(target_text))
        
        # Quality checks
        if len(input_text) < config.min_text_length:
            stats['quality_issues'].append(f"Input text too short: {len(input_text)} chars")
        
        if len(target_text) < config.min_text_length:
            stats['quality_issues'].append(f"Target text too short: {len(target_text)} chars")
    
    # Calculate statistics
    if stats['text_length_stats']['input_lengths']:
        import numpy as np
        stats['text_length_stats']['input_mean'] = np.mean(stats['text_length_stats']['input_lengths'])
        stats['text_length_stats']['input_std'] = np.std(stats['text_length_stats']['input_lengths'])
        stats['text_length_stats']['target_mean'] = np.mean(stats['text_length_stats']['target_lengths'])
        stats['text_length_stats']['target_std'] = np.std(stats['text_length_stats']['target_lengths'])
    
    return stats

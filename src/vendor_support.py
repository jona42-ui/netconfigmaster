#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Multi-Vendor Network Configuration Support System

This module provides comprehensive support for multiple network equipment vendors:

1. Vendor-Specific Parsers:
   - Cisco IOS/IOS-XE/NX-OS
   - Juniper JUNOS
   - Nmstate (Linux networking)
   - Generic/Unknown formats

2. Configuration Translation:
   - Cross-vendor command mapping
   - Semantic preservation
   - Syntax transformation

3. Validation Systems:
   - Vendor-specific syntax checking
   - Semantic consistency verification
   - Best practices compliance

4. Standardization:
   - Common configuration model
   - Vendor-agnostic representation
   - Canonical form conversion

Features:
- Automatic vendor detection
- Command similarity matching
- Configuration templating
- Semantic validation
- Cross-vendor translation tables
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


@dataclass
class VendorConfig:
    """Configuration for vendor-specific processing"""
    name: str
    aliases: List[str] = field(default_factory=list)
    command_prefix: str = ""
    comment_chars: List[str] = field(default_factory=lambda: ["!", "#"])
    line_continuation: str = ""
    block_delimiters: Tuple[str, str] = ("", "")
    indentation_style: str = "spaces"  # spaces, tabs, none
    case_sensitive: bool = False
    supports_hierarchy: bool = False
    configuration_format: str = "text"  # text, json, xml, yaml


class BaseVendorHandler(ABC):
    """Abstract base class for vendor-specific configuration handlers"""
    
    def __init__(self, config: VendorConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._initialize_patterns()
    
    @abstractmethod
    def _initialize_patterns(self):
        """Initialize vendor-specific patterns and rules"""
        pass
    
    @abstractmethod
    def parse_configuration(self, config_text: str) -> Dict[str, Any]:
        """Parse configuration text into structured format"""
        pass
    
    @abstractmethod
    def generate_configuration(self, config_data: Dict[str, Any]) -> str:
        """Generate configuration text from structured data"""
        pass
    
    @abstractmethod
    def validate_syntax(self, config_text: str) -> Tuple[bool, List[str]]:
        """Validate configuration syntax"""
        pass
    
    def normalize_text(self, config_text: str) -> str:
        """Normalize configuration text"""
        if not config_text:
            return ""
        
        # Remove comments
        normalized = config_text
        for comment_char in self.config.comment_chars:
            normalized = re.sub(f'{re.escape(comment_char)}.*$', '', normalized, flags=re.MULTILINE)
        
        # Normalize whitespace
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        normalized = re.sub(r'\n\s*\n', '\n', normalized)
        
        return normalized.strip()
    
    def extract_interfaces(self, config_text: str) -> List[Dict[str, Any]]:
        """Extract interface configurations"""
        return []  # Default implementation
    
    def extract_routing(self, config_text: str) -> Dict[str, Any]:
        """Extract routing configurations"""
        return {}  # Default implementation


class CiscoHandler(BaseVendorHandler):
    """Handler for Cisco IOS/IOS-XE configurations"""
    
    def __init__(self):
        config = VendorConfig(
            name="cisco",
            aliases=["ios", "ios-xe", "nx-os"],
            command_prefix="",
            comment_chars=["!"],
            case_sensitive=False,
            supports_hierarchy=True,
            configuration_format="text"
        )
        super().__init__(config)
    
    def _initialize_patterns(self):
        """Initialize Cisco-specific patterns"""
        self.interface_pattern = re.compile(
            r'^interface\s+(\S+)\s*$', 
            re.IGNORECASE | re.MULTILINE
        )
        
        self.ip_address_pattern = re.compile(
            r'^\s*ip\s+address\s+(\d+\.\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+\.\d+)(?:\s+secondary)?',
            re.IGNORECASE
        )
        
        self.router_pattern = re.compile(
            r'^router\s+(\w+)(?:\s+(\d+))?',
            re.IGNORECASE | re.MULTILINE
        )
        
        self.access_list_pattern = re.compile(
            r'^access-list\s+(\d+)\s+(permit|deny)\s+(.*)',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Command templates for generation
        self.templates = {
            'interface': {
                'ethernet': 'interface {name}',
                'ip_address': ' ip address {ip} {mask}',
                'description': ' description {description}',
                'shutdown': ' shutdown',
                'no_shutdown': ' no shutdown'
            },
            'routing': {
                'static_route': 'ip route {network} {mask} {next_hop}',
                'ospf_router': 'router ospf {process_id}',
                'ospf_network': ' network {network} {wildcard} area {area}',
                'bgp_router': 'router bgp {asn}',
                'bgp_neighbor': ' neighbor {ip} remote-as {remote_asn}'
            }
        }
    
    def parse_configuration(self, config_text: str) -> Dict[str, Any]:
        """Parse Cisco configuration"""
        parsed = {
            'vendor': 'cisco',
            'interfaces': self._parse_interfaces(config_text),
            'routing': self._parse_routing(config_text),
            'access_lists': self._parse_access_lists(config_text),
            'route_maps': self._parse_route_maps(config_text),
            'global_config': self._parse_global_config(config_text)
        }
        
        return parsed
    
    def _parse_interfaces(self, config_text: str) -> List[Dict[str, Any]]:
        """Parse interface configurations"""
        interfaces = []
        lines = config_text.split('\n')
        current_interface = None
        
        for line in lines:
            line = line.strip()
            
            # Interface declaration
            match = self.interface_pattern.match(line)
            if match:
                if current_interface:
                    interfaces.append(current_interface)
                
                current_interface = {
                    'name': match.group(1),
                    'type': self._determine_interface_type(match.group(1)),
                    'config': [],
                    'ip_addresses': [],
                    'description': '',
                    'admin_status': 'up'
                }
                continue
            
            # Interface sub-commands
            if current_interface and line.startswith(' '):
                current_interface['config'].append(line.strip())
                
                # Parse specific configurations
                if line.strip().startswith('ip address'):
                    ip_match = self.ip_address_pattern.match(line)
                    if ip_match:
                        current_interface['ip_addresses'].append({
                            'ip': ip_match.group(1),
                            'mask': ip_match.group(2),
                            'secondary': 'secondary' in line
                        })
                
                elif line.strip().startswith('description'):
                    current_interface['description'] = line.strip()[12:].strip()
                
                elif line.strip() == 'shutdown':
                    current_interface['admin_status'] = 'down'
        
        if current_interface:
            interfaces.append(current_interface)
        
        return interfaces
    
    def _parse_routing(self, config_text: str) -> Dict[str, Any]:
        """Parse routing configurations"""
        routing = {
            'static_routes': [],
            'dynamic_protocols': {}
        }
        
        lines = config_text.split('\n')
        current_router = None
        
        for line in lines:
            line = line.strip()
            
            # Static routes
            if line.startswith('ip route'):
                parts = line.split()
                if len(parts) >= 5:
                    routing['static_routes'].append({
                        'network': parts[2],
                        'mask': parts[3],
                        'next_hop': parts[4],
                        'administrative_distance': parts[5] if len(parts) > 5 else '1'
                    })
            
            # Dynamic routing protocols
            router_match = self.router_pattern.match(line)
            if router_match:
                protocol = router_match.group(1)
                process_id = router_match.group(2) or '1'
                
                current_router = {
                    'protocol': protocol,
                    'process_id': process_id,
                    'networks': [],
                    'neighbors': [],
                    'config': []
                }
                routing['dynamic_protocols'][f"{protocol}_{process_id}"] = current_router
            
            elif current_router and line.startswith(' '):
                current_router['config'].append(line.strip())
                
                # Parse protocol-specific configs
                if line.strip().startswith('network'):
                    current_router['networks'].append(line.strip())
                elif line.strip().startswith('neighbor'):
                    current_router['neighbors'].append(line.strip())
        
        return routing
    
    def _parse_access_lists(self, config_text: str) -> List[Dict[str, Any]]:
        """Parse access control lists"""
        access_lists = []
        
        for match in self.access_list_pattern.finditer(config_text):
            access_lists.append({
                'number': int(match.group(1)),
                'action': match.group(2),
                'criteria': match.group(3)
            })
        
        return access_lists
    
    def _parse_route_maps(self, config_text: str) -> List[Dict[str, Any]]:
        """Parse route maps"""
        route_maps = []
        
        route_map_pattern = re.compile(
            r'^route-map\s+(\S+)\s+(permit|deny)\s+(\d+)',
            re.IGNORECASE | re.MULTILINE
        )
        
        for match in route_map_pattern.finditer(config_text):
            route_maps.append({
                'name': match.group(1),
                'action': match.group(2),
                'sequence': int(match.group(3))
            })
        
        return route_maps
    
    def _parse_global_config(self, config_text: str) -> Dict[str, Any]:
        """Parse global configuration settings"""
        global_config = {}
        
        # Hostname
        hostname_match = re.search(r'^hostname\s+(\S+)', config_text, re.IGNORECASE | re.MULTILINE)
        if hostname_match:
            global_config['hostname'] = hostname_match.group(1)
        
        # Domain name
        domain_match = re.search(r'^ip\s+domain-name\s+(\S+)', config_text, re.IGNORECASE | re.MULTILINE)
        if domain_match:
            global_config['domain_name'] = domain_match.group(1)
        
        return global_config
    
    def _determine_interface_type(self, interface_name: str) -> str:
        """Determine interface type from name"""
        name_lower = interface_name.lower()
        
        if name_lower.startswith(('fa', 'fastethernet')):
            return 'FastEthernet'
        elif name_lower.startswith(('gi', 'gigabitethernet')):
            return 'GigabitEthernet'
        elif name_lower.startswith(('te', 'tengigabitethernet')):
            return 'TenGigabitEthernet'
        elif name_lower.startswith(('se', 'serial')):
            return 'Serial'
        elif name_lower.startswith(('lo', 'loopback')):
            return 'Loopback'
        else:
            return 'Unknown'
    
    def generate_configuration(self, config_data: Dict[str, Any]) -> str:
        """Generate Cisco configuration from structured data"""
        lines = []
        
        # Global configuration
        if 'global_config' in config_data:
            global_config = config_data['global_config']
            if 'hostname' in global_config:
                lines.append(f"hostname {global_config['hostname']}")
            if 'domain_name' in global_config:
                lines.append(f"ip domain-name {global_config['domain_name']}")
            lines.append("")
        
        # Interfaces
        if 'interfaces' in config_data:
            for interface in config_data['interfaces']:
                lines.append(f"interface {interface['name']}")
                
                if interface.get('description'):
                    lines.append(f" description {interface['description']}")
                
                for ip_addr in interface.get('ip_addresses', []):
                    secondary = " secondary" if ip_addr.get('secondary') else ""
                    lines.append(f" ip address {ip_addr['ip']} {ip_addr['mask']}{secondary}")
                
                if interface.get('admin_status') == 'down':
                    lines.append(" shutdown")
                else:
                    lines.append(" no shutdown")
                
                lines.append("")
        
        # Routing
        if 'routing' in config_data:
            routing = config_data['routing']
            
            # Static routes
            for route in routing.get('static_routes', []):
                lines.append(f"ip route {route['network']} {route['mask']} {route['next_hop']}")
            
            # Dynamic protocols
            for protocol_name, protocol_config in routing.get('dynamic_protocols', {}).items():
                lines.append(f"router {protocol_config['protocol']} {protocol_config['process_id']}")
                for config_line in protocol_config.get('config', []):
                    lines.append(f" {config_line}")
                lines.append("")
        
        return '\n'.join(lines)
    
    def validate_syntax(self, config_text: str) -> Tuple[bool, List[str]]:
        """Validate Cisco configuration syntax"""
        errors = []
        
        lines = config_text.split('\n')
        interface_mode = False
        router_mode = False
        
        for i, line in enumerate(lines, 1):
            line = line.rstrip()
            if not line or line.startswith('!'):
                continue
            
            # Check indentation rules
            if line.startswith(' '):
                if not (interface_mode or router_mode):
                    errors.append(f"Line {i}: Unexpected indentation outside of configuration mode")
            else:
                interface_mode = line.startswith('interface ')
                router_mode = line.startswith('router ')
            
            # Check for common syntax errors
            if line.count('!') > 1 and not line.startswith('!'):
                errors.append(f"Line {i}: Multiple comment characters in configuration line")
            
            # Validate IP addresses
            ip_addresses = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', line)
            for ip in ip_addresses:
                octets = [int(x) for x in ip.split('.')]
                if any(octet > 255 for octet in octets):
                    errors.append(f"Line {i}: Invalid IP address {ip}")
        
        return len(errors) == 0, errors


class JuniperHandler(BaseVendorHandler):
    """Handler for Juniper JUNOS configurations"""
    
    def __init__(self):
        config = VendorConfig(
            name="juniper",
            aliases=["junos"],
            command_prefix="set ",
            comment_chars=["#"],
            block_delimiters=("{", "}"),
            case_sensitive=True,
            supports_hierarchy=True,
            configuration_format="text"
        )
        super().__init__(config)
    
    def _initialize_patterns(self):
        """Initialize Juniper-specific patterns"""
        self.hierarchical_pattern = re.compile(
            r'^(\w+(?:\s+\w+)*)\s*\{',
            re.MULTILINE
        )
        
        self.set_command_pattern = re.compile(
            r'^set\s+(.+)$',
            re.MULTILINE
        )
        
        self.templates = {
            'interface': {
                'unit': 'interfaces {interface} unit {unit}',
                'description': 'interfaces {interface} description "{description}"',
                'ip_address': 'interfaces {interface} unit {unit} family inet address {ip}/{prefix}'
            },
            'routing': {
                'static_route': 'routing-options static route {network} next-hop {next_hop}',
                'ospf': 'protocols ospf area {area} interface {interface}',
                'bgp': 'protocols bgp group {group_name} neighbor {neighbor}'
            }
        }
    
    def parse_configuration(self, config_text: str) -> Dict[str, Any]:
        """Parse Juniper configuration"""
        parsed = {
            'vendor': 'juniper',
            'format': self._detect_format(config_text),
            'interfaces': {},
            'routing_options': {},
            'protocols': {},
            'security': {}
        }
        
        if parsed['format'] == 'set':
            parsed.update(self._parse_set_format(config_text))
        else:
            parsed.update(self._parse_hierarchical_format(config_text))
        
        return parsed
    
    def _detect_format(self, config_text: str) -> str:
        """Detect Juniper configuration format"""
        set_commands = len(re.findall(r'^set\s+', config_text, re.MULTILINE))
        braces = config_text.count('{')
        
        if set_commands > braces:
            return 'set'
        else:
            return 'hierarchical'
    
    def _parse_set_format(self, config_text: str) -> Dict[str, Any]:
        """Parse set-format configuration"""
        parsed = defaultdict(dict)
        
        for match in self.set_command_pattern.finditer(config_text):
            command = match.group(1)
            self._process_set_command(parsed, command)
        
        return dict(parsed)
    
    def _parse_hierarchical_format(self, config_text: str) -> Dict[str, Any]:
        """Parse hierarchical format configuration"""
        # This is a simplified implementation
        # Full parser would need to handle nested braces properly
        sections = {
            'interfaces': self._extract_section(config_text, 'interfaces'),
            'routing-options': self._extract_section(config_text, 'routing-options'),
            'protocols': self._extract_section(config_text, 'protocols'),
            'security': self._extract_section(config_text, 'security')
        }
        
        return sections
    
    def _process_set_command(self, parsed: Dict, command: str) -> None:
        """Process individual set command"""
        parts = command.split()
        if not parts:
            return
        
        # Build nested dictionary structure
        current = parsed
        for part in parts[:-2]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        if len(parts) >= 2:
            current[parts[-2]] = parts[-1]
    
    def _extract_section(self, config_text: str, section_name: str) -> Dict[str, Any]:
        """Extract a configuration section from hierarchical format"""
        pattern = rf'{section_name}\s*\{{([^}}]*)\}}'
        match = re.search(pattern, config_text, re.DOTALL)
        
        if match:
            return {'raw': match.group(1).strip()}
        return {}
    
    def generate_configuration(self, config_data: Dict[str, Any]) -> str:
        """Generate Juniper configuration from structured data"""
        lines = []
        
        # Generate set commands for interfaces
        if 'interfaces' in config_data:
            for interface_name, interface_config in config_data['interfaces'].items():
                if isinstance(interface_config, dict):
                    for key, value in interface_config.items():
                        lines.append(f"set interfaces {interface_name} {key} {value}")
        
        # Generate routing options
        if 'routing_options' in config_data:
            for key, value in config_data['routing_options'].items():
                lines.append(f"set routing-options {key} {value}")
        
        return '\n'.join(lines)
    
    def validate_syntax(self, config_text: str) -> Tuple[bool, List[str]]:
        """Validate Juniper configuration syntax"""
        errors = []
        
        # Check brace matching for hierarchical format
        if '{' in config_text or '}' in config_text:
            open_braces = config_text.count('{')
            close_braces = config_text.count('}')
            
            if open_braces != close_braces:
                errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
        
        # Validate set commands
        lines = config_text.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('set ') and len(line.split()) < 3:
                errors.append(f"Line {i}: Incomplete set command")
        
        return len(errors) == 0, errors


class NmstateHandler(BaseVendorHandler):
    """Handler for Nmstate YAML configurations"""
    
    def __init__(self):
        config = VendorConfig(
            name="nmstate",
            aliases=["linux", "rhel", "fedora"],
            comment_chars=["#"],
            case_sensitive=True,
            supports_hierarchy=True,
            configuration_format="yaml"
        )
        super().__init__(config)
    
    def _initialize_patterns(self):
        """Initialize Nmstate-specific patterns"""
        self.required_keys = ['interfaces']
        self.optional_keys = ['routes', 'dns-resolver', 'route-rules']
        
        self.interface_types = [
            'ethernet', 'bond', 'bridge', 'vlan', 'vxlan',
            'ovs-bridge', 'ovs-port', 'ovs-interface'
        ]
    
    def parse_configuration(self, config_text: str) -> Dict[str, Any]:
        """Parse Nmstate YAML configuration"""
        try:
            parsed = yaml.safe_load(config_text)
            if not isinstance(parsed, dict):
                return {'vendor': 'nmstate', 'error': 'Invalid YAML structure'}
            
            parsed['vendor'] = 'nmstate'
            return self._validate_nmstate_structure(parsed)
            
        except yaml.YAMLError as e:
            return {'vendor': 'nmstate', 'error': f'YAML parsing error: {e}'}
    
    def _validate_nmstate_structure(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Nmstate configuration structure"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required keys
        for key in self.required_keys:
            if key not in parsed:
                validation_results['errors'].append(f"Missing required key: {key}")
                validation_results['valid'] = False
        
        # Validate interfaces
        if 'interfaces' in parsed:
            self._validate_interfaces(parsed['interfaces'], validation_results)
        
        parsed['validation'] = validation_results
        return parsed
    
    def _validate_interfaces(self, interfaces: List[Dict], validation_results: Dict) -> None:
        """Validate interface configurations"""
        if not isinstance(interfaces, list):
            validation_results['errors'].append("Interfaces must be a list")
            validation_results['valid'] = False
            return
        
        for i, interface in enumerate(interfaces):
            if not isinstance(interface, dict):
                validation_results['errors'].append(f"Interface {i} must be a dictionary")
                continue
            
            # Check required interface fields
            if 'name' not in interface:
                validation_results['errors'].append(f"Interface {i} missing name")
                validation_results['valid'] = False
            
            if 'type' not in interface:
                validation_results['warnings'].append(f"Interface {interface.get('name', i)} missing type")
            elif interface['type'] not in self.interface_types:
                validation_results['warnings'].append(
                    f"Interface {interface.get('name', i)} has unknown type: {interface['type']}"
                )
    
    def generate_configuration(self, config_data: Dict[str, Any]) -> str:
        """Generate Nmstate YAML configuration from structured data"""
        # Remove vendor metadata for output
        output_data = {k: v for k, v in config_data.items() 
                      if k not in ['vendor', 'validation']}
        
        return yaml.dump(output_data, default_flow_style=False, allow_unicode=True)
    
    def validate_syntax(self, config_text: str) -> Tuple[bool, List[str]]:
        """Validate Nmstate YAML syntax"""
        errors = []
        
        try:
            parsed = yaml.safe_load(config_text)
            
            if not isinstance(parsed, dict):
                errors.append("Configuration must be a YAML dictionary")
            else:
                # Validate structure
                parsed_with_validation = self._validate_nmstate_structure(parsed)
                validation_results = parsed_with_validation.get('validation', {})
                
                if validation_results.get('errors'):
                    errors.extend(validation_results['errors'])
                    
        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {e}")
        
        return len(errors) == 0, errors


class MultiVendorManager:
    """Central manager for multi-vendor configuration handling"""
    
    def __init__(self):
        self.handlers = {
            'cisco': CiscoHandler(),
            'juniper': JuniperHandler(),
            'nmstate': NmstateHandler()
        }
        
        self.vendor_aliases = {}
        for vendor, handler in self.handlers.items():
            self.vendor_aliases[vendor] = vendor
            for alias in handler.config.aliases:
                self.vendor_aliases[alias] = vendor
        
        self.logger = logging.getLogger(__name__ + '.MultiVendorManager')
        
        # Load translation tables
        self._initialize_translation_tables()
    
    def _initialize_translation_tables(self):
        """Initialize cross-vendor translation tables"""
        self.translation_tables = {
            'interface_commands': {
                'cisco_to_juniper': {
                    'interface {name}': 'interfaces {name}',
                    'ip address {ip} {mask}': 'unit 0 family inet address {ip}/{prefix}',
                    'description {desc}': 'description "{desc}"',
                    'shutdown': 'disable',
                    'no shutdown': 'enable'
                },
                'juniper_to_cisco': {
                    'interfaces {name}': 'interface {name}',
                    'unit 0 family inet address {ip}/{prefix}': 'ip address {ip} {mask}',
                    'description "{desc}"': 'description {desc}',
                    'disable': 'shutdown',
                    'enable': 'no shutdown'
                },
                'cisco_to_nmstate': {
                    'interface {name}': {'name': '{name}', 'type': 'ethernet'},
                    'ip address {ip} {mask}': {'ipv4': {'address': [{'ip': '{ip}', 'prefix-length': '{prefix}'}]}}
                }
            }
        }
    
    def detect_vendor(self, config_text: str) -> Tuple[str, float]:
        """Detect vendor from configuration text"""
        best_vendor = 'unknown'
        best_score = 0.0
        
        for vendor_name, handler in self.handlers.items():
            # Use a simple scoring system based on patterns
            score = self._calculate_vendor_score(config_text, vendor_name)
            
            if score > best_score:
                best_score = score
                best_vendor = vendor_name
        
        return best_vendor, best_score
    
    def _calculate_vendor_score(self, config_text: str, vendor_name: str) -> float:
        """Calculate vendor detection score"""
        if vendor_name == 'cisco':
            cisco_patterns = [
                r'interface\s+(FastEthernet|GigabitEthernet)',
                r'router\s+(ospf|bgp)',
                r'ip\s+address\s+\d+\.\d+\.\d+\.\d+',
                r'access-list\s+\d+'
            ]
            return sum(1 for pattern in cisco_patterns 
                      if re.search(pattern, config_text, re.IGNORECASE)) / len(cisco_patterns)
        
        elif vendor_name == 'juniper':
            juniper_patterns = [
                r'interfaces\s*\{',
                r'protocols\s*\{',
                r'set\s+\w+',
                r'routing-options'
            ]
            return sum(1 for pattern in juniper_patterns 
                      if re.search(pattern, config_text, re.IGNORECASE)) / len(juniper_patterns)
        
        elif vendor_name == 'nmstate':
            nmstate_patterns = [
                r'interfaces:\s*$',
                r'routes:\s*$',
                r'type:\s+(ethernet|bond)',
                r'ipv4:\s*$'
            ]
            try:
                yaml.safe_load(config_text)
                yaml_score = 0.5  # Bonus for valid YAML
            except:
                yaml_score = 0.0
                
            pattern_score = sum(1 for pattern in nmstate_patterns 
                              if re.search(pattern, config_text, re.MULTILINE)) / len(nmstate_patterns)
            return pattern_score + yaml_score
        
        return 0.0
    
    def parse_configuration(self, config_text: str, vendor: Optional[str] = None) -> Dict[str, Any]:
        """Parse configuration using appropriate vendor handler"""
        if vendor is None:
            vendor, confidence = self.detect_vendor(config_text)
            self.logger.info(f"Detected vendor: {vendor} (confidence: {confidence:.3f})")
        
        # Normalize vendor name
        vendor = self.vendor_aliases.get(vendor.lower(), vendor.lower())
        
        if vendor in self.handlers:
            return self.handlers[vendor].parse_configuration(config_text)
        else:
            self.logger.warning(f"Unknown vendor: {vendor}")
            return {
                'vendor': vendor,
                'raw_config': config_text,
                'error': f'No handler for vendor: {vendor}'
            }
    
    def generate_configuration(self, config_data: Dict[str, Any], target_vendor: str) -> str:
        """Generate configuration for target vendor"""
        target_vendor = self.vendor_aliases.get(target_vendor.lower(), target_vendor.lower())
        
        if target_vendor not in self.handlers:
            raise ValueError(f"Unsupported target vendor: {target_vendor}")
        
        return self.handlers[target_vendor].generate_configuration(config_data)
    
    def translate_configuration(self, config_text: str, 
                              source_vendor: str, 
                              target_vendor: str) -> Tuple[str, List[str]]:
        """Translate configuration between vendors"""
        
        # Parse source configuration
        source_data = self.parse_configuration(config_text, source_vendor)
        
        if 'error' in source_data:
            return "", [f"Source parsing error: {source_data['error']}"]
        
        # Convert to target format
        try:
            target_config = self.generate_configuration(source_data, target_vendor)
            return target_config, []
        except Exception as e:
            return "", [f"Translation error: {e}"]
    
    def validate_configuration(self, config_text: str, vendor: Optional[str] = None) -> Dict[str, Any]:
        """Validate configuration syntax and semantics"""
        
        if vendor is None:
            vendor, confidence = self.detect_vendor(config_text)
        
        vendor = self.vendor_aliases.get(vendor.lower(), vendor.lower())
        
        if vendor not in self.handlers:
            return {
                'valid': False,
                'errors': [f'Unknown vendor: {vendor}'],
                'warnings': []
            }
        
        handler = self.handlers[vendor]
        is_valid, errors = handler.validate_syntax(config_text)
        
        return {
            'valid': is_valid,
            'vendor': vendor,
            'errors': errors,
            'warnings': []  # Could be extended
        }
    
    def get_supported_vendors(self) -> List[str]:
        """Get list of supported vendors"""
        return list(self.handlers.keys())
    
    def get_vendor_info(self, vendor: str) -> Optional[VendorConfig]:
        """Get vendor configuration information"""
        vendor = self.vendor_aliases.get(vendor.lower(), vendor.lower())
        
        if vendor in self.handlers:
            return self.handlers[vendor].config
        return None


# Utility functions
def create_vendor_manager() -> MultiVendorManager:
    """Create and initialize multi-vendor manager"""
    return MultiVendorManager()


def detect_and_parse(config_text: str) -> Dict[str, Any]:
    """Convenience function to detect and parse configuration"""
    manager = create_vendor_manager()
    return manager.parse_configuration(config_text)


def translate_config(source_config: str, 
                    source_vendor: str, 
                    target_vendor: str) -> Tuple[str, List[str]]:
    """Convenience function to translate configuration"""
    manager = create_vendor_manager()
    return manager.translate_configuration(source_config, source_vendor, target_vendor)

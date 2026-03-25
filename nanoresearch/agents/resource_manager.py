"""Resource manager for handling local datasets and models based on DATASETS.MD and MODELS.MD.

This module provides intelligent resource detection and management:
1. Reads DATASETS.MD and MODELS.MD files
2. Matches experiment requirements with available local resources
3. Prioritizes local resources over downloads
4. Provides metadata for proper resource usage
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages local datasets and models with intelligent matching."""

    def __init__(self, project_root: Path | str):
        """Initialize resource manager.

        Args:
            project_root: Root directory of the NanoResearch project
        """
        self.project_root = Path(project_root)
        self.datasets_md = self.project_root / "DATASETS.md"
        self.models_md = self.project_root / "MODELS.md"
        self.local_datasets_dir = self.project_root / "local_datasets"
        self.local_models_dir = self.project_root / "local_models"

        self._datasets_cache: Dict[str, Dict[str, Any]] = {}
        self._models_cache: Dict[str, Dict[str, Any]] = {}

    def load_resources(self) -> None:
        """Load and parse DATASETS.MD and MODELS.MD files."""
        if self.datasets_md.exists():
            self._datasets_cache = self._parse_datasets_md()
            logger.info(f"Loaded {len(self._datasets_cache)} datasets from DATASETS.md")
        else:
            logger.warning(f"DATASETS.md not found at {self.datasets_md}")

        if self.models_md.exists():
            self._models_cache = self._parse_models_md()
            logger.info(f"Loaded {len(self._models_cache)} models from MODELS.md")
        else:
            logger.warning(f"MODELS.md not found at {self.models_md}")

    def _parse_datasets_md(self) -> Dict[str, Dict[str, Any]]:
        """Parse DATASETS.md file and extract dataset information."""
        datasets = {}
        current_dataset = None

        with open(self.datasets_md, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Look for dataset headers (## DatasetName)
            if line.startswith('## ') and not line.startswith('---'):
                dataset_name = line[3:].strip()
                if dataset_name and dataset_name not in ['Dataset Index', 'How to Add New Datasets', 'Notes for Pipeline']:
                    current_dataset = dataset_name
                    datasets[current_dataset] = {
                        'name': dataset_name,
                        'location': '',
                        'task': '',
                        'class_count': 0,
                        'description': '',
                        'file_structure': '',
                        'key_files': [],
                        'concepts': [],
                        'aliases': self._generate_aliases(dataset_name)
                    }

            # Parse key information for current dataset
            elif current_dataset:
                # Location
                if '**Location**:' in line:
                    location = line.split('**Location**:')[1].strip().strip('`')
                    datasets[current_dataset]['location'] = location

                # Task
                elif '**Purpose**:' in line:
                    task = line.split('**Purpose**:')[1].strip()
                    datasets[current_dataset]['task'] = task

                # File structure
                elif line.startswith('```') and 'text' in line:
                    # Capture file structure
                    structure_lines = []
                    current_line_idx = None
                    for i, l in enumerate(lines):
                        if l.strip() == line:
                            current_line_idx = i
                            break

                    if current_line_idx is not None:
                        idx = current_line_idx + 1
                        while idx < len(lines) and not lines[idx].strip().startswith('```'):
                            structure_lines.append(lines[idx].rstrip())
                            idx += 1
                        datasets[current_dataset]['file_structure'] = '\n'.join(structure_lines)

                # Key files table
                elif '| File |' in line and '| Format |' in line and '| Usage |' in line:
                    # Skip header
                    current_line_idx = None
                    for i, l in enumerate(lines):
                        if l.strip() == line:
                            current_line_idx = i
                            break

                    if current_line_idx is not None:
                        idx = current_line_idx + 2
                        key_files = []
                        while idx < len(lines) and lines[idx].strip().startswith('|'):
                            parts = [p.strip() for p in lines[idx].split('|')[1:-1]]
                            if len(parts) >= 3 and parts[0] and not parts[0].startswith('-'):
                                key_files.append({
                                    'file': parts[0],
                                    'format': parts[1],
                                    'usage': parts[2]
                                })
                            idx += 1
                        datasets[current_dataset]['key_files'] = key_files

                # Concepts
                elif '**Concepts for CBM**' in line or '### Concepts' in line:
                    # Look for concept information in following lines
                    current_line_idx = None
                    for i, l in enumerate(lines):
                        if l.strip() == line:
                            current_line_idx = i
                            break

                    if current_line_idx is not None:
                        idx = current_line_idx + 1
                        concepts = []
                        while idx < len(lines) and not lines[idx].startswith('##') and not lines[idx].startswith('---'):
                            if '**' in lines[idx] and 'classes' in lines[idx]:
                                # Extract number of classes
                                match = re.search(r'(\d+)\s+classes?', lines[idx])
                                if match:
                                    datasets[current_dataset]['class_count'] = int(match.group(1))
                            elif '•' in lines[idx] or '-' in lines[idx]:
                                # Bullet points with concept info
                                concept_line = lines[idx].strip('•- ').strip()
                                if concept_line and not concept_line.startswith('```'):
                                    concepts.append(concept_line)
                            idx += 1
                        datasets[current_dataset]['concepts'] = concepts

        return datasets

    def _parse_models_md(self) -> Dict[str, Dict[str, Any]]:
        """Parse MODELS.md file and extract model information."""
        models = {}
        current_model = None

        with open(self.models_md, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Look for model headers (## ModelName)
            if line.startswith('## ') and not line.startswith('---'):
                model_name = line[3:].strip()
                if model_name and model_name not in ['Model Index', 'How to Add New Models', 'Notes for Pipeline']:
                    current_model = model_name
                    models[current_model] = {
                        'name': model_name,
                        'location': '',
                        'type': '',
                        'description': '',
                        'parameters': 0,
                        'architecture': '',
                        'training_data': '',
                        'usage_example': '',
                        'aliases': self._generate_aliases(model_name)
                    }

            # Parse key information for current model
            elif current_model:
                # Location
                if '**Location**:' in line:
                    location = line.split('**Location**:')[1].strip().strip('`')
                    models[current_model]['location'] = location

                # Purpose/Type
                elif '**Purpose**:' in line:
                    purpose = line.split('**Purpose**:')[1].strip()
                    models[current_model]['description'] = purpose
                    if 'language model' in purpose.lower():
                        models[current_model]['type'] = 'language_model'
                    elif 'vision' in purpose.lower():
                        models[current_model]['type'] = 'vision_model'
                    elif 'protein' in purpose.lower():
                        models[current_model]['type'] = 'protein_model'

                # Parameters
                elif '**' in line and 'parameters' in line:
                    match = re.search(r'(\d+(?:\.\d+)?[BM]?)\s+parameters?', line)
                    if match:
                        param_str = match.group(1)
                        if param_str.endswith('M'):
                            models[current_model]['parameters'] = int(float(param_str[:-1]) * 1e6)
                        elif param_str.endswith('B'):
                            models[current_model]['parameters'] = int(float(param_str[:-1]) * 1e9)
                        else:
                            models[current_model]['parameters'] = int(param_str)

                # Architecture
                elif 'Based on' in line or 'Architecture:' in line:
                    models[current_model]['architecture'] = line

                # Training data
                elif 'Trained on' in line:
                    models[current_model]['training_data'] = line

                # Usage example
                elif line.startswith('```python'):
                    # Capture Python usage example
                    example_lines = []
                    current_line_idx = None
                    for i, l in enumerate(lines):
                        if l.strip() == line:
                            current_line_idx = i
                            break

                    if current_line_idx is not None:
                        idx = current_line_idx + 1
                        while idx < len(lines) and not lines[idx].strip().startswith('```'):
                            example_lines.append(lines[idx].rstrip())
                            idx += 1
                        models[current_model]['usage_example'] = '\n'.join(example_lines)

        return models

    def _generate_aliases(self, name: str) -> List[str]:
        """Generate various aliases for a resource name."""
        aliases = [name]

        # Add variations
        name_lower = name.lower()
        aliases.append(name_lower)

        # Replace spaces and underscores
        no_spaces = name_lower.replace(' ', '_')
        aliases.append(no_spaces)
        aliases.append(name_lower.replace(' ', ''))
        aliases.append(name_lower.replace('_', ''))
        aliases.append(name_lower.replace('-', ''))

        # Add common abbreviations
        if 'cub' in name_lower and 'bird' in name_lower:
            aliases.extend(['cub', 'cub200', 'cub_200'])
        if 'awa' in name_lower:
            aliases.extend(['awa', 'awa2'])
        if 'esm' in name_lower:
            aliases.extend(['esm2', 'esm-2'])

        return list(set(aliases))

    def find_dataset(self, requirement: str) -> Optional[Dict[str, Any]]:
        """Find a matching dataset for the given requirement.

        Args:
            requirement: Dataset name or description from experiment blueprint

        Returns:
            Dataset information if found, None otherwise
        """
        if not self._datasets_cache:
            self.load_resources()

        requirement_lower = requirement.lower()

        # Direct match
        for name, info in self._datasets_cache.items():
            if requirement_lower == name.lower():
                return info

        # Alias match
        for name, info in self._datasets_cache.items():
            if any(alias in requirement_lower for alias in info['aliases']):
                return info

        # Keyword match with improved scoring
        best_match = None
        best_score = 0

        for name, info in self._datasets_cache.items():
            score = 0

            # Check name similarity
            name_lower = name.lower()
            if requirement_lower in name_lower:
                score += 10

            # Check task description
            task_lower = info['task'].lower()
            task_keywords = requirement_lower.split()
            matched_keywords = sum(1 for keyword in task_keywords if keyword in task_lower)
            score += matched_keywords * 2

            # Check concepts
            for concept in info.get('concepts', []):
                concept_lower = concept.lower()
                if requirement_lower in concept_lower:
                    score += 5
                else:
                    # Check individual keywords
                    for keyword in task_keywords:
                        if keyword in concept_lower:
                            score += 1

            # Check key files for relevance
            for key_file in info.get('key_files', []):
                file_text = f"{key_file['file']} {key_file['usage']}".lower()
                if requirement_lower in file_text:
                    score += 3

            # Update best match
            if score > best_score:
                best_score = score
                best_match = info

        # Return match if score is sufficient
        if best_score >= 5:  # Threshold for relevance
            return best_match

        return None

    def find_model(self, requirement: str) -> Optional[Dict[str, Any]]:
        """Find a matching model for the given requirement.

        Args:
            requirement: Model name or description from experiment blueprint

        Returns:
            Model information if found, None otherwise
        """
        if not self._models_cache:
            self.load_resources()

        requirement_lower = requirement.lower()

        # Direct match
        for name, info in self._models_cache.items():
            if requirement_lower == name.lower():
                return info

        # Alias match
        for name, info in self._models_cache.items():
            if any(alias in requirement_lower for alias in info['aliases']):
                return info

        # Keyword match with improved scoring
        best_match = None
        best_score = 0

        for name, info in self._models_cache.items():
            score = 0

            # Check name similarity
            name_lower = name.lower()
            if requirement_lower in name_lower:
                score += 10

            # Check description/purpose
            desc_lower = info.get('description', '').lower()
            desc_keywords = requirement_lower.split()
            matched_keywords = sum(1 for keyword in desc_keywords if keyword in desc_lower)
            score += matched_keywords * 2

            # Check type
            if info['type'] and info['type'] in requirement_lower:
                score += 5

            # Check architecture and training data
            arch_lower = info.get('architecture', '').lower()
            train_lower = info.get('training_data', '').lower()
            for text in [arch_lower, train_lower]:
                for keyword in desc_keywords:
                    if keyword in text:
                        score += 1

            # Update best match
            if score > best_score:
                best_score = score
                best_match = info

        # Return match if score is sufficient
        if best_score >= 5:  # Threshold for relevance
            return best_match

        return None

    def copy_dataset_to_workspace(self, dataset_name: str, workspace_dir: Path) -> Optional[Path]:
        """Copy a local dataset to the workspace.

        Args:
            dataset_name: Name of the dataset
            workspace_dir: Target workspace directory

        Returns:
            Path to copied dataset in workspace, None if failed
        """
        dataset_info = self.find_dataset(dataset_name)
        if not dataset_info:
            logger.error(f"Dataset '{dataset_name}' not found in local resources")
            return None

        # Get actual path from location field
        location = dataset_info.get('location', '')
        if not location:
            logger.error(f"No location specified for dataset '{dataset_name}'")
            return None

        # Resolve path
        if location.startswith('local_datasets/'):
            source_path = self.project_root / location
        elif location.startswith('/'):
            source_path = Path(location)
        else:
            source_path = self.local_datasets_dir / location

        if not source_path.exists():
            logger.error(f"Dataset source path does not exist: {source_path}")
            return None

        # Copy to workspace
        target_path = workspace_dir / "datasets" / dataset_name.replace(" ", "_")
        target_path.mkdir(parents=True, exist_ok=True)

        try:
            if source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_path)

            logger.info(f"Copied dataset '{dataset_name}' to workspace: {target_path}")
            return target_path

        except Exception as e:
            logger.error(f"Failed to copy dataset '{dataset_name}': {e}")
            return None

    def get_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset metadata including file structure, concepts, etc.
        """
        dataset_info = self.find_dataset(dataset_name)
        if not dataset_info:
            return {}

        # Add file analysis if dataset exists locally
        location = dataset_info.get('location', '')
        if location:
            if location.startswith('local_datasets/'):
                source_path = self.project_root / location
            else:
                source_path = self.local_datasets_dir / location

            if source_path.exists():
                dataset_info['exists_locally'] = True
                dataset_info['local_path'] = str(source_path)

                # Analyze directory structure
                if source_path.is_dir():
                    subdirs = [d.name for d in source_path.iterdir() if d.is_dir()]
                    files = [f.name for f in source_path.iterdir() if f.is_file()]
                    dataset_info['directory_contents'] = {
                        'subdirectories': subdirs[:20],  # Limit to first 20
                        'files': files[:20]
                    }

        return dataset_info

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for a model.

        Args:
            model_name: Name of the model

        Returns:
            Model metadata including architecture, parameters, etc.
        """
        return self.find_model(model_name) or {}

    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        if not self._datasets_cache:
            self.load_resources()
        return list(self._datasets_cache.keys())

    def list_available_models(self) -> List[str]:
        """List all available models."""
        if not self._models_cache:
            self.load_resources()
        return list(self._models_cache.keys())

    def generate_resource_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of available resources.

        Returns:
            Dictionary containing summary of all resources
        """
        self.load_resources()

        report = {
            'datasets': {
                'total': len(self._datasets_cache),
                'available': [],
                'summary': {}
            },
            'models': {
                'total': len(self._models_cache),
                'available': [],
                'summary': {}
            }
        }

        # Analyze datasets
        for name, info in self._datasets_cache.items():
            location = info.get('location', '')
            exists = False
            if location:
                if location.startswith('local_datasets/'):
                    path = self.project_root / location
                else:
                    path = self.local_datasets_dir / location
                exists = path.exists()

            report['datasets']['available'].append({
                'name': name,
                'task': info.get('task', ''),
                'class_count': info.get('class_count', 0),
                'exists_locally': exists
            })

        # Analyze models
        for name, info in self._models_cache.items():
            location = info.get('location', '')
            exists = False
            if location:
                if location.startswith('local_models/'):
                    path = self.project_root / location
                else:
                    path = self.local_models_dir / location
                exists = path.exists()

            report['models']['available'].append({
                'name': name,
                'type': info.get('type', ''),
                'parameters': info.get('parameters', 0),
                'exists_locally': exists
            })

        return report
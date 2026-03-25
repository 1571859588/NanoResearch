# Local Datasets

This document lists all local datasets available for use in NanoResearch experiments.

## Dataset Index

| Name | Location | Task | Class Count | Description |
|------|----------|------|-------------|-------------|
| CUB_200_2011 | `local_datasets/CUB_200_2011/` | Fine-grained bird classification | 200 | Caltech-UCSD Birds dataset with attributes, parts, and segmentations |
| AwA2 | `local_datasets/AwA2-data/` | Attribute-based classification | 50 | Animals with Attributes dataset v2 |

---

## CUB_200_2011 (Caltech-UCSD Birds-200-2011)

**Purpose**: Fine-grained visual classification, concept bottleneck models, zero-shot learning evaluation

**Location**: `local_datasets/CUB_200_2011/`

### File Structure

```
CUB_200_2011/
├── data/
│   ├── images/                 # Raw images organized by species (200 subdirectories)
│   ├── parts/                  # Part annotations
│   ├── attributes/             # Attribute annotations
│   ├── bounding_boxes.txt      # Bounding box annotations
│   ├── classes.txt             # Class labels (200 bird species)
│   ├── image_class_labels.txt  # Image-to-class mappings
│   ├── train_test_split.txt    # Train/test split
│   └── README                  # Official documentation
└── segmentations/              # Segmentation masks aligned with images
```

### Key Files for Experiments

| File | Format | Usage |
|------|--------|-------|
| `data/images/` | JPEG images | Input data for vision models |
| `data/classes.txt` | `<id> <class_name>` | Concept definitions for CBMs |
| `data/image_class_labels.txt` | `<img_id> <class_id>` | Training labels |
| `data/train_test_split.txt` | `<img_id> <is_train>` | Train/test partition |
| `data/parts/parts.txt` | `<part_id> <part_name>` | Concept-level attributes (15 parts) |
| `data/attributes/*.txt` | Various | Fine-grained attribute annotations |

### Usage in NanoResearch

```python
from pathlib import Path

DATASET_PATH = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_datasets/CUB_200_2011")

# Load class definitions
with open(DATASET_PATH / "data" / "classes.txt") as f:
    classes = {line.split()[0]: " ".join(line.split()[1:]) for line in f}

# Load image labels
with open(DATASET_PATH / "data" / "image_class_labels.txt") as f:
    labels = {line.split()[0]: int(line.split()[1]) for line in f}

# Get train/test split
with open(DATASET_PATH / "data" / "train_test_split.txt") as f:
    train_images = {line.split()[0] for line in f if line.split()[1] == "1"}
```

### Concepts for CBM

This dataset is ideal for Concept Bottleneck Models with:
- **200 fine-grained classes** (bird species)
- **15 anatomical parts** (beak, back, tail, etc.)
- **312 visual attributes** (color patterns, size, etc.)

---

## AwA2 (Animals with Attributes 2)

**Purpose**: Attribute-based classification, zero-shot learning, concept bottleneck models

**Location**: `local_datasets/AwA2-data/`

### Dataset Overview

- **50 animal classes** (mammals and birds)
- **85 human-annotated attributes** per class
- Split into 37 training classes and 13 test classes for zero-shot evaluation

### Usage in NanoResearch

```python
from pathlib import Path

DATASET_PATH = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_datasets/AwA2-data")

# Load class definitions (110 total: 50 classes + 60 attributes as concepts)
with open(DATASET_PATH / "classes_all.txt") as f:
    classes = {line.split()[0]: " ".join(line.split()[1:]) for line in f}

# Load attribute definitions (concepts for CBM)
with open(DATASET_PATH / "attributes.txt") as f:
    attributes = {int(line.split()[0]): line.split()[1] for line in f}
```

---

## How to Add New Datasets

1. Place dataset files in `local_datasets/<dataset_name>/`
2. Create a `README.md` describing the dataset structure
3. Add entry to this `DATASETS.md` file with:
   - Dataset name and location
   - Task type and class count
   - File structure
   - Usage instructions

---

## Notes for Pipeline

When the pipeline runs, it will:
1. Check `DATASETS.md` to discover available local resources
2. When the experiment blueprint requests a dataset, first check if it exists locally
3. If found locally, copy to workspace instead of downloading from the internet
4. Provide dataset metadata (class names, structure) to the CODING stage for proper dataset.py generation

# Using Local Resources in NanoResearch

This guide explains how to use local datasets and models in the NanoResearch pipeline, allowing you to avoid redundant downloads and leverage your existing resources.

## Overview

The NanoResearch pipeline now automatically detects and uses local resources based on the `DATASETS.md` and `MODELS.md` files in your project root. When planning experiments, the system will:

1. Check if required resources exist locally
2. Copy them to your workspace if found
3. Fall back to downloading only if not available locally

## Setting Up Local Resources

### Step 1: Organize Your Local Resources

Place your datasets and models in the appropriate directories:

```
/mnt/public/sichuan_a/nyt/ai/NanoResearch/
├── local_datasets/
│   ├── CUB_200_2011/
│   ├── YourDataset/
│   └── AnotherDataset/
├── local_models/
│   ├── ESM-2/
│   └── YourModel/
├── DATASETS.md
└── MODELS.md
```

### Step 2: Document Your Resources

#### For Datasets

Edit `DATASETS.md` to add your dataset:

```markdown
## YourDataset

**Purpose**: Brief description of what this dataset is used for

**Location**: `local_datasets/YourDataset/`

### File Structure

```
YourDataset/
├── data/
│   ├── images/          # Data files
│   ├── annotations/     # Labels/annotations
│   └── metadata.txt     # Dataset info
└── README.md            # Documentation
```

### Key Files for Experiments

| File | Format | Usage |
|------|--------|-------|
| `data/images/` | JPEG/PNG | Input data for models |
| `labels.csv` | CSV | Training labels |
| `train_test_split.txt` | Text | Dataset split information |

### Concepts for CBM

This dataset is ideal for Concept Bottleneck Models with:
- **100 classes** (object categories)
- **20 visual attributes** (color, shape, etc.)
- **Hierarchical structure** (class taxonomy)
```

#### For Models

Edit `MODELS.md` to add your model:

```markdown
## YourModel

**Purpose**: Brief description of the model's purpose

**Location**: `local_models/YourModel/`

### Model Overview

- **Architecture**: Transformer-based
- **Parameters**: 100M
- **Training data**: Domain-specific corpus
- **Output**: 768-dimensional embeddings

### Usage in NanoResearch

```python
# Example code to load and use the model
from transformers import AutoModel

model = AutoModel.from_pretrained("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_models/YourModel")
```
```

### Step 3: Verify Your Setup

Run the resource checker to ensure your resources are detected:

```bash
python test_local_resources.py
```

## How It Works

### During Pipeline Execution

1. **Resource Detection**: The pipeline reads your `DATASETS.md` and `MODELS.md` files
2. **Intelligent Matching**: When a dataset/model is needed, the system:
   - Matches the requirement to local resources
   - Scores matches based on name, description, and keywords
   - Selects the best match if score ≥ 5
3. **Local Copy**: If found locally, copies to workspace instead of downloading
4. **Metadata Provision**: Provides detailed metadata to downstream stages

### Matching Algorithm

The system uses a scoring algorithm:
- **Direct name match**: +10 points
- **Alias match**: +5 points
- **Keyword in description**: +2 points per keyword
- **Keyword in concepts**: +1 point per keyword
- **Minimum threshold**: 5 points for a valid match

## Best Practices

### 1. Dataset Organization

```
YourDataset/
├── data/                    # Raw data
│   ├── images/             # Image files
│   └── annotations/        # Annotations/labels
├── metadata/               # Dataset information
│   ├── classes.txt         # Class definitions
│   ├── train_test_split.txt # Dataset splits
│   └── attributes.txt      # Attribute definitions
└── README.md              # Documentation
```

### 2. Clear Documentation

- Use descriptive names in `DATASETS.md`
- Include file structure information
- Specify key files and their formats
- Mention the number of classes/attributes
- Add usage examples

### 3. Resource Naming

- Use consistent naming (e.g., `CUB_200_2011` not `cub200`)
- Include version numbers if applicable
- Avoid special characters in directory names

## Troubleshooting

### Resource Not Detected

1. Check the resource name in your requirement vs. `DATASETS.md`
2. Verify the directory exists in `local_datasets/`
3. Run `python test_local_resources.py` to see available resources
4. Check logs for matching attempts

### Copy Failed

1. Verify source directory permissions
2. Check available disk space
3. Ensure no files are locked/open
4. Check workspace directory permissions

### Wrong Resource Selected

1. Improve documentation in `DATASETS.md`
2. Add more specific keywords in the description
3. Use more precise names for your resources

## Advanced Usage

### Custom Matching Logic

You can extend the matching by adding aliases:

```python
# In your dataset documentation, the system automatically generates:
aliases = [
    "CUB_200_2011",
    "cub_200_2011",
    "cub200",
    "cub",
    "caltech_birds",
    # Add more as needed
]
```

### Programmatic Access

```python
from nanoresearch.agents.resource_manager import ResourceManager

# Initialize resource manager
manager = ResourceManager("/mnt/public/sichuan_a/nyt/ai/NanoResearch")
manager.load_resources()

# Find resources
dataset = manager.find_dataset("bird classification")
model = manager.find_model("protein language model")

# Get metadata
metadata = manager.get_dataset_metadata("CUB_200_2011")
```

### Resource Validation

```python
# Check if resource exists locally
report = manager.generate_resource_report()
for ds in report['datasets']['available']:
    print(f"{ds['name']}: exists_locally={ds['exists_locally']}")
```

## Examples

### Example 1: Adding a Custom Dataset

1. Place your dataset:
```bash
mkdir -p /mnt/public/sichuan_a/nyt/ai/NanoResearch/local_datasets/MyCustomDataset
cp -r /path/to/your/data/* /mnt/public/sichuan_a/nyt/ai/NanoResearch/local_datasets/MyCustomDataset/
```

2. Update `DATASETS.md`:
```markdown
## MyCustomDataset

**Purpose**: Custom dataset for medical image classification

**Location**: `local_datasets/MyCustomDataset/`

### File Structure

```
MyCustomDataset/
├── images/           # Medical images
├── masks/            # Segmentation masks
├── labels.csv        # Classification labels
└── metadata.json     # Image metadata
```

### Key Files

| File | Format | Usage |
|------|--------|-------|
| `images/*.jpg` | JPEG | Medical scan images |
| `labels.csv` | CSV | Disease classification |
| `metadata.json` | JSON | Patient demographics |
```

3. Use in research topic:
```bash
nanoresearch run --topic "Develop a CNN for medical image classification using MyCustomDataset"
```

### Example 2: Using Multiple Local Datasets

When your research requires multiple datasets:

```markdown
## MultiDatasetExperiment

**Purpose**: Combined analysis of multiple datasets

**Datasets Used**:
- CUB_200_2011 (birds)
- MyCustomDataset (medical)
- Additional synthetic data

**Integration Strategy**: [Describe how datasets are combined]
```

## Migration from Old System

If you have existing experiments:

1. Move downloaded datasets to `local_datasets/`
2. Document them in `DATASETS.md`
3. Future experiments will automatically use local copies
4. Existing workspaces remain unchanged

## Performance Considerations

- **Copy Speed**: Local copy is typically 10-100x faster than download
- **Disk Space**: Ensure sufficient space for both local and workspace copies
- **Network**: No bandwidth usage for local resources
- **Reliability**: No dependency on external servers

## Support

For issues or questions:
1. Check existing documentation
2. Run diagnostic scripts: `test_local_resources.py`
3. Review logs in your workspace
4. Contact the development team

Remember: The system is designed to make your research more efficient by leveraging local resources intelligently!
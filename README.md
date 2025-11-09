# LGD Mapping Application

A robust, modular Python application for mapping village, block, and district data to their corresponding LGD (Local Government Directory) codes using exact and fuzzy matching strategies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Configuration Options](#configuration-options)
  - [Hierarchical Mapping](#hierarchical-mapping)
  - [District Code Mapping](#district-code-mapping)
  - [Input File Formats](#input-file-formats)
  - [Output Files](#output-files)
- [Common Workflows](#common-workflows)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

## Overview

The LGD Mapping Application automates the process of mapping administrative entities (districts, blocks, and villages) to their official LGD codes. It uses a multi-strategy approach combining exact matching and configurable fuzzy matching to achieve high accuracy while handling data variations and inconsistencies.

### Key Capabilities

- **Flexible Hierarchical Mapping**: Supports full administrative hierarchy from state to village with automatic level detection
- **Multiple Matching Strategies**: Exact UID matching followed by fuzzy matching at configurable thresholds
- **Automatic Code Mapping**: Intelligently maps missing codes at all hierarchical levels (state, district, block, GP)
- **Data Quality Validation**: Comprehensive validation and quality checks throughout the process
- **Performance Optimization**: Automatic optimization for large datasets with chunk processing
- **Detailed Reporting**: Comprehensive statistics, quality metrics, and alternative match suggestions

## Features

- ✅ **Modular Architecture**: Clean separation of concerns with well-defined components
- ✅ **Flexible Hierarchical Support**: Automatic detection and mapping across all administrative levels (state, district, block, GP, village)
- ✅ **Automatic Code Mapping**: Intelligently maps missing codes at all hierarchical levels using exact and fuzzy matching
- ✅ **Exact Matching**: Fast UID-based matching for direct correspondences with variable-length hierarchical UIDs
- ✅ **Fuzzy Matching**: Configurable threshold-based fuzzy matching using RapidFuzz with level-specific thresholds
- ✅ **Progress Tracking**: Real-time progress bars and detailed logging
- ✅ **Error Handling**: Robust error handling with graceful degradation
- ✅ **Memory Optimization**: Automatic chunk processing for large datasets
- ✅ **Quality Metrics**: Detailed match statistics and data quality indicators
- ✅ **Alternative Suggestions**: Provides alternative matches for manual review
- ✅ **Organized Output**: Separate files for each matching strategy with timestamps
- ✅ **Backward Compatible**: Existing 3-level datasets work without any changes

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Standard Installation

1. Clone or download the repository:
```bash
git clone <repository-url>
cd lgd-mapping
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Development Installation

For development with editable installation:
```bash
pip install -e .
```

### Verify Installation

```bash
python main.py --help
```

You should see the command-line help documentation.

## Quick Start

### Basic Example

1. Prepare your input files (see [Input File Formats](#input-file-formats))

2. Run the mapping:
```bash
python main.py --entities entities.csv --codes codes.csv --output ./output
```

3. Check the output directory for results

### Using Sample Data

Try the application with sample data:

**3-level hierarchy (basic):**
```bash
python main.py --entities examples/sample_entities.csv --codes examples/sample_codes.csv --output ./test_output
```

**4-level hierarchy (with GP):**
```bash
python main.py --entities examples/sample_entities_4level.csv --codes examples/sample_codes_4level.csv --output ./test_output_4level
```

**5-level hierarchy (with state):**
```bash
python main.py --entities examples/sample_entities_5level.csv --codes examples/sample_codes_5level.csv --output ./test_output_5level --enable-state-code-mapping
```

## Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

#### Required Arguments

- `--entities PATH`: Path to entities CSV file containing records to be mapped
- `--codes PATH`: Path to LGD codes CSV file containing reference data
- `--output PATH`: Output directory for results (created if doesn't exist)

#### Optional Arguments

- `--thresholds INT [INT ...]`: Fuzzy matching thresholds (default: 95 90)
  - Example: `--thresholds 95 90 85`
  - Higher values = stricter matching
  - Values must be between 0-100

- `--log-level LEVEL`: Logging verbosity (default: INFO)
  - Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Example: `--log-level DEBUG`

- `--chunk-size INT`: Process data in chunks (for large datasets)
  - Example: `--chunk-size 5000`
  - Auto-detected for datasets >50,000 records

- `--district-mapping PATH`: JSON file with district name variations
  - Example: `--district-mapping examples/district_mapping.json`
  - Maps alternate spellings to standard codes

- `--cleanup-days INT`: Clean up old output files (default: 30)
  - Example: `--cleanup-days 7`
  - Set to 0 to disable cleanup

### Configuration Options

#### Using District Mapping

Create a JSON file to handle district name variations:

```json
{
  "district_code_mapping": {
    "Khordha": 101,
    "Khurda": 101,
    "KHORDHA": 101,
    "Cuttack": 102,
    "CUTTACK": 102
  }
}
```

Use it with:
```bash
python main.py --entities entities.csv --codes codes.csv --output ./output --district-mapping district_map.json
```

#### Programmatic Configuration

```python
from lgd_mapping.config import MappingConfig
from lgd_mapping.mapping_engine import MappingEngine
from lgd_mapping.logging_config import setup_logging

# Create configuration
config = MappingConfig(
    input_entities_file="entities.csv",
    input_codes_file="codes.csv",
    output_directory="./output",
    fuzzy_thresholds=[95, 90, 85],
    district_code_mapping={"Khordha": 101, "Khurda": 101},
    enable_district_code_mapping=True,
    district_fuzzy_threshold=85,
    chunk_size=5000,
    log_level="INFO"
)

# Initialize and run
logger = setup_logging(config)
engine = MappingEngine(config, logger)
results, stats = engine.run_complete_mapping()
```

### Hierarchical Mapping

The application supports flexible hierarchical mapping across all administrative levels in India. While the basic 3-level hierarchy (district → block → village) is always supported, the system can automatically detect and utilize additional levels when present in your data.

#### Quick Reference

| Hierarchy Type | Levels | UID Format | Sample Files | Use Case |
|----------------|--------|------------|--------------|----------|
| **3-Level** | District → Block → Village | `D_B_V` | `sample_entities.csv` | Basic datasets, backward compatibility |
| **4-Level** | District → Block → GP → Village | `D_B_G_V` | `sample_entities_4level.csv` | Regional data with GP info |
| **5-Level** | State → District → Block → GP → Village | `S_D_B_G_V` | `sample_entities_5level.csv` | Multi-state/national datasets |

**Key Benefits:**
- ✅ Automatic level detection - no configuration needed
- ✅ Backward compatible - existing 3-level data works unchanged
- ✅ Improved accuracy - more levels = better matching
- ✅ Flexible code mapping - automatically maps missing codes at any level

#### Supported Hierarchy Levels

The application recognizes the following administrative levels:

1. **State** (optional): State-level information
2. **District** (required): District-level information
3. **Block/Subdistrict** (required): Block or subdistrict level
4. **Gram Panchayat (GP)** (optional): GP-level information
5. **Village** (required): Village-level information

#### Automatic Hierarchy Detection

The system automatically detects which hierarchical levels are present in your input data by scanning for specific column names:

- State: `state`, `state_code`
- District: `district`, `district_code`
- Block: `block`, `block_code`
- GP: `gp`, `gp_code`
- Village: `village`, `village_code`

The detection process:
1. Scans both entity and LGD reference files for hierarchical columns
2. Validates that detected hierarchy is consistent (no gaps in levels)
3. Logs detected levels and hierarchy depth
4. Configures matching strategies based on available levels

**Example - 3-level hierarchy (basic):**
```csv
district,district_code,block,village
Khordha,362,Bhubaneswar,Patia
```

**Example - 4-level hierarchy (with GP):**
```csv
district,district_code,block,block_code,gp,gp_code,village,village_code
Khordha,362,Bhubaneswar,3621,Patia GP,36211,Patia,362101
```

**Example - 5-level hierarchy (full):**
```csv
state,state_code,district,district_code,block,block_code,gp,gp_code,village,village_code
Odisha,21,Khordha,362,Bhubaneswar,3621,Patia GP,36211,Patia,362101
```

#### Hierarchical Data Preparation

Preparing your data for hierarchical mapping ensures optimal results. Follow these guidelines:

##### Minimum Requirements

For basic 3-level mapping (always supported):
- **Required columns**: `district`, `block`, `village`
- **Optional columns**: `district_code`, `block_code`, `village_code`

##### Adding Hierarchical Levels

To enable enhanced hierarchical mapping, add any combination of these columns:

**For 4-level mapping (with GP):**
```csv
district,block,gp,village
Khordha,Bhubaneswar,Patia GP,Patia
```

**For 5-level mapping (with state):**
```csv
state,district,block,gp,village
Odisha,Khordha,Bhubaneswar,Patia GP,Patia
```

##### Column Naming Conventions

The system recognizes these exact column names (case-sensitive):

| Level | Name Column | Code Column |
|-------|-------------|-------------|
| State | `state` | `state_code` |
| District | `district` | `district_code` |
| Block | `block` | `block_code` |
| GP | `gp` | `gp_code` |
| Village | `village` | `village_code` |

**Important**: Use these exact names. Variations like "State", "DISTRICT", or "gram_panchayat" will not be detected.

##### Code Columns: Optional but Recommended

While code columns are optional (the system can map them automatically), providing codes improves:
- **Matching accuracy**: Eliminates ambiguity in name matching
- **Processing speed**: Reduces fuzzy matching overhead
- **Reliability**: Avoids potential name variation issues

**Best practice**: Provide codes when available, let the system map missing ones.

##### Data Quality Guidelines

1. **Consistency**: Ensure hierarchical relationships are valid
   - Example: All villages in "Bhubaneswar" block should belong to "Khordha" district

2. **Completeness**: Avoid missing values in hierarchical columns
   - Bad: `district=Khordha, block=, village=Patia`
   - Good: `district=Khordha, block=Bhubaneswar, village=Patia`

3. **Standardization**: Use consistent naming conventions
   - Decide on: "Gram Panchayat" vs "GP" vs "G.P."
   - Stick with one format throughout

4. **Encoding**: Use UTF-8 encoding for files with special characters

##### Example Data Preparation Workflow

**Step 1: Start with basic data**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
Khordha,Bhubaneswar,Sundarpada
```

**Step 2: Add GP information (if available)**
```csv
district,block,gp,village
Khordha,Bhubaneswar,Patia GP,Patia
Khordha,Bhubaneswar,Sundarpada GP,Sundarpada
```

**Step 3: Add codes (if available)**
```csv
district,district_code,block,block_code,gp,gp_code,village,village_code
Khordha,362,Bhubaneswar,3621,Patia GP,36211,Patia,362101
Khordha,362,Bhubaneswar,3621,Sundarpada GP,36212,Sundarpada,362102
```

**Step 4: Run mapping**
```bash
python main.py --entities prepared_data.csv --codes lgd_codes.csv --output ./results
```

The system will:
- Detect 4-level hierarchy (district → block → GP → village)
- Use provided codes directly
- Map any missing codes automatically
- Generate 4-level UIDs for improved matching

#### Hierarchical Code Mapping

When hierarchical levels are detected, the system automatically maps missing codes at each level:

- **State codes**: Maps state names to state codes (if `enable_state_code_mapping` is enabled)
- **District codes**: Maps district names to district codes (enabled by default)
- **Block codes**: Maps block names to block codes within the correct district (enabled by default)
- **GP codes**: Maps GP names to GP codes within the correct block (enabled by default)

Each level uses parent-scoped matching to ensure accuracy. For example, block code mapping only considers blocks within the matched district.

#### Hierarchical Configuration Options

##### Command-Line Options

```bash
# Enable/disable hierarchical mapping
--enable-hierarchical-mapping          # Enable full hierarchical mapping (default)
--disable-hierarchical-mapping         # Use legacy 3-level mapping only

# Enable/disable code mapping per level
--enable-state-code-mapping            # Enable state code mapping (default: disabled)
--disable-district-code-mapping        # Disable district code mapping
--disable-block-code-mapping           # Disable block code mapping
--disable-gp-code-mapping              # Disable GP code mapping

# Level-specific fuzzy thresholds
--state-fuzzy-threshold 85             # State matching threshold (default: 85)
--district-fuzzy-threshold 85          # District matching threshold (default: 85)
--block-fuzzy-threshold 90             # Block matching threshold (default: 90)
--gp-fuzzy-threshold 90                # GP matching threshold (default: 90)
--village-fuzzy-threshold 95           # Village matching threshold (default: 95)

# Hierarchy validation
--disable-hierarchy-consistency        # Disable consistency validation
--disallow-partial-hierarchy           # Require all levels (strict mode)
```

##### Programmatic Configuration

```python
from lgd_mapping.config import MappingConfig

config = MappingConfig(
    input_entities_file="entities.csv",
    input_codes_file="codes.csv",
    output_directory="./output",
    
    # Hierarchical mapping settings
    enable_hierarchical_mapping=True,
    enable_state_code_mapping=False,
    enable_district_code_mapping=True,
    enable_block_code_mapping=True,
    enable_gp_code_mapping=True,
    
    # Level-specific fuzzy thresholds
    state_fuzzy_threshold=85,
    district_fuzzy_threshold=85,
    block_fuzzy_threshold=90,
    gp_fuzzy_threshold=90,
    village_fuzzy_threshold=95,
    
    # Hierarchy validation
    enforce_hierarchy_consistency=True,
    allow_partial_hierarchy=True
)
```

#### Hierarchical UID Generation

The system generates Unique Identifiers (UIDs) based on all available hierarchical levels:

- **3-level UID**: `district_code_block_code_village_code` (e.g., `362_3621_362101`)
- **4-level UID**: `district_code_block_code_gp_code_village_code` (e.g., `362_3621_36211_362101`)
- **5-level UID**: `state_code_district_code_block_code_gp_code_village_code` (e.g., `21_362_3621_36211_362101`)

UIDs are automatically generated based on detected hierarchy levels, improving match accuracy when more detailed data is available.

#### Benefits of Hierarchical Mapping

1. **Improved Accuracy**: More hierarchical levels = better matching precision
2. **Flexible Data Support**: Works with any combination of available levels
3. **Backward Compatible**: Existing 3-level datasets work without changes
4. **Automatic Detection**: No manual configuration needed for hierarchy detection
5. **Parent-Scoped Matching**: Each level is matched within its parent's context

#### Example Workflows

**Workflow 1: Basic 3-level mapping (default)**

Use this for traditional datasets with district, block, and village only.

```bash
python main.py \
  --entities entities_3level.csv \
  --codes lgd_codes.csv \
  --output ./results
```

**Input example (entities_3level.csv):**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
Cuttack,Cuttack Sadar,Bidanasi
```

**Expected output:**
- 3-level UIDs: `362_3621_362101`
- Standard matching process
- Backward compatible with legacy systems

---

**Workflow 2: 4-level mapping with GP**

Use this when your data includes Gram Panchayat information.

```bash
python main.py \
  --entities entities_4level.csv \
  --codes lgd_codes.csv \
  --output ./results \
  --gp-fuzzy-threshold 90
```

**Input example (entities_4level.csv):**
```csv
district,block,gp,village
Khordha,Bhubaneswar,Patia GP,Patia
Khordha,Bhubaneswar,Sundarpada GP,Sundarpada
```

**Expected output:**
- 4-level UIDs: `362_3621_36211_362101`
- Improved matching accuracy with GP context
- GP code mapping if codes not provided

---

**Workflow 3: Full 5-level hierarchical mapping**

Use this for comprehensive state-level datasets.

```bash
python main.py \
  --entities entities_5level.csv \
  --codes lgd_codes_full.csv \
  --output ./results \
  --enable-state-code-mapping \
  --state-fuzzy-threshold 85
```

**Input example (entities_5level.csv):**
```csv
state,district,block,gp,village
Odisha,Khordha,Bhubaneswar,Patia GP,Patia
Odisha,Cuttack,Cuttack Sadar,Bidanasi GP,Bidanasi
West Bengal,Kolkata,Kolkata,Park Street GP,Park Street
```

**Expected output:**
- 5-level UIDs: `21_362_3621_36211_362101`
- State-level scoping for multi-state datasets
- Highest matching accuracy with full hierarchy

---

**Workflow 4: Custom hierarchy with adjusted thresholds**

Use this to fine-tune matching sensitivity at each level.

```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./results \
  --district-fuzzy-threshold 85 \
  --block-fuzzy-threshold 88 \
  --gp-fuzzy-threshold 90 \
  --village-fuzzy-threshold 95
```

**Use case:**
- District names have more variations → lower threshold (85)
- Block names are more standardized → medium threshold (88)
- GP names are fairly consistent → higher threshold (90)
- Village names must match closely → highest threshold (95)

---

**Workflow 5: Partial hierarchy with code mapping disabled**

Use this when you have complete codes and want to skip automatic mapping.

```bash
python main.py \
  --entities entities_with_codes.csv \
  --codes codes.csv \
  --output ./results \
  --disable-district-code-mapping \
  --disable-block-code-mapping \
  --disable-gp-code-mapping
```

**Input example (entities_with_codes.csv):**
```csv
district,district_code,block,block_code,gp,gp_code,village,village_code
Khordha,362,Bhubaneswar,3621,Patia GP,36211,Patia,362101
```

**Use case:**
- All codes are already accurate
- Skip code mapping for faster processing
- Useful for pre-validated datasets

---

**Workflow 6: Strict hierarchical validation**

Use this to enforce complete hierarchy and catch data quality issues.

```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./results \
  --disallow-partial-hierarchy \
  --enforce-hierarchy-consistency
```

**What it does:**
- Requires all detected levels to be present in every record
- Validates hierarchical relationships (e.g., block belongs to district)
- Fails fast on inconsistencies
- Useful for data quality auditing

### District Code Mapping

The application includes an intelligent district code mapping feature that automatically enriches entity records with district codes before matching begins. This significantly improves match rates, especially when input data lacks district codes or has district name variations.

#### How It Works

The district code mapping process runs automatically during data loading and follows these steps:

1. **Extraction**: Identifies unique district names from your entity data
2. **Normalization**: Standardizes district names by:
   - Removing parenthetical text (e.g., "Keonjhar (Kendujhar)" → "Keonjhar")
   - Trimming whitespace
   - Converting to lowercase for comparison
3. **Exact Matching**: Attempts to match normalized district names against LGD reference data
4. **Fuzzy Matching**: If exact match fails, uses fuzzy string matching to handle spelling variations
5. **Enrichment**: Populates the `district_code` field in entity records with matched codes

#### Benefits

- **Improved Match Rates**: Entities without district codes can now be matched
- **Handles Name Variations**: Automatically handles spelling differences like "Keonjhar" vs "Kendujhar"
- **Transparent Processing**: Detailed logging shows mapping progress and statistics
- **No Manual Intervention**: Works automatically without requiring additional mapping files

#### Configuration Parameters

##### enable_district_code_mapping

Controls whether automatic district code mapping is enabled.

- **Type**: Boolean
- **Default**: `true`
- **When to disable**: If your entities already have accurate district codes or you want to use manual district mapping only

**Example - Disable district code mapping:**

```json
{
  "enable_district_code_mapping": false
}
```

Or via programmatic configuration:

```python
config = MappingConfig(
    input_entities_file="entities.csv",
    input_codes_file="codes.csv",
    output_directory="./output",
    enable_district_code_mapping=False
)
```

##### district_fuzzy_threshold

Sets the minimum similarity score (0-100) for fuzzy matching district names.

- **Type**: Integer
- **Default**: `85`
- **Range**: 0-100
- **Recommended**: 80-90 for most use cases

**Higher values (90-95)**:
- More strict matching
- Fewer false positives
- May miss valid variations

**Lower values (75-85)**:
- More lenient matching
- Catches more variations
- May include incorrect matches

**Example - Adjust fuzzy threshold:**

```json
{
  "district_fuzzy_threshold": 90
}
```

Or via programmatic configuration:

```python
config = MappingConfig(
    input_entities_file="entities.csv",
    input_codes_file="codes.csv",
    output_directory="./output",
    district_fuzzy_threshold=90
)
```

#### Understanding the Logs

The district code mapping process provides detailed logging:

**INFO Level - Mapping Statistics:**
```
INFO: Starting district code mapping
INFO: Found 15 unique districts requiring mapping
INFO: District code mapping completed: 14/15 districts mapped (93.3% success rate)
INFO: Mapping details: 12 exact matches, 2 fuzzy matches
```

**WARNING Level - Unmapped Districts:**
```
WARNING: Could not map district name to code: "Unknown District"
WARNING: Low district code mapping success rate: 45.0%
```

**DEBUG Level - Detailed Matching:**
```
DEBUG: Normalized district name: "keonjhar (kendujhar)" -> "keonjhar"
DEBUG: Exact match found: "keonjhar" -> district code 361
DEBUG: Fuzzy match: "kendujhar" matched "keonjhar" with score 92
```

#### Example Scenarios

##### Scenario 1: Missing District Codes

**Input entities.csv:**
```csv
district,block,village
Keonjhar (Kendujhar),Anandapur,Bhuinpur
Khordha,Bhubaneswar,Patia
```

**Result**: The system automatically maps "Keonjhar (Kendujhar)" to district code 361 and "Khordha" to its corresponding code, enabling successful matching.

##### Scenario 2: Spelling Variations

**Input entities.csv:**
```csv
district,block,village
Kendujhar,Anandapur,Bhuinpur
Keonjhar,Champua,Joda
```

**Result**: Both "Kendujhar" and "Keonjhar" are recognized as the same district (code 361) through fuzzy matching.

##### Scenario 3: Mixed Data Quality

**Input entities.csv:**
```csv
district,district_code,block,village
Khordha,362,Bhubaneswar,Patia
Cuttack,,Cuttack Sadar,Bidanasi
```

**Result**: Preserves existing district code (362) for Khordha, maps district code for Cuttack automatically.

#### Best Practices

1. **Review Mapping Statistics**: Check the logs to ensure high mapping success rates (>90% is ideal)
2. **Adjust Threshold Carefully**: Start with default (85) and adjust based on your data quality
3. **Combine with Manual Mapping**: Use `district_code_mapping` parameter for known variations alongside automatic mapping
4. **Monitor Unmapped Districts**: Review WARNING logs to identify districts that need attention
5. **Validate Results**: Check a sample of mapped entities to ensure accuracy

### Input File Formats

#### Column Naming Conventions

The application uses **strict, case-sensitive column names**. Use these exact names in your CSV files:

| Level | Name Column | Code Column | Required | Notes |
|-------|-------------|-------------|----------|-------|
| State | `state` | `state_code` | No | For multi-state datasets |
| District | `district` | `district_code` | Yes | Core level, always required |
| Subdistrict | `subdistrict` | `subdistrict_code` | No | Alternative to block |
| Block | `block` | `block_code` | Yes | Core level, always required |
| GP | `gp` | `gp_code` | No | Gram Panchayat level |
| Village | `village` | `village_code` | Yes | Core level, always required |

**Important Notes:**
- Column names are **case-sensitive**: use lowercase (`state`, not `State` or `STATE`)
- Use exact names: `gp` not `gram_panchayat`, `block` not `block_name`
- Code columns are optional - the system can map them automatically
- Name columns are required for the levels you want to include

**Special Case for LGD Reference File:**
- LGD files may use `state_name` instead of `state` (automatically handled)
- All other columns must use standard names

#### Entities File (entities.csv)

The entities file contains your source data to be mapped to LGD codes.

##### Minimum Required Structure (3-level)

```csv
district,block,village
Khordha,Bhubaneswar,Patia
Cuttack,Cuttack Sadar,Bidanasi
Puri,Puri Sadar,Penthakata
```

**Required columns:**
- `district`: District name (string)
- `block`: Block name (string)
- `village`: Village name (string)

##### With Optional Codes (Recommended)

```csv
district,district_code,block,block_code,village,village_code
Khordha,362,Bhubaneswar,3621,Patia,362101
Cuttack,363,Cuttack Sadar,3631,Bidanasi,363101
Puri,364,Puri Sadar,3641,Penthakata,364101
```

**Benefits of providing codes:**
- Faster processing (skips code mapping)
- Higher accuracy (eliminates name ambiguity)
- Better exact matching results

##### 4-Level Structure (with GP)

```csv
district,district_code,block,block_code,gp,gp_code,village,village_code
Khordha,362,Bhubaneswar,3621,Patia GP,36211,Patia,362101
Cuttack,363,Cuttack Sadar,3631,Bidanasi GP,36311,Bidanasi,363101
```

**Additional columns:**
- `gp`: Gram Panchayat name (string)
- `gp_code`: Gram Panchayat code (integer, optional)

##### 5-Level Structure (with State)

```csv
state,state_code,district,district_code,block,block_code,gp,gp_code,village,village_code
Odisha,21,Khordha,362,Bhubaneswar,3621,Patia GP,36211,Patia,362101
Odisha,21,Cuttack,363,Cuttack Sadar,3631,Bidanasi GP,36311,Bidanasi,363101
West Bengal,19,Kolkata,341,Kolkata,3411,Park Street GP,34111,Park Street,341101
```

**Additional columns:**
- `state`: State name (string)
- `state_code`: State code (integer, optional)

##### Data Type Specifications

| Column Type | Data Type | Format | Example |
|-------------|-----------|--------|---------|
| Name columns | String | UTF-8 text | `Khordha`, `Bhubaneswar` |
| Code columns | Integer | Numeric only | `362`, `3621`, `362101` |

**Important:**
- Code columns should contain integers only (no decimals, no text)
- Name columns should be UTF-8 encoded for special characters
- Avoid leading/trailing spaces in names
- Empty values should be truly empty, not "NA" or "NULL"

##### File Format Requirements

- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8 (recommended) or ASCII
- **Line endings**: Unix (LF) or Windows (CRLF) - both supported
- **Header row**: Required (first row must contain column names)
- **Delimiter**: Comma (`,`)
- **Quote character**: Double quote (`"`) for values containing commas

#### LGD Codes File (codes.csv)

The LGD codes file contains the reference data with official LGD codes.

##### Minimum Required Structure (3-level)

```csv
district_code,district,block_code,block,village_code,village
362,Khordha,3621,Bhubaneswar,362101,Patia
363,Cuttack,3631,Cuttack Sadar,363101,Bidanasi
364,Puri,3641,Puri Sadar,364101,Penthakata
```

**Required columns:**
- `district_code`: LGD district code (integer)
- `district`: District name (string)
- `block_code`: LGD block code (integer)
- `block`: Block name (string)
- `village_code`: LGD village code (integer)
- `village`: Village name (string)

##### 4-Level Structure (with GP)

```csv
district_code,district,block_code,block,gp_code,gp,village_code,village
362,Khordha,3621,Bhubaneswar,36211,Patia GP,362101,Patia
363,Cuttack,3631,Cuttack Sadar,36311,Bidanasi GP,363101,Bidanasi
```

**Additional columns:**
- `gp_code`: Gram Panchayat code (integer)
- `gp`: Gram Panchayat name (string)

##### 5-Level Structure (with State)

```csv
state_code,state_name,district_code,district,block_code,block,gp_code,gp,village_code,village
21,Odisha,362,Khordha,3621,Bhubaneswar,36211,Patia GP,362101,Patia
21,Odisha,363,Cuttack,3631,Cuttack Sadar,36311,Bidanasi GP,363101,Bidanasi
19,West Bengal,341,Kolkata,3411,Kolkata,34111,Park Street GP,341101,Park Street
```

**Additional columns:**
- `state_code`: State code (integer)
- `state_name`: State name (string) - **Note:** LGD files use `state_name` not `state`

**Special Note:** The LGD reference file uses `state_name` instead of `state` for the state name column. This is automatically handled by the system.

##### File Format Requirements

Same as entities file:
- **Format**: CSV
- **Encoding**: UTF-8
- **Header row**: Required
- **All code columns**: Must be present and contain valid integers

#### Common File Preparation Issues

##### Issue: Column Name Mismatches

**Problem:**
```csv
District,Block Name,Village Name
Khordha,Bhubaneswar,Patia
```

**Solution:**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
```

##### Issue: Extra Spaces in Column Names

**Problem:**
```csv
district ,block, village
Khordha,Bhubaneswar,Patia
```

**Solution:**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
```

##### Issue: Code Columns as Text

**Problem:**
```csv
district,district_code,block,village
Khordha,"362",Bhubaneswar,Patia
```

**Solution:**
```csv
district,district_code,block,village
Khordha,362,Bhubaneswar,Patia
```

##### Issue: Missing Header Row

**Problem:**
```csv
Khordha,Bhubaneswar,Patia
Cuttack,Cuttack Sadar,Bidanasi
```

**Solution:**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
Cuttack,Cuttack Sadar,Bidanasi
```

#### Validating Your Input Files

Before processing, validate your files:

```bash
# Check column names (should match exactly)
head -1 entities.csv

# Check for extra spaces
head -1 entities.csv | cat -A

# Count records
wc -l entities.csv

# Check for encoding issues
file -i entities.csv
```

Or use Python:

```python
import pandas as pd

# Load and inspect
df = pd.read_csv('entities.csv')

# Check columns
print("Columns:", df.columns.tolist())

# Check for required columns
required = ['district', 'block', 'village']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"Missing required columns: {missing}")
else:
    print("All required columns present!")

# Check data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
```

### Output Files

The application generates timestamped output files in the specified output directory:

#### Match Result Files

1. **exact_matches_YYYYMMDD_HHMMSS.csv**
   - Records with exact UID matches
   - Highest confidence matches

2. **fuzzy_95_matches_YYYYMMDD_HHMMSS.csv**
   - Records matched with 95% similarity threshold
   - High confidence fuzzy matches

3. **fuzzy_90_matches_YYYYMMDD_HHMMSS.csv**
   - Records matched with 90% similarity threshold
   - Medium confidence fuzzy matches

4. **unmatched_with_alternatives_YYYYMMDD_HHMMSS.csv**
   - Unmatched records with suggested alternatives
   - Requires manual review

5. **unmatched_no_alternatives_YYYYMMDD_HHMMSS.csv**
   - Records with no viable matches found
   - May need data correction

#### Report Files

6. **mapping_summary_report_YYYYMMDD_HHMMSS.txt**
   - Comprehensive statistics and quality metrics
   - Match rates and processing time
   - Data quality indicators

7. **mapping_log_YYYYMMDD_HHMMSS.txt**
   - Detailed processing log with timestamps
   - Progress tracking and warnings
   - Error messages and diagnostics

## Common Workflows

### Workflow 1: Standard Mapping

For typical datasets with good data quality:

```bash
python main.py \
  --entities my_entities.csv \
  --codes lgd_codes.csv \
  --output ./results
```

### Workflow 2: Large Dataset Processing

For datasets with >50,000 records:

```bash
python main.py \
  --entities large_entities.csv \
  --codes lgd_codes.csv \
  --output ./results \
  --chunk-size 10000 \
  --log-level INFO
```

### Workflow 3: High Accuracy Mapping

For critical data requiring high accuracy:

```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./results \
  --thresholds 98 95 \
  --log-level DEBUG
```

### Workflow 4: Handling District Variations

When source data has inconsistent district names, the automatic district code mapping handles most variations. For additional known variations:

```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./results \
  --district-mapping district_variations.json \
  --thresholds 95 90 85
```

**Note**: Automatic district code mapping is enabled by default and handles common variations like "Keonjhar" vs "Kendujhar". Use manual mapping for additional edge cases.

### Workflow 5: Iterative Refinement

1. Run initial mapping:
```bash
python main.py --entities entities.csv --codes codes.csv --output ./round1
```

2. Review district code mapping statistics in logs:
```bash
grep "district code mapping" output/mapping_log_*.txt
```

3. Review unmatched records with alternatives

4. Create manual district mapping for unmapped districts or adjust fuzzy threshold:
```bash
python main.py --entities entities.csv --codes codes.csv --output ./round2 --district-mapping refined_mapping.json
```

Or adjust the district fuzzy threshold in configuration:
```json
{
  "district_fuzzy_threshold": 80
}
```

## Performance Tuning

### Dataset Size Guidelines

| Records | Chunk Size | Thresholds | Expected Time |
|---------|------------|------------|---------------|
| < 10K   | Auto       | 95, 90     | < 1 minute    |
| 10K-50K | Auto       | 95, 90     | 1-5 minutes   |
| 50K-100K| 5,000      | 95, 90     | 5-15 minutes  |
| 100K+   | 10,000     | 95         | 15+ minutes   |

### Optimization Strategies

#### For Speed

1. **Reduce fuzzy thresholds**: Use only one threshold (95)
```bash
--thresholds 95
```

2. **Enable chunk processing**: Process in smaller batches
```bash
--chunk-size 5000
```

3. **Reduce logging**: Use WARNING level
```bash
--log-level WARNING
```

#### For Accuracy

1. **Multiple thresholds**: Use graduated thresholds
```bash
--thresholds 98 95 90 85
```

2. **District mapping**: Handle name variations
```bash
--district-mapping variations.json
```

3. **Debug logging**: Identify data quality issues
```bash
--log-level DEBUG
```

#### For Memory Efficiency

1. **Smaller chunks**: Reduce memory footprint
```bash
--chunk-size 2000
```

2. **Automatic optimization**: Let the system decide
   - Auto-enabled for datasets >50,000 records
   - Monitors memory usage during processing

### Performance Monitoring

The application automatically tracks:
- Processing time per phase
- Memory usage (peak and growth)
- Processing rate (entities/second)
- Match rates by strategy

Check the summary report for detailed metrics.

## Troubleshooting

### Common Issues

#### Issue: File Not Found Error

**Symptom**: `FileNotFoundError: Entities file not found`

**Solutions**:
- Verify file path is correct
- Use absolute paths: `/full/path/to/file.csv`
- Check file permissions
- Ensure file exists: `ls -l entities.csv`

#### Issue: Missing Required Columns

**Symptom**: `KeyError: 'district'` or similar column errors

**Solutions**:
- Verify CSV has required columns (see [Input File Formats](#input-file-formats))
- Check for typos in column names (case-sensitive)
- Ensure no extra spaces in headers
- Validate with: `head -1 entities.csv`

#### Issue: Low District Code Mapping Success Rate

**Symptom**: Log shows "Low district code mapping success rate: X%" where X < 50%

**Causes**:
- District names in entities don't match LGD reference data
- Misspelled or non-standard district names
- Districts from different states not in LGD reference data
- Data encoding issues (special characters)

**Solutions**:

1. **Review unmapped districts in logs**:
   ```bash
   grep "Could not map district" output/mapping_log_*.txt
   ```

2. **Check district names in your data**:
   ```bash
   # List unique districts in entities
   cut -d',' -f1 input/entities.csv | sort -u
   ```

3. **Verify LGD reference data coverage**:
   - Ensure LGD codes file includes all states/districts in your entities
   - Check for spelling differences between entities and LGD data

4. **Adjust fuzzy matching threshold**:
   - Lower threshold for more lenient matching (may increase false positives):
   ```json
   {
     "district_fuzzy_threshold": 80
   }
   ```
   - Raise threshold for stricter matching (may miss valid variations):
   ```json
   {
     "district_fuzzy_threshold": 90
   }
   ```

5. **Use manual district mapping for known variations**:
   ```json
   {
     "district_code_mapping": {
       "Khurda": 362,
       "Khordha": 362,
       "KHORDHA": 362
     }
   }
   ```

6. **Clean source data**:
   - Standardize district names before processing
   - Remove extra spaces, special characters
   - Ensure consistent spelling

7. **Enable DEBUG logging for detailed analysis**:
   ```bash
   python main.py --entities entities.csv --codes codes.csv --output ./output --log-level DEBUG
   ```

**Expected Success Rates**:
- **>90%**: Excellent - data is well-aligned with LGD reference
- **70-90%**: Good - minor variations, consider manual mapping for unmapped districts
- **50-70%**: Fair - review data quality and consider cleaning source data
- **<50%**: Poor - significant data quality issues or mismatched reference data

#### Issue: Low Match Rates

**Symptom**: High percentage of unmatched records after complete mapping

**Solutions**:
1. **Check district code mapping first**:
   - Review district code mapping statistics in logs
   - Low district code mapping success directly impacts overall match rates
   - See "Low District Code Mapping Success Rate" above

2. **Review unmatched records**:
   ```bash
   cat output/unmatched_with_alternatives_*.csv
   ```

3. **Create district mapping for variations**:
   ```json
   {
     "district_code_mapping": {
       "Variation1": 101,
       "Variation2": 101
     }
   }
   ```

4. **Lower fuzzy matching thresholds**:
   ```bash
   --thresholds 90 85 80
   ```

5. **Review data quality in source files**:
   - Check for typos in block and village names
   - Ensure hierarchical consistency (district → block → village)

#### Issue: Memory Errors

**Symptom**: `MemoryError` or system slowdown

**Solutions**:
1. Enable chunk processing:
   ```bash
   --chunk-size 5000
   ```

2. Reduce chunk size further:
   ```bash
   --chunk-size 2000
   ```

3. Close other applications

4. Process in batches (split input file)

#### Issue: Slow Performance

**Symptom**: Processing takes too long

**Solutions**:
1. Use fewer fuzzy thresholds:
   ```bash
   --thresholds 95
   ```

2. Increase chunk size (if memory allows):
   ```bash
   --chunk-size 10000
   ```

3. Reduce logging verbosity:
   ```bash
   --log-level WARNING
   ```

4. Check system resources (CPU, memory)

#### Issue: Incorrect Matches

**Symptom**: Mapped records don't look correct

**Solutions**:
1. Increase fuzzy threshold:
   ```bash
   --thresholds 98 95
   ```

2. Review exact matches first (highest confidence)

3. Check district code mapping accuracy

4. Validate LGD codes file data quality

#### Issue: Hierarchical Columns Not Detected

**Symptom**: Log shows "Detected hierarchy levels: ['district', 'block', 'village']" but you have GP or state columns

**Causes**:
- Column names don't match expected format
- Typos in column headers
- Case sensitivity issues
- Extra spaces in column names

**Solutions**:

1. **Verify exact column names** (case-sensitive):
   ```bash
   head -1 entities.csv
   ```
   
   Expected names: `state`, `district`, `block`, `gp`, `village`
   Not: `State`, `DISTRICT`, `Block Name`, `gram_panchayat`

2. **Check for extra spaces**:
   ```bash
   # Bad: "district ", " block", "gp "
   # Good: "district", "block", "gp"
   ```

3. **Standardize column names**:
   ```python
   import pandas as pd
   df = pd.read_csv('entities.csv')
   df.columns = df.columns.str.strip().str.lower()
   df.to_csv('entities_fixed.csv', index=False)
   ```

4. **Enable debug logging** to see detection details:
   ```bash
   python main.py --entities entities.csv --codes codes.csv --output ./output --log-level DEBUG
   ```

#### Issue: Hierarchical Code Mapping Failures

**Symptom**: Log shows "Block code mapping: 45/100 (45.0%)" or similar low success rates

**Causes**:
- Block/GP names don't match LGD reference data
- Missing parent codes (e.g., district_code needed for block mapping)
- Hierarchical inconsistencies in data
- LGD reference data incomplete

**Solutions**:

1. **Check parent code availability**:
   - Block code mapping requires district_code
   - GP code mapping requires block_code
   - Ensure parent levels are mapped first

2. **Review unmapped items in logs**:
   ```bash
   grep "Could not map.*code" output/mapping_log_*.txt
   ```

3. **Verify hierarchical consistency**:
   ```python
   # Check if blocks belong to correct districts
   import pandas as pd
   entities = pd.read_csv('entities.csv')
   codes = pd.read_csv('codes.csv')
   
   # Verify block-district relationships
   merged = entities.merge(
       codes[['district', 'block', 'block_code']],
       on=['district', 'block'],
       how='left'
   )
   print(merged[merged['block_code'].isna()])
   ```

4. **Adjust level-specific fuzzy thresholds**:
   ```bash
   python main.py \
     --entities entities.csv \
     --codes codes.csv \
     --output ./output \
     --block-fuzzy-threshold 85 \
     --gp-fuzzy-threshold 85
   ```

5. **Check LGD reference data coverage**:
   ```bash
   # List unique blocks in LGD data
   cut -d',' -f3 codes.csv | sort -u | wc -l
   
   # List unique blocks in entities
   cut -d',' -f2 entities.csv | sort -u | wc -l
   ```

#### Issue: Hierarchical Validation Errors

**Symptom**: `HierarchyValidationError` or warnings about hierarchical inconsistencies

**Causes**:
- Block doesn't belong to specified district
- GP doesn't belong to specified block
- Village doesn't belong to specified GP
- Mismatched codes in input data

**Solutions**:

1. **Review validation warnings**:
   ```bash
   grep "hierarchical inconsistency" output/mapping_log_*.txt
   ```

2. **Disable strict validation** (if data quality is known issue):
   ```bash
   python main.py \
     --entities entities.csv \
     --codes codes.csv \
     --output ./output \
     --disable-hierarchy-consistency
   ```

3. **Fix data quality issues**:
   ```python
   import pandas as pd
   
   entities = pd.read_csv('entities.csv')
   codes = pd.read_csv('codes.csv')
   
   # Validate district-block relationships
   valid_blocks = codes[['district_code', 'block_code']].drop_duplicates()
   entities_validated = entities.merge(
       valid_blocks,
       on=['district_code', 'block_code'],
       how='inner'
   )
   
   # Save cleaned data
   entities_validated.to_csv('entities_validated.csv', index=False)
   ```

4. **Check for duplicate codes**:
   ```bash
   # Check for duplicate block codes across districts
   cut -d',' -f2,3 codes.csv | sort | uniq -d
   ```

#### Issue: Variable-Length UID Matching Problems

**Symptom**: Exact matches failing despite correct codes, or UID format errors

**Causes**:
- Inconsistent hierarchy depth between entities and LGD data
- Missing codes at intermediate levels
- UID generation using different hierarchy levels

**Solutions**:

1. **Verify hierarchy consistency**:
   ```bash
   # Check detected hierarchy in logs
   grep "Detected hierarchy" output/mapping_log_*.txt
   grep "Hierarchy depth" output/mapping_log_*.txt
   ```

2. **Ensure matching hierarchy levels**:
   - Entities and LGD data should have same hierarchical levels
   - If entities have GP but LGD doesn't, matching will fail

3. **Check UID generation**:
   ```bash
   # Enable debug logging to see UID generation
   python main.py --entities entities.csv --codes codes.csv --output ./output --log-level DEBUG | grep "UID"
   ```

4. **Validate code completeness**:
   ```python
   import pandas as pd
   
   df = pd.read_csv('entities.csv')
   
   # Check for missing codes at each level
   print("Missing district codes:", df['district_code'].isna().sum())
   print("Missing block codes:", df['block_code'].isna().sum())
   if 'gp_code' in df.columns:
       print("Missing GP codes:", df['gp_code'].isna().sum())
   print("Missing village codes:", df['village_code'].isna().sum())
   ```

#### Issue: Performance Degradation with Hierarchical Mapping

**Symptom**: Processing much slower with hierarchical data compared to 3-level data

**Causes**:
- More levels = more code mapping operations
- Larger lookup dictionaries for hierarchical matching
- Additional validation overhead

**Solutions**:

1. **Disable unnecessary code mapping**:
   ```bash
   # If you already have all codes
   python main.py \
     --entities entities.csv \
     --codes codes.csv \
     --output ./output \
     --disable-district-code-mapping \
     --disable-block-code-mapping \
     --disable-gp-code-mapping
   ```

2. **Use chunk processing**:
   ```bash
   python main.py \
     --entities large_entities.csv \
     --codes codes.csv \
     --output ./output \
     --chunk-size 5000
   ```

3. **Reduce fuzzy matching overhead**:
   ```bash
   # Use higher thresholds (fewer candidates to check)
   python main.py \
     --entities entities.csv \
     --codes codes.csv \
     --output ./output \
     --block-fuzzy-threshold 92 \
     --gp-fuzzy-threshold 92 \
     --village-fuzzy-threshold 95
   ```

4. **Monitor performance metrics**:
   ```bash
   # Check processing rates in summary report
   grep "Processing rate" output/mapping_summary_report_*.txt
   ```

### Data Quality Warnings

The application reports data quality issues:

- **Low district code coverage**: Many entities missing district codes
  - Solution: Provide comprehensive district mapping

- **High unmatched rate**: >20% records unmatched
  - Solution: Review data quality, adjust thresholds

- **Duplicate records**: Same entity appears multiple times
  - Solution: Clean source data before processing

### Getting Help

1. **Check the log file**: Detailed error messages and context
   ```bash
   cat output/mapping_log_*.txt
   ```

2. **Enable debug logging**: More detailed information
   ```bash
   --log-level DEBUG
   ```

3. **Review summary report**: Statistics and recommendations
   ```bash
   cat output/mapping_summary_report_*.txt
   ```

## FAQ

### General Questions

**Q: What is LGD?**
A: LGD (Local Government Directory) is a repository of local government data in India, providing unique codes for administrative entities.

**Q: How accurate is the fuzzy matching?**
A: At 95% threshold, fuzzy matching is highly accurate (typically >95% precision). Lower thresholds increase recall but may reduce precision.

**Q: Can I process data for multiple states?**
A: Yes, the application handles multi-state data. Ensure your LGD codes file includes all relevant states.

**Q: How long does processing take?**
A: Depends on dataset size. Typical rates: 1,000-5,000 entities/second for exact matching, 100-500 entities/second for fuzzy matching.

### Technical Questions

**Q: What fuzzy matching algorithm is used?**
A: RapidFuzz library using Levenshtein distance-based similarity scoring.

**Q: Can I customize the matching logic?**
A: Yes, the modular architecture allows extending matching strategies. See `lgd_mapping/matching/` for implementations.

**Q: Does it support parallel processing?**
A: The application uses optimized vectorized operations. True parallel processing can be added by extending the MappingEngine class.

**Q: What happens to unmatched records?**
A: Unmatched records are saved separately with alternative match suggestions for manual review.

### Data Questions

**Q: What if my district names don't match LGD?**
A: The application automatically handles district name variations through fuzzy matching. For known variations, you can also use the `--district-mapping` option or the `district_code_mapping` configuration parameter.

**Q: Do I need to provide district codes in my input data?**
A: No! The application automatically maps district names to codes using the LGD reference data. If your entities already have district codes, they will be preserved.

**Q: How does the automatic district code mapping work?**
A: The system extracts district names from your entities, normalizes them (removes parentheses, trims spaces, converts to lowercase), and matches them against the LGD reference data using exact and fuzzy matching. See the [District Code Mapping](#district-code-mapping) section for details.

**Q: What if district code mapping fails for some districts?**
A: The application logs warnings for unmapped districts and continues processing with the successfully mapped ones. You can review the logs and use manual `district_code_mapping` for problematic districts.

**Q: Can I disable automatic district code mapping?**
A: Yes, set `enable_district_code_mapping: false` in your configuration if you want to disable this feature.

**Q: Can I map only villages without blocks?**
A: The application requires hierarchical data (district → block → village) for accurate mapping.

**Q: How do I handle special characters in names?**
A: The application handles special characters automatically. Ensure files are UTF-8 encoded.

**Q: What if I have duplicate villages in different blocks?**
A: The UID-based matching handles this correctly by including district and block in the unique identifier.

### Hierarchical Mapping Questions

**Q: What hierarchy levels does the system support?**
A: The system supports 5 administrative levels: State → District → Block → GP (Gram Panchayat) → Village. The minimum requirement is 3 levels (district, block, village). State and GP are optional and automatically detected when present.

**Q: Do I need to provide all 5 levels?**
A: No! The system automatically detects which levels are present in your data. You can use:
- 3 levels: district → block → village (minimum)
- 4 levels: district → block → GP → village
- 5 levels: state → district → block → GP → village

**Q: How does the system detect hierarchical levels?**
A: The system scans your CSV files for specific column names: `state`, `district`, `block`, `gp`, `village` (and their corresponding `_code` columns). It automatically configures matching based on detected levels.

**Q: What if my column names are different (e.g., "State Name" instead of "state")?**
A: The system requires exact column names (case-sensitive). Rename your columns to match: `state`, `district`, `block`, `gp`, `village`. You can do this in Excel or using a script before processing.

**Q: Can I mix different hierarchy depths in the same file?**
A: No, all records in a file should have the same hierarchical structure. If you have mixed data, split it into separate files by hierarchy depth and process separately.

**Q: Do I need to provide codes for all levels?**
A: No! The system can automatically map missing codes at any level:
- State codes (if `enable_state_code_mapping` is enabled)
- District codes (enabled by default)
- Block codes (enabled by default)
- GP codes (enabled by default)
- Village codes (mapped during main matching process)

**Q: How accurate is hierarchical code mapping?**
A: Accuracy depends on data quality and fuzzy thresholds:
- Exact matches: 100% accurate
- Fuzzy matches: Typically 85-95% accurate depending on threshold
- Parent-scoped matching improves accuracy (e.g., blocks matched within correct district)

**Q: What are the benefits of using more hierarchical levels?**
A: More levels provide:
1. **Higher matching accuracy**: More context reduces ambiguity
2. **Better disambiguation**: Handles duplicate names at lower levels
3. **Improved validation**: Detects hierarchical inconsistencies
4. **Richer output**: More detailed administrative information

**Q: Will hierarchical mapping slow down processing?**
A: Slightly, but the impact is minimal:
- Each additional level adds code mapping overhead
- Use `--disable-*-code-mapping` flags if you already have all codes
- Chunk processing helps with large hierarchical datasets
- The accuracy improvement usually justifies the small performance cost

**Q: Can I use hierarchical mapping with my existing 3-level data?**
A: Yes! The system is fully backward compatible. Your existing 3-level datasets will work exactly as before without any changes.

**Q: How do I prepare data for 5-level hierarchical mapping?**
A: Follow these steps:
1. Ensure your CSV has columns: `state`, `district`, `block`, `gp`, `village`
2. Optionally add code columns: `state_code`, `district_code`, `block_code`, `gp_code`, `village_code`
3. Verify hierarchical consistency (e.g., all blocks in a district actually belong to that district)
4. Use UTF-8 encoding
5. Run with `--enable-state-code-mapping` if state codes need mapping

**Q: What happens if hierarchical validation fails?**
A: The system logs warnings about inconsistencies but continues processing. You can:
- Review warnings in the log file
- Use `--disable-hierarchy-consistency` to skip validation
- Fix data quality issues and reprocess
- Use `--disallow-partial-hierarchy` for strict validation (fails on inconsistencies)

**Q: Can I customize fuzzy thresholds for each hierarchical level?**
A: Yes! Use level-specific threshold flags:
```bash
--state-fuzzy-threshold 85
--district-fuzzy-threshold 85
--block-fuzzy-threshold 90
--gp-fuzzy-threshold 90
--village-fuzzy-threshold 95
```
Higher levels typically use lower thresholds (more variation), lower levels use higher thresholds (more standardized).

**Q: How are hierarchical UIDs generated?**
A: UIDs are created by concatenating codes from all available levels with underscores:
- 3-level: `district_code_block_code_village_code` (e.g., `362_3621_362101`)
- 4-level: `district_code_block_code_gp_code_village_code` (e.g., `362_3621_36211_362101`)
- 5-level: `state_code_district_code_block_code_gp_code_village_code` (e.g., `21_362_3621_36211_362101`)

**Q: What if my LGD reference data doesn't have GP information?**
A: The system adapts to available data:
- If entities have GP but LGD doesn't, GP information is used for code mapping but not for UID matching
- Matching falls back to available levels in both datasets
- Consider obtaining complete LGD reference data for best results

**Q: Can I process multi-state data?**
A: Yes! Include the `state` and `state_code` columns in your data and enable state code mapping:
```bash
python main.py --entities multi_state.csv --codes lgd_codes.csv --output ./results --enable-state-code-mapping
```
The system will handle state-level scoping automatically.

**Q: How do I troubleshoot low hierarchical code mapping success rates?**
A: Follow these steps:
1. Check logs for unmapped items: `grep "Could not map" output/mapping_log_*.txt`
2. Verify parent codes are available (e.g., district_code needed for block mapping)
3. Check hierarchical consistency in your data
4. Lower fuzzy thresholds for problematic levels
5. Review LGD reference data coverage
6. Enable DEBUG logging for detailed analysis

**Q: What's the difference between hierarchical mapping and district code mapping?**
A: District code mapping is a subset of hierarchical mapping:
- **District code mapping**: Maps district names to codes (one level)
- **Hierarchical mapping**: Maps codes at ALL levels (state, district, block, GP) and uses full hierarchy for matching
- Both can be enabled/disabled independently
- Hierarchical mapping includes district code mapping as one of its steps

## Project Structure

```
lgd-mapping/
├── lgd_mapping/                    # Main package
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── logging_config.py           # Logging infrastructure
│   ├── models.py                   # Data models
│   ├── exceptions.py               # Custom exceptions
│   ├── mapping_engine.py           # Core mapping orchestration
│   ├── data/                       # Data loading and validation
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── matching/                   # Matching strategies
│   │   ├── __init__.py
│   │   ├── exact_matcher.py
│   │   └── fuzzy_matcher.py
│   ├── output/                     # Output generation
│   │   ├── __init__.py
│   │   └── output_generator.py
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── uid_generator.py
│       └── district_mapper.py
├── examples/                       # Example files and documentation
│   ├── README.md
│   ├── sample_entities.csv
│   ├── sample_codes.csv
│   ├── config_template.json
│   └── district_mapping.json
├── tests/                          # Test files
│   ├── test_*.py
│   └── ...
├── main.py                         # CLI entry point
├── setup.py                        # Package setup
├── requirements.txt                # Dependencies
├── pytest.ini                      # Test configuration
└── README.md                       # This file
```

## Development

### Setting Up Development Environment

1. Clone the repository
2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
```

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=lgd_mapping --cov-report=html
```

Run specific test file:
```bash
pytest test_mapping_engine.py
```

### Code Quality

Format code:
```bash
black lgd_mapping/
```

Lint code:
```bash
flake8 lgd_mapping/
```

Type checking:
```bash
mypy lgd_mapping/
```

### Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass before submitting

### Architecture Overview

The application follows a modular architecture:

1. **Data Layer**: Loading and validation (`data/`)
2. **Matching Layer**: Exact and fuzzy matching strategies (`matching/`)
3. **Orchestration Layer**: Mapping engine coordination (`mapping_engine.py`)
4. **Output Layer**: Results generation and reporting (`output/`)
5. **Utilities**: Cross-cutting concerns (`utils/`, `config.py`, `logging_config.py`)

## License

[Add your license information here]

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-08  
**Maintained By**: [Your Team/Organization]
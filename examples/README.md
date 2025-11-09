# LGD Mapping Examples

This directory contains example files, configurations, and utilities to help you get started with the LGD Mapping application.

## Contents

- [Sample Data Files](#sample-data-files)
- [Configuration Templates](#configuration-templates)
- [Validation Script](#validation-script)
- [Usage Examples](#usage-examples)
- [Expected Output](#expected-output)

## Sample Data Files

### `sample_entities.csv`
Example input file containing entity records with district, block, and village information.

**Format:**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
Cuttack,Cuttack Sadar,Bidanasi
```

**Required Columns:**
- `district`: District name
- `block`: Block name
- `village`: Village name

**Usage:**
```bash
python main.py --entities examples/sample_entities.csv --codes examples/sample_codes.csv --output ./output
```

### `sample_codes.csv`
Example LGD codes file containing the reference data for mapping.

**Format:**
```csv
district_code,district,block_code,block,village_code,village,gp_code,gp
101,Khordha,1001,Bhubaneswar,100101,Patia,10001,Patia GP
102,Cuttack,1021,Cuttack Sadar,102101,Bidanasi,10021,Bidanasi GP
```

**Required Columns:**
- `district_code`: LGD district code (integer)
- `district`: District name
- `block_code`: LGD block code (integer)
- `block`: Block name
- `village_code`: LGD village code (integer)
- `village`: Village name

**Optional Columns:**
- `gp_code`: Gram Panchayat code
- `gp`: Gram Panchayat name

## Configuration Templates

### `config_template.json`
Basic configuration template showing all available options.

**Use for:** Understanding configuration structure and available parameters.

```json
{
  "input_entities_file": "path/to/entities.csv",
  "input_codes_file": "path/to/codes.csv",
  "output_directory": "path/to/output",
  "fuzzy_thresholds": [95, 90],
  "district_code_mapping": {},
  "chunk_size": null,
  "log_level": "INFO"
}
```

### `config_small_dataset.json`
Optimized configuration for small datasets (<10,000 records).

**Features:**
- No chunk processing (not needed)
- Standard fuzzy thresholds (95, 90)
- INFO log level for balanced output

**Use for:** Quick processing of small datasets with good data quality.

### `config_large_dataset.json`
Optimized configuration for large datasets (>50,000 records).

**Features:**
- Chunk processing enabled (10,000 records per chunk)
- WARNING log level to reduce output volume
- Memory-efficient processing

**Use for:** Processing large datasets where memory efficiency is important.

**Performance Tips Included:**
- For speed: Single threshold [95], larger chunks
- For memory: Smaller chunks (5,000)
- For accuracy: More thresholds [98, 95, 90, 85]

### `config_high_accuracy.json`
Configuration for maximum accuracy requirements.

**Features:**
- Multiple graduated thresholds (98, 95, 92, 90)
- DEBUG logging for detailed information
- No chunk processing for complete analysis

**Use for:**
- Critical administrative data
- Legal or compliance requirements
- Data quality analysis
- Initial data exploration

### `config_hierarchical.json`
Configuration for full hierarchical mapping with all administrative levels.

**Features:**
- Enables hierarchical mapping across all levels (state, district, block, GP, village)
- Automatic code mapping for district, block, and GP levels
- Level-specific fuzzy thresholds for fine-tuned matching
- Hierarchy consistency validation enabled
- Supports partial hierarchy for backward compatibility

**Use for:**
- Datasets with GP (Gram Panchayat) information
- Multi-level administrative data
- State-level datasets requiring state code mapping
- Maximum accuracy with hierarchical context

**Example:**
```bash
python main.py \
  --entities entities_with_gp.csv \
  --codes codes_full_hierarchy.csv \
  --output ./output \
  --enable-state-code-mapping \
  --gp-fuzzy-threshold 90
```

### `config_legacy_mode.json`
Configuration for legacy 3-level mapping mode (backward compatibility).

**Features:**
- Disables hierarchical mapping enhancements
- Uses only district → block → village mapping
- District code mapping only (no block/GP code mapping)
- No hierarchy consistency validation
- Compatible with older workflows

**Use for:**
- Maintaining compatibility with existing workflows
- Simple 3-level datasets without GP information
- When hierarchical features are not needed
- Testing backward compatibility

**Example:**
```bash
python main.py \
  --entities entities_3level.csv \
  --codes codes_3level.csv \
  --output ./output \
  --disable-hierarchical-mapping
```

### `district_mapping.json`
Comprehensive district name variations mapping for Odisha districts.

**Features:**
- Handles case variations (KHORDHA, Khordha, khordha)
- Handles spelling variations (Khordha/Khurda, Balasore/Baleshwar)
- Handles format variations (with/without "District" suffix)
- Includes 10+ districts with common variations

**Usage:**
```bash
python main.py --entities entities.csv --codes codes.csv --output ./output --district-mapping examples/district_mapping.json
```

**Customization:**
Add your own district variations based on your source data:
```json
{
  "district_code_mapping": {
    "Your District Name": 101,
    "Alternate Spelling": 101,
    "UPPERCASE VERSION": 101
  }
}
```

## Validation Script

### `validate_input.py`
Python script to validate input CSV files before processing.

**Features:**
- Checks file existence and readability
- Validates required columns are present
- Checks data types for numeric columns
- Identifies null values in required fields
- Detects duplicate records
- Provides detailed error and warning messages

**Usage:**
```bash
python examples/validate_input.py --entities entities.csv --codes codes.csv
```

**With verbose output:**
```bash
python examples/validate_input.py --entities entities.csv --codes codes.csv --verbose
```

**Exit Codes:**
- `0`: Validation passed (no errors)
- `1`: Validation failed (errors found)

**Example Output:**
```
LGD Mapping Input Validation
============================================================

============================================================
VALIDATING ENTITIES FILE
============================================================
✓ Checking Entities file: entities.csv
✓ Successfully read 1000 rows
✓ All required columns present
✓ No null values in required columns
✓ No duplicate records found

============================================================
VALIDATING LGD CODES FILE
============================================================
✓ Checking Codes file: codes.csv
✓ Successfully read 5000 rows
✓ All required columns present
✓ Column 'district_code' has valid numeric values
⚠ Missing optional columns: gp_code, gp

============================================================
VALIDATION SUMMARY
============================================================

✓ No critical errors found.
Warnings indicate potential data quality issues but won't prevent processing.
```

## Hierarchical Mapping Examples

The application supports flexible hierarchical mapping from 3 to 5 administrative levels. This section provides detailed examples for each hierarchy configuration.

### Understanding Hierarchy Levels

| Level | Column Names | Required | Description |
|-------|--------------|----------|-------------|
| State | `state`, `state_code` | No | State-level (for multi-state datasets) |
| District | `district`, `district_code` | Yes | District-level |
| Block | `block`, `block_code` | Yes | Block/Subdistrict level |
| GP | `gp`, `gp_code` | No | Gram Panchayat level |
| Village | `village`, `village_code` | Yes | Village-level |

### Example H1: 3-Level Hierarchy (Basic)

**Use case:** Traditional datasets with district, block, and village only.

**Entities file (entities_3level.csv):**
```csv
district,block,village
Khordha,Bhubaneswar,Patia
Khordha,Bhubaneswar,Sundarpada
Cuttack,Cuttack Sadar,Bidanasi
Puri,Puri Sadar,Penthakata
```

**LGD codes file (codes_3level.csv):**
```csv
district_code,district,block_code,block,village_code,village
362,Khordha,3621,Bhubaneswar,362101,Patia
362,Khordha,3621,Bhubaneswar,362102,Sundarpada
363,Cuttack,3631,Cuttack Sadar,363101,Bidanasi
364,Puri,3641,Puri Sadar,364101,Penthakata
```

**Command:**
```bash
python main.py \
  --entities examples/entities_3level.csv \
  --codes examples/codes_3level.csv \
  --output ./output_3level
```

**What happens:**
- System detects 3-level hierarchy: district → block → village
- Generates 3-level UIDs: `362_3621_362101`
- Maps district codes automatically (if missing)
- Standard matching process
- Fully backward compatible

**Expected log output:**
```
INFO: Detected hierarchy levels: ['district', 'block', 'village']
INFO: Hierarchy depth: 3
INFO: District code mapping completed: 4/4 districts mapped (100.0% success rate)
```

---

### Example H2: 4-Level Hierarchy (with GP)

**Use case:** Datasets with Gram Panchayat information for improved accuracy.

**Entities file (entities_4level.csv):**
```csv
district,block,gp,village
Khordha,Bhubaneswar,Patia GP,Patia
Khordha,Bhubaneswar,Patia GP,Patia Sasan
Khordha,Bhubaneswar,Sundarpada GP,Sundarpada
Cuttack,Cuttack Sadar,Bidanasi GP,Bidanasi
```

**LGD codes file (codes_4level.csv):**
```csv
district_code,district,block_code,block,gp_code,gp,village_code,village
362,Khordha,3621,Bhubaneswar,36211,Patia GP,362101,Patia
362,Khordha,3621,Bhubaneswar,36211,Patia GP,362102,Patia Sasan
362,Khordha,3621,Bhubaneswar,36212,Sundarpada GP,362103,Sundarpada
363,Cuttack,3631,Cuttack Sadar,36311,Bidanasi GP,363101,Bidanasi
```

**Command:**
```bash
python main.py \
  --entities examples/entities_4level.csv \
  --codes examples/codes_4level.csv \
  --output ./output_4level \
  --gp-fuzzy-threshold 90
```

**What happens:**
- System detects 4-level hierarchy: district → block → GP → village
- Generates 4-level UIDs: `362_3621_36211_362101`
- Maps district, block, and GP codes automatically (if missing)
- GP context improves matching accuracy for villages with same names
- Hierarchical validation ensures GP belongs to correct block

**Expected log output:**
```
INFO: Detected hierarchy levels: ['district', 'block', 'gp', 'village']
INFO: Hierarchy depth: 4
INFO: District code mapping completed: 2/2 districts mapped (100.0% success rate)
INFO: Block code mapping completed: 2/2 blocks mapped (100.0% success rate)
INFO: GP code mapping completed: 3/3 GPs mapped (100.0% success rate)
```

**Benefits:**
- Disambiguates villages with same names in different GPs
- Higher exact match rates due to GP-level UIDs
- Better hierarchical validation

---

### Example H3: 5-Level Hierarchy (Full)

**Use case:** Multi-state datasets requiring state-level scoping.

**Entities file (entities_5level.csv):**
```csv
state,district,block,gp,village
Odisha,Khordha,Bhubaneswar,Patia GP,Patia
Odisha,Cuttack,Cuttack Sadar,Bidanasi GP,Bidanasi
West Bengal,Kolkata,Kolkata,Park Street GP,Park Street
West Bengal,Howrah,Howrah,Shibpur GP,Shibpur
```

**LGD codes file (codes_5level.csv):**
```csv
state_code,state,district_code,district,block_code,block,gp_code,gp,village_code,village
21,Odisha,362,Khordha,3621,Bhubaneswar,36211,Patia GP,362101,Patia
21,Odisha,363,Cuttack,3631,Cuttack Sadar,36311,Bidanasi GP,363101,Bidanasi
19,West Bengal,701,Kolkata,7011,Kolkata,70111,Park Street GP,701101,Park Street
19,West Bengal,702,Howrah,7021,Howrah,70211,Shibpur GP,702101,Shibpur
```

**Command:**
```bash
python main.py \
  --entities examples/entities_5level.csv \
  --codes examples/codes_5level.csv \
  --output ./output_5level \
  --enable-state-code-mapping \
  --state-fuzzy-threshold 85 \
  --gp-fuzzy-threshold 90
```

**What happens:**
- System detects 5-level hierarchy: state → district → block → GP → village
- Generates 5-level UIDs: `21_362_3621_36211_362101`
- Maps codes at all levels (state, district, block, GP)
- State-level scoping prevents cross-state matching errors
- Maximum hierarchical context for highest accuracy

**Expected log output:**
```
INFO: Detected hierarchy levels: ['state', 'district', 'block', 'gp', 'village']
INFO: Hierarchy depth: 5
INFO: State code mapping completed: 2/2 states mapped (100.0% success rate)
INFO: District code mapping completed: 4/4 districts mapped (100.0% success rate)
INFO: Block code mapping completed: 4/4 blocks mapped (100.0% success rate)
INFO: GP code mapping completed: 4/4 GPs mapped (100.0% success rate)
```

**Benefits:**
- Handles multi-state datasets correctly
- Prevents matching villages from different states
- Complete administrative hierarchy
- Ideal for national-level datasets

---

### Example H4: Partial Codes with Automatic Mapping

**Use case:** Data with some codes missing, relying on automatic code mapping.

**Entities file (entities_partial_codes.csv):**
```csv
district,district_code,block,gp,village
Khordha,362,Bhubaneswar,,Patia
Khordha,362,Bhubaneswar,,Sundarpada
Cuttack,,Cuttack Sadar,Bidanasi GP,Bidanasi
```

**Command:**
```bash
python main.py \
  --entities examples/entities_partial_codes.csv \
  --codes examples/codes_4level.csv \
  --output ./output_partial \
  --enable-district-code-mapping \
  --enable-block-code-mapping \
  --enable-gp-code-mapping \
  --block-fuzzy-threshold 88 \
  --gp-fuzzy-threshold 88
```

**What happens:**
- District code for "Cuttack" is mapped automatically
- Block codes are mapped for all blocks
- GP codes are mapped for all GPs (including empty GP for first two records)
- System handles missing data gracefully
- Generates UIDs after code mapping completes

**Expected log output:**
```
INFO: District code mapping completed: 1/1 districts mapped (100.0% success rate)
INFO: Block code mapping completed: 2/2 blocks mapped (100.0% success rate)
INFO: GP code mapping completed: 2/2 GPs mapped (100.0% success rate)
WARNING: Some records have missing GP information
```

---

### Example H5: Custom Thresholds per Level

**Use case:** Fine-tuning matching sensitivity at each hierarchical level.

**Command:**
```bash
python main.py \
  --entities examples/entities_4level.csv \
  --codes examples/codes_4level.csv \
  --output ./output_custom \
  --state-fuzzy-threshold 80 \
  --district-fuzzy-threshold 85 \
  --block-fuzzy-threshold 88 \
  --gp-fuzzy-threshold 90 \
  --village-fuzzy-threshold 95
```

**Rationale:**
- **State (80)**: State names have more variations (e.g., "Orissa" vs "Odisha")
- **District (85)**: District names have moderate variations
- **Block (88)**: Block names are more standardized
- **GP (90)**: GP names are fairly consistent
- **Village (95)**: Village names should match closely

**Use when:**
- You know data quality varies by level
- Higher levels have more name variations
- Lower levels are more standardized
- You want to optimize accuracy vs. recall per level

---

### Example H6: Strict Hierarchical Validation

**Use case:** Data quality auditing with strict validation.

**Command:**
```bash
python main.py \
  --entities examples/entities_4level.csv \
  --codes examples/codes_4level.csv \
  --output ./output_strict \
  --disallow-partial-hierarchy \
  --enforce-hierarchy-consistency \
  --log-level DEBUG
```

**What happens:**
- Requires all detected levels to be present in every record
- Validates hierarchical relationships (e.g., GP belongs to correct block)
- Fails fast on inconsistencies
- Provides detailed validation errors
- Useful for identifying data quality issues

**Expected behavior:**
- Errors if any record is missing a hierarchical level
- Warnings for hierarchical inconsistencies
- Detailed DEBUG logs showing validation checks
- Processing stops if critical validation fails

---

### Example H7: Legacy Mode (Disable Hierarchical Features)

**Use case:** Maintaining compatibility with older workflows or simple 3-level mapping.

**Command:**
```bash
python main.py \
  --entities examples/entities_3level.csv \
  --codes examples/codes_3level.csv \
  --output ./output_legacy \
  --disable-hierarchical-mapping \
  --disable-block-code-mapping \
  --disable-gp-code-mapping
```

**What happens:**
- Uses only basic 3-level mapping (district → block → village)
- Skips hierarchical detection
- No automatic block or GP code mapping
- Only district code mapping enabled
- Mimics behavior of older versions

**Use when:**
- Testing backward compatibility
- Simple datasets without GP information
- Hierarchical features not needed
- Maintaining existing workflows

---

### Comparing Hierarchy Levels

| Feature | 3-Level | 4-Level (GP) | 5-Level (State) |
|---------|---------|--------------|-----------------|
| **UID Format** | `D_B_V` | `D_B_G_V` | `S_D_B_G_V` |
| **UID Example** | `362_3621_362101` | `362_3621_36211_362101` | `21_362_3621_36211_362101` |
| **Exact Match Rate** | Good | Better | Best |
| **Disambiguation** | Basic | Good | Excellent |
| **Multi-State Support** | Limited | Limited | Full |
| **Processing Time** | Fastest | Fast | Moderate |
| **Data Requirements** | Minimal | Moderate | Complete |
| **Use Case** | Simple datasets | Regional data | National data |

### Hierarchy Detection Logs

Understanding what the system detects:

**3-Level Detection:**
```
INFO: Detected hierarchy levels: ['district', 'block', 'village']
INFO: Hierarchy depth: 3
INFO: Using 3-level UID format: district_code_block_code_village_code
```

**4-Level Detection:**
```
INFO: Detected hierarchy levels: ['district', 'block', 'gp', 'village']
INFO: Hierarchy depth: 4
INFO: Using 4-level UID format: district_code_block_code_gp_code_village_code
```

**5-Level Detection:**
```
INFO: Detected hierarchy levels: ['state', 'district', 'block', 'gp', 'village']
INFO: Hierarchy depth: 5
INFO: Using 5-level UID format: state_code_district_code_block_code_gp_code_village_code
```

## Usage Examples

### Example 1: Basic Usage with Sample Data
```bash
python main.py --entities examples/sample_entities.csv --codes examples/sample_codes.csv --output ./test_output
```

### Example 2: Validate Before Processing
```bash
# First validate
python examples/validate_input.py --entities my_entities.csv --codes my_codes.csv

# If validation passes, run mapping
python main.py --entities my_entities.csv --codes my_codes.csv --output ./output
```

### Example 3: Using District Mapping
```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./output \
  --district-mapping examples/district_mapping.json
```

### Example 4: High Accuracy Processing
```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./output \
  --thresholds 98 95 92 90 \
  --log-level DEBUG
```

### Example 5: Large Dataset Processing
```bash
python main.py \
  --entities large_entities.csv \
  --codes codes.csv \
  --output ./output \
  --chunk-size 10000 \
  --log-level WARNING
```

### Example 6: Hierarchical Mapping with GP Data
```bash
python main.py \
  --entities entities_with_gp.csv \
  --codes codes_full_hierarchy.csv \
  --output ./output \
  --enable-hierarchical-mapping \
  --gp-fuzzy-threshold 90 \
  --block-fuzzy-threshold 90
```

### Example 7: Custom Hierarchical Thresholds
```bash
python main.py \
  --entities entities.csv \
  --codes codes.csv \
  --output ./output \
  --state-fuzzy-threshold 85 \
  --district-fuzzy-threshold 85 \
  --block-fuzzy-threshold 88 \
  --gp-fuzzy-threshold 88 \
  --village-fuzzy-threshold 92
```

### Example 8: Legacy 3-Level Mode
```bash
python main.py \
  --entities entities_3level.csv \
  --codes codes_3level.csv \
  --output ./output \
  --disable-hierarchical-mapping \
  --disable-block-code-mapping \
  --disable-gp-code-mapping
```

### Example 9: Using Configuration File (Python)
```python
import json
from lgd_mapping.config import MappingConfig
from lgd_mapping.mapping_engine import MappingEngine
from lgd_mapping.logging_config import setup_logging

# Load configuration from JSON
with open('examples/config_large_dataset.json', 'r') as f:
    config_dict = json.load(f)

# Update with actual file paths
config_dict.update({
    'input_entities_file': 'my_entities.csv',
    'input_codes_file': 'my_codes.csv',
    'output_directory': './output'
})

# Create configuration and run
config = MappingConfig.from_dict(config_dict)
logger = setup_logging(config)
engine = MappingEngine(config, logger)
results, stats = engine.run_complete_mapping()
```

## Expected Output

When you run the mapping with the sample data, you should get the following files in your output directory:

### Match Result Files

1. **exact_matches_YYYYMMDD_HHMMSS.csv**
   - Records with exact UID matches
   - Highest confidence matches
   - Typically 60-80% of records

2. **fuzzy_95_matches_YYYYMMDD_HHMMSS.csv**
   - Records matched with 95% similarity threshold
   - High confidence fuzzy matches
   - Typically 10-20% of records

3. **fuzzy_90_matches_YYYYMMDD_HHMMSS.csv**
   - Records matched with 90% similarity threshold
   - Medium confidence fuzzy matches
   - Typically 5-10% of records

4. **unmatched_with_alternatives_YYYYMMDD_HHMMSS.csv**
   - Unmatched records with suggested alternatives
   - Requires manual review
   - Contains alternative match suggestions

5. **unmatched_no_alternatives_YYYYMMDD_HHMMSS.csv**
   - Records with no viable matches found
   - May need data correction or additional reference data

### Report Files

6. **mapping_summary_report_YYYYMMDD_HHMMSS.txt**
   - Comprehensive statistics and quality metrics
   - Match rates by strategy
   - Processing time and performance metrics
   - Data quality indicators
   - Recommendations for improvement

7. **mapping_log_YYYYMMDD_HHMMSS.txt**
   - Detailed processing log with timestamps
   - Progress tracking for each phase
   - Warnings and error messages
   - Diagnostic information

### Sample Output Structure
```
output/
├── exact_matches_20251108_143022.csv
├── fuzzy_95_matches_20251108_143022.csv
├── fuzzy_90_matches_20251108_143022.csv
├── unmatched_with_alternatives_20251108_143022.csv
├── unmatched_no_alternatives_20251108_143022.csv
├── mapping_summary_report_20251108_143022.txt
└── mapping_log_20251108_143022.txt
```

## Quick Reference

### File Format Requirements

**Entities File:**
- Format: CSV with headers
- Encoding: UTF-8
- Required columns: district, block, village
- No specific column order required

**Codes File:**
- Format: CSV with headers
- Encoding: UTF-8
- Required columns: district_code, district, block_code, block, village_code, village
- Optional columns: gp_code, gp
- Numeric columns must contain valid integers

### Common Workflows

1. **First Time Setup:**
   - Use `sample_entities.csv` and `sample_codes.csv` to test
   - Validate your actual data with `validate_input.py`
   - Start with `config_small_dataset.json` settings

2. **Production Processing:**
   - Validate input files first
   - Use appropriate configuration template
   - Review unmatched records
   - Create district mapping for variations
   - Re-run with refined configuration

3. **Troubleshooting:**
   - Check validation script output
   - Review mapping log file
   - Examine unmatched records with alternatives
   - Adjust fuzzy thresholds
   - Add district mappings

### Performance Guidelines

| Dataset Size | Chunk Size | Thresholds | Log Level | Expected Time |
|--------------|------------|------------|-----------|---------------|
| < 10K        | Auto       | 95, 90     | INFO      | < 1 min       |
| 10K-50K      | Auto       | 95, 90     | INFO      | 1-5 min       |
| 50K-100K     | 5,000      | 95, 90     | WARNING   | 5-15 min      |
| 100K+        | 10,000     | 95         | WARNING   | 15+ min       |

## Getting Help

If you encounter issues:

1. **Run validation script first:**
   ```bash
   python examples/validate_input.py --entities your_file.csv --codes codes.csv --verbose
   ```

2. **Check the main README:** See `../README.md` for comprehensive troubleshooting

3. **Review log files:** Check `mapping_log_*.txt` for detailed error messages

4. **Enable debug logging:**
   ```bash
   python main.py --entities entities.csv --codes codes.csv --output ./output --log-level DEBUG
   ```
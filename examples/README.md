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

### Example 6: Using Configuration File (Python)
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
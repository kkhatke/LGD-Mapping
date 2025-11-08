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

- **Hierarchical Mapping**: Maps district → block → village in a structured manner
- **Multiple Matching Strategies**: Exact UID matching followed by fuzzy matching at configurable thresholds
- **Data Quality Validation**: Comprehensive validation and quality checks throughout the process
- **Performance Optimization**: Automatic optimization for large datasets with chunk processing
- **Detailed Reporting**: Comprehensive statistics, quality metrics, and alternative match suggestions

## Features

- ✅ **Modular Architecture**: Clean separation of concerns with well-defined components
- ✅ **Automatic District Code Mapping**: Intelligently maps district names to codes using exact and fuzzy matching
- ✅ **Exact Matching**: Fast UID-based matching for direct correspondences
- ✅ **Fuzzy Matching**: Configurable threshold-based fuzzy matching using RapidFuzz
- ✅ **Progress Tracking**: Real-time progress bars and detailed logging
- ✅ **Error Handling**: Robust error handling with graceful degradation
- ✅ **Memory Optimization**: Automatic chunk processing for large datasets
- ✅ **Quality Metrics**: Detailed match statistics and data quality indicators
- ✅ **Alternative Suggestions**: Provides alternative matches for manual review
- ✅ **Organized Output**: Separate files for each matching strategy with timestamps

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
```bash
python main.py --entities examples/sample_entities.csv --codes examples/sample_codes.csv --output ./test_output
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

#### Entities File (entities.csv)

Required columns:
- `district`: District name
- `block`: Block name
- `village`: Village name

Optional columns:
- `district_code`: District code (automatically mapped if not provided)

Example without district codes (recommended - let the system map them):
```csv
district,block,village
Khordha,Bhubaneswar,Patia
Cuttack,Cuttack Sadar,Bidanasi
Puri,Puri Sadar,Penthakata
```

Example with district codes (will be preserved):
```csv
district,district_code,block,village
Khordha,362,Bhubaneswar,Patia
Cuttack,363,Cuttack Sadar,Bidanasi
Puri,364,Puri Sadar,Penthakata
```

#### LGD Codes File (codes.csv)

Required columns:
- `district_code`: LGD district code (integer)
- `district`: District name
- `block_code`: LGD block code (integer)
- `block`: Block name
- `village_code`: LGD village code (integer)
- `village`: Village name

Optional columns:
- `gp_code`: Gram Panchayat code
- `gp`: Gram Panchayat name

Example:
```csv
district_code,district,block_code,block,village_code,village,gp_code,gp
101,Khordha,1001,Bhubaneswar,100101,Patia,10001,Patia GP
102,Cuttack,1021,Cuttack Sadar,102101,Bidanasi,10021,Bidanasi GP
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
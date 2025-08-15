# Boltz-2 Benchmark Repository

A comprehensive data management and analysis pipeline for protein-ligand affinity prediction using the [Boltz-2](https://github.com/jwohlwend/boltz) deep learning model. This repository provides tools to process, organize, and analyze Boltz-2 prediction results for various protein-drug combinations and mutations.

**Note**: This repository is a benchmark implementation using Boltz-2. For the official Boltz-2 model and latest updates, please refer to the [official Boltz repository](https://github.com/jwohlwend/boltz).

## What This Pipeline Actually Does

This is **NOT just a simple Boltz-2 runner**. It's a comprehensive data management system that:

1. **Processes Boltz-2 JSON outputs** into structured CSV metadata
2. **Calculates derived values** including IC50 conversions and statistical means
3. **Organizes data** for multiple protein-drug combinations and mutations
4. **Provides analysis tools** for comparing affinity predictions across different conditions
5. **Maintains data integrity** with proper error handling and validation

## How the Pipeline Works

The pipeline follows this workflow:

1. **Input Preparation**: YAML files define protein sequences and ligand SMILES strings
2. **Boltz-2 Execution**: Run Boltz-2 predictions (external to this repository)
3. **Data Processing**: Extract affinity values from JSON outputs and calculate derived metrics
4. **Metadata Generation**: Create structured CSV files with all affinity data and calculations
5. **Data Analysis**: Provide tools to analyze and visualize the processed results

## Repository Structure

```
Boltz-2_-Benchmark/
├── affinity_inputs/           # YAML input files for different protein-drug combinations
│   ├── EGFR/                  # EGFR protein with various drugs (afatinib, dacomitinib, etc.)
│   │   ├── afatinib/         # Contains EGFR-FULL_afatinib.yaml, EGFR-TRUNC110_afatinib.yaml, etc.
│   │   ├── erlotinib/        # Contains EGFR-FULL_erlotinib.yaml, EGFR-TRUNC110_erlotinib.yaml, etc.
│   │   └── ...               # Other drugs
│   ├── EGFR_mutation_790-810/ # EGFR mutations in positions 790-810
│   │   ├── erlotinib/        # Contains EGFR-T790A.yaml, EGFR-Y801H.yaml, etc.
│   │   ├── gefitinib/        # Contains mutation files for gefitinib
│   │   └── osimertinib/      # Contains mutation files for osimertinib
│   └── VEGFR2/               # VEGFR2 protein with various drugs
├── affinity_metadata/         # CSV files with processed affinity data
├── affinity_plots/            # Generated visualization plots
├── boltz_affinity_output/     # Boltz-2 model outputs and results
├── boltz_log/                 # Log files from Boltz-2 runs
├── boltz_mutation_logs/       # Mutation-specific log files
├── add_affinity_values.py     # Core data processing script for regular proteins
└── add_mutation_affinity_values.py # Core data processing script for mutations
```

## Supported Proteins and Drugs

### EGFR (Epidermal Growth Factor Receptor)
- **Drugs**: Afatinib, Dacomitinib, Erlotinib, Gefitinib, Osimertinib, Sorafenib
- **Mutations**: Positions 790-810 (including T790A, Y801H, V802Y, etc.)
- **Truncations**: Various C-terminal truncations (TRUNC110, TRUNC130, TRUNC150, etc.)

### VEGFR2 (Vascular Endothelial Growth Factor Receptor 2)
- **Drugs**: Gefitinib, Sorafenib
- **Truncations**: Various C-terminal truncations

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Boltz-2 installation (see [official installation guide](https://github.com/jwohlwend/boltz#installation))

### Dependencies
```bash
# Install Boltz-2 (recommended method)
pip install boltz[cuda] -U

# Or install from source
git clone https://github.com/jwohlwend/boltz.git
cd boltz; pip install -e .[cuda]

# Install additional Python packages
pip install pandas numpy matplotlib seaborn pathlib pyyaml
```

## How to Use the Pipeline

### 1. Running Boltz-2 Predictions

First, run Boltz-2 predictions for your protein-drug combinations:

```bash
# Example: Run Boltz-2 for EGFR with afatinib
boltz predict affinity_inputs/EGFR/afatinib \
  --out_dir boltz_affinity_output/EGFR_afatinib \
  --use_msa_server \
  --sampling_steps_affinity 300 \
  --diffusion_samples_affinity 10 \
  --affinity_mw_correction \
  --num_workers 16 \
  --max_parallel_samples 10 \
  --override
```

### 2. Processing Results into Metadata

After Boltz-2 completes, use the Python scripts to process the JSON outputs into structured CSV metadata:

```bash
# Format: python script.py <protein> <drug>
python add_affinity_values.py EGFR afatinib

# Format: python script.py <protein> <drug> <mutation_range>
python add_mutation_affinity_values.py EGFR erlotinib 790-810

# Process all available combinations
python add_affinity_values.py --all
python add_mutation_affinity_values.py --all
```

### 3. What Gets Generated

The scripts create comprehensive CSV metadata files with:

- **Raw affinity values**: 3 prediction values and 3 probability values from Boltz-2
- **Calculated means**: Statistical averages of predictions and probabilities
- **IC50 conversions**: Derived IC50 values using the formula `(6 - y) * 1.364`
- **IC50 means**: Statistical averages of IC50 predictions
- **Protein information**: Sequences, truncations, SMILES strings, file paths

### 4. Creating Input Files for New Proteins/Drugs

#### For Regular Proteins (Non-mutations)

Create a new directory structure:
```bash
mkdir -p affinity_inputs/YOUR_PROTEIN/YOUR_DRUG
```

Create YAML files for each protein variant. Example template:

```yaml
# affinity_inputs/YOUR_PROTEIN/YOUR_DRUG/YOUR_PROTEIN-FULL_YOUR_DRUG.yaml
version: 1
sequences:
  - protein:
      id: YOUR_PROTEIN
      sequence: YOUR_PROTEIN_SEQUENCE_HERE
  - ligand:
      id: A
      smiles: YOUR_DRUG_SMILES_STRING_HERE
options:
  structure: false
  affinity: true
properties:
  - affinity:
      binder: A
```

#### For Protein Mutations

Create a new directory structure:
```bash
mkdir -p affinity_inputs/YOUR_PROTEIN_mutation_790-810/YOUR_DRUG
```

Create YAML files for each mutation. Example template:

```yaml
# affinity_inputs/YOUR_PROTEIN_mutation_790-810/YOUR_DRUG/YOUR_PROTEIN-T790A_YOUR_DRUG.yaml
version: 1
sequences:
  - protein:
      id: YOUR_PROTEIN
      sequence: YOUR_MUTATED_PROTEIN_SEQUENCE_HERE
  - ligand:
      id: A
      smiles: YOUR_DRUG_SMILES_STRING_HERE
options:
  structure: false
  affinity: true
properties:
  - affinity:
      binder: A
```

## Code Examples

### Running Boltz-2 for Different Proteins/Drugs

```python
import subprocess
import os

def run_boltz_for_combination(protein_name, drug_name):
    """Run Boltz-2 for a specific protein-drug combination"""
    
    # Create input directory path
    input_dir = f"affinity_inputs/{protein_name}/{drug_name}"
    
    # Create output directory path
    output_dir = f"boltz_affinity_output/{protein_name}_{drug_name}"
    
    # Run Boltz-2 command
    cmd = [
        "boltz", "predict", input_dir,
        "--out_dir", output_dir,
        "--use_msa_server",
        "--sampling_steps_affinity", "300",
        "--diffusion_samples_affinity", "10",
        "--affinity_mw_correction",
        "--num_workers", "16",
        "--max_parallel_samples", "10",
        "--override"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully ran Boltz-2 for {protein_name} + {drug_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to run Boltz-2 for {protein_name} + {drug_name}: {e}")
        return False

# Example usage
combinations = [
    ("EGFR", "afatinib"),
    ("EGFR", "erlotinib"),
    ("VEGFR2", "sorafenib")
]

for protein, drug in combinations:
    run_boltz_for_combination(protein, drug)
```

### Analyzing Results

The scripts can be called directly from the command line with arguments:

```bash
# Format: python script.py <protein> <drug>
python add_affinity_values.py EGFR afatinib

# Format: python script.py <protein> <drug> <mutation_range>
python add_mutation_affinity_values.py EGFR erlotinib 790-810

# Process all available combinations
python add_affinity_values.py --all
python add_mutation_affinity_values.py --all
```

For custom analysis, you can import the functions in your own scripts:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_affinity_heatmap(protein_name, drug_name):
    """Create a heatmap visualization of affinity results"""
    
    # Load metadata
    metadata_file = f"affinity_metadata/affinity_{protein_name}.csv"
    df = pd.read_csv(metadata_file)
    
    # Filter for specific drug
    drug_data = df[df['drug_name'] == drug_name]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    heatmap_data = drug_data.pivot_table(
        values='affinity_pred_mean', 
        index='truncation', 
        columns='drug_name'
    )
    
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', fmt='.2f')
    plt.title(f'{protein_name} + {drug_name} Affinity Heatmap')
    plt.tight_layout()
    plt.savefig(f'affinity_plots/{protein_name}_{drug_name}_heatmap.png', dpi=300)
    plt.show()
    
    return drug_data
```

## Input File Format

### Protein Sequence Files

Each YAML file contains:
- **Protein ID**: Unique identifier for the protein
- **Protein Sequence**: Amino acid sequence (FASTA format)
- **Ligand ID**: Unique identifier for the drug
- **SMILES String**: Chemical structure representation of the drug
- **Options**: Set `affinity: true` for affinity prediction
- **Properties**: Specify which ligand binds to the protein

### File Naming Convention

- **Regular proteins**: `{PROTEIN}-{VARIANT}_{DRUG}.yaml`
  - Example: `EGFR-FULL_afatinib.yaml`, `EGFR-TRUNC110_afatinib.yaml`
- **Mutations**: `{PROTEIN}-{MUTATION}_{DRUG}.yaml`
  - Example: `EGFR-T790A_erlotinib.yaml`, `EGFR-Y801H_gefitinib.yaml`

## Output Files

### Generated Files
- **MSA Files**: Multiple sequence alignment results (`.csv`, `.a3m`)
- **Predictions**: Boltz-2 model predictions (`.json`)
- **Logs**: Detailed execution logs (`.log`)
- **Metadata**: Processed affinity data in CSV format

### Key Outputs
- `boltz_affinity_output/{PROTEIN}_{DRUG}/`: Main results directory
- `affinity_metadata/affinity_{PROTEIN}.csv`: Processed affinity data with calculated values
- `affinity_metadata/affinity_{PROTEIN}_mutation.csv`: Mutation affinity data with calculated values

## Adding New Proteins/Drugs

### Step 1: Prepare Protein Sequences
1. Obtain the protein sequence in FASTA format
2. Create truncation variants if needed
3. Create mutation variants if needed

### Step 2: Prepare Drug Information
1. Obtain the SMILES string for the drug
2. Verify chemical structure is correct

### Step 3: Create Input Files
1. Create the directory structure
2. Generate YAML files for each protein variant
3. Follow the naming convention

### Step 4: Run the Pipeline
1. Run Boltz-2 predictions using the `boltz predict` command
2. Use the Python scripts to process results
3. Analyze the generated data

### Step 5: Analyze Results
1. Check the generated CSV files
2. Create visualizations
3. Compare with experimental data

## Troubleshooting

### Common Issues

1. **Directory Not Found**: Ensure input directories exist and follow the naming convention
2. **Missing YAML Files**: Check that all required YAML files are present
3. **Boltz-2 Errors**: Check log files in `boltz_log/` directory
4. **Memory Issues**: Reduce `--num_workers` and `--max_parallel_samples` in Boltz-2 commands

### Log Files
Check the following log files for detailed error information:
- `boltz_log/boltz-{protein}-{drug}.log`
- `boltz_mutation_logs/boltz-{protein}-{drug}-mut{range}.log`

## Support

For questions and support:
- Review the [Boltz-2 documentation](https://github.com/jwohlwend/boltz)
- Join the [Boltz Slack channel](https://github.com/jwohlwend/boltz#introduction) for community support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{boltz2_benchmark,
  title={Boltz-2 Benchmark: Protein-Ligand Affinity Prediction Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Boltz-2_-Benchmark}
}
```

## Acknowledgments

- [Boltz-2 development team](https://github.com/jwohlwend/boltz) - For the core model and methodology
- Protein structure databases
- Drug compound databases
- Open-source scientific computing community

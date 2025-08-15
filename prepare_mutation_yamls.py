import pandas as pd
import yaml
import os
from pathlib import Path
import argparse

class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

def prepare_mutation_yamls(protein_name, drug_name, mutation_start, mutation_end):
    """
    Generate Boltz2-compatible YAML files for all mutations in the specified range for a protein-drug combination.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_start (int): Start position of mutation range
        mutation_end (int): End position of mutation range
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        yaml_dir = base_dir / "affinity_inputs"
        
        # Create YAML directory structure
        mutation_range = f"{mutation_start}-{mutation_end}"
        protein_mutation_dir = yaml_dir / f"{protein_name}_mutation_{mutation_range}"
        drug_dir = protein_mutation_dir / drug_name
        drug_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the mutation CSV file
        mutation_csv = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not mutation_csv.exists():
            print(f"Mutation CSV file not found: {mutation_csv}")
            return False
        
        # Read the mutation CSV file
        df = pd.read_csv(mutation_csv)
        
        # Filter data for the specific drug and mutation range
        filtered_df = df[
            (df['drug_name'] == drug_name) & 
            (df['mutation_range'] == mutation_range)
        ]
        
        if filtered_df.empty:
            print(f"No data found for {protein_name} + {drug_name} in range {mutation_range}")
            return False
        
        print(f"Found {len(filtered_df)} mutations for {protein_name} + {drug_name} in range {mutation_range}")
        
        # Generate YAML files for each mutation
        yaml_count = 0
        
        for index, row in filtered_df.iterrows():
            # Get mutation details
            protein_name_mut = row['protein_name']  # This contains the mutated name like EGFR-A791M
            sequence = row['protein_sequence']
            smiles = row['smiles']
            sequence_length = row['sequence_length']
            
            # For YAML content, use the base protein name (without mutation suffix)
            # Extract base protein name (e.g., "EGFR" from "EGFR-A791M")
            if '-' in protein_name_mut:
                base_protein_name = protein_name_mut.split('-')[0]
            else:
                base_protein_name = protein_name_mut
            
            # Create YAML data in Boltz2 format matching prepare_affinity_inputs structure
            # Always use 'A' as ligand ID (same as prepare_affinity_inputs behavior)
            ligand_id = 'A'
            
            yaml_data = {
                "version": 1,
                "sequences": [
                    {
                        "protein": {
                            "id": base_protein_name,  # Use base protein name in YAML content
                            "sequence": sequence
                        }
                    },
                    {
                        "ligand": {
                            "id": ligand_id,
                            "smiles": smiles
                        }
                    }
                ],
                "options": {
                    "structure": False,  # no structure output
                    "affinity": True      # output affinity prediction
                },
                "properties": [
                    {
                        "affinity": {
                            "binder": ligand_id
                        }
                    }
                ]
            }
            
            # Create YAML filename
            yaml_filename = f"{protein_name_mut}.yaml"
            yaml_path = drug_dir / yaml_filename
            
            # Write YAML file
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, Dumper=CustomDumper, default_flow_style=False, sort_keys=False, indent=2)
            
            yaml_count += 1
        
        print(f"Generated {yaml_count} YAML files")
        print(f"Directory: {drug_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error generating YAML files: {str(e)}")
        return False

def list_available_mutations():
    """
    List all available mutation datasets in the metadata directory.
    """
    try:
        metadata_dir = Path("affinity_metadata")
        
        if not metadata_dir.exists():
            print("Metadata directory not found")
            return
        
        # Find all mutation CSV files
        mutation_files = list(metadata_dir.glob("affinity_*_mutation.csv"))
        
        if not mutation_files:
            print("No mutation CSV files found")
            return
        
        print("Available mutation datasets:")
        
        for csv_file in mutation_files:
            protein_name = csv_file.stem.replace("affinity_", "").replace("_mutation", "")
            print(f"\nProtein: {protein_name}")
            
            # Read the CSV to get available drugs and ranges
            df = pd.read_csv(csv_file)
            
            # Group by drug and mutation range
            for (drug, mutation_range), group in df.groupby(['drug_name', 'mutation_range']):
                mutation_count = len(group[group['is_mutation'] == True])
                original_count = len(group[group['is_mutation'] == False])
                
                print(f"   Drug: {drug}")
                print(f"     Range: {mutation_range}")
                print(f"     Mutations: {mutation_count}")
                print(f"     Original sequences: {original_count}")
                print(f"     Total: {len(group)}")
        
    except Exception as e:
        print(f"Error listing mutations: {str(e)}")

def main():
    """Main function to run the YAML generation."""
    parser = argparse.ArgumentParser(description='Generate YAML files for mutations')
    parser.add_argument('protein', help='Protein name (e.g., EGFR)')
    parser.add_argument('drug', help='Drug name (e.g., osimertinib)')
    parser.add_argument('start', type=int, help='Start position of mutation range')
    parser.add_argument('end', type=int, help='End position of mutation range')
    parser.add_argument('--list', action='store_true', help='List available mutation datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_mutations()
        return
    
    print("Generating YAML files for mutations...")
    success = prepare_mutation_yamls(
        args.protein, 
        args.drug, 
        args.start, 
        args.end
    )
    
    if success:
        print("YAML generation completed successfully!")
        print(f"Check affinity_inputs/{args.protein}_mutation_{args.start}-{args.end}/{args.drug}/")
    else:
        print("YAML generation failed!")

if __name__ == "__main__":
    main() 
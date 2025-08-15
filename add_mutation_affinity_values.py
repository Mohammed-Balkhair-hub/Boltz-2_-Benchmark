import os
import json
import pandas as pd
from pathlib import Path

def add_mutation_affinity_values_to_metadata(protein_name, drug_name, mutation_range):
    """
    Add affinity values from mutation JSON files to the protein mutation metadata CSV file.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_range (str): Mutation range (e.g., '790-810')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        boltz_output_dir = base_dir / "boltz_affinity_output"
        metadata_dir = base_dir / "affinity_metadata"
        
        # Path to the mutation metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not metadata_file.exists():
            print(f"Mutation metadata file not found: {metadata_file}")
            return False
        
        # Path to the boltz results directory for mutations
        # Try the standard naming convention first
        boltz_results_dir = boltz_output_dir / f"{protein_name}_mut{mutation_range}" / f"boltz_results_{drug_name}" / "predictions"
        
        # If not found, try alternative naming convention with drug name in directory
        if not boltz_results_dir.exists():
            boltz_results_dir = boltz_output_dir / f"{protein_name}_{drug_name}_mut{mutation_range}" / f"boltz_results_{drug_name}" / "predictions"
        
        # If still not found, try with common typos in drug names
        if not boltz_results_dir.exists():
            # Handle common typo in osimertinib -> osimetrinib
            if drug_name == "osimertinib":
                boltz_results_dir = boltz_output_dir / f"{protein_name}_osimetrinib_mut{mutation_range}" / f"boltz_results_{drug_name}" / "predictions"
        
        if not boltz_results_dir.exists():
            print(f"Boltz mutation results directory not found: {boltz_results_dir}")
            return False
        
        # Read the existing mutation metadata CSV
        df = pd.read_csv(metadata_file)
        
        # Initialize new columns for affinity values
        affinity_columns = [
            'affinity_pred_value',
            'affinity_probability_binary', 
            'affinity_pred_value1',
            'affinity_probability_binary1',
            'affinity_pred_value2',
            'affinity_probability_binary2',
            'affinity_pred_mean',
            'affinity_probability_mean',
            'ic50_pred_value',
            'ic50_pred_value1',
            'ic50_pred_value2',
            'ic50_pred_mean'
        ]
        
        # Add new columns if they don't exist
        for col in affinity_columns:
            if col not in df.columns:
                df[col] = None
        
        # Process each row in the mutation metadata
        updated_count = 0
        missing_files = []
        
        for index, row in df.iterrows():
            mutation_id = row['mutation_id']
            drug = row['drug_name']
            
            # Skip if this row is for a different drug
            if drug != drug_name:
                continue
            
            # Skip if this row is for a different mutation range
            if row['mutation_range'] != mutation_range:
                continue
            
            # Construct the JSON file path
            # For mutations, the directory name is the protein_name + mutation_id (e.g., "EGFR-T790A")
            if mutation_id == "ORIGINAL":
                # For original sequence, directory is just "EGFR"
                json_file_path = boltz_results_dir / protein_name / f"affinity_{protein_name}.json"
            else:
                # For mutations, directory is "EGFR-T790A" format
                json_file_path = boltz_results_dir / f"{protein_name}-{mutation_id}" / f"affinity_{protein_name}-{mutation_id}.json"
            
            if json_file_path.exists():
                try:
                    # Read the JSON file
                    with open(json_file_path, 'r') as f:
                        affinity_data = json.load(f)
                    
                    # Extract the 6 affinity values
                    affinity_values = {
                        'affinity_pred_value': affinity_data.get('affinity_pred_value'),
                        'affinity_probability_binary': affinity_data.get('affinity_probability_binary'),
                        'affinity_pred_value1': affinity_data.get('affinity_pred_value1'),
                        'affinity_probability_binary1': affinity_data.get('affinity_probability_binary1'),
                        'affinity_pred_value2': affinity_data.get('affinity_pred_value2'),
                        'affinity_probability_binary2': affinity_data.get('affinity_probability_binary2')
                    }
                    
                    # Convert affinity prediction values to IC50 values using formula: (6 - y) * 1.364
                    def convert_to_ic50(affinity_value):
                        if affinity_value is not None:
                            return (6 - affinity_value) * 1.364
                        return None
                    
                    # Convert the three affinity prediction values to IC50
                    ic50_values = {
                        'ic50_pred_value': convert_to_ic50(affinity_values['affinity_pred_value']),
                        'ic50_pred_value1': convert_to_ic50(affinity_values['affinity_pred_value1']),
                        'ic50_pred_value2': convert_to_ic50(affinity_values['affinity_pred_value2'])
                    }
                    
                    # Add IC50 values to affinity_values dictionary
                    affinity_values.update(ic50_values)
                    
                    # Calculate means of the 3 prediction values and 3 probability values
                    pred_values = [
                        affinity_values['affinity_pred_value'],
                        affinity_values['affinity_pred_value1'],
                        affinity_values['affinity_pred_value2']
                    ]
                    prob_values = [
                        affinity_values['affinity_probability_binary'],
                        affinity_values['affinity_probability_binary1'],
                        affinity_values['affinity_probability_binary2']
                    ]
                    ic50_pred_values = [
                        ic50_values['ic50_pred_value'],
                        ic50_values['ic50_pred_value1'],
                        ic50_values['ic50_pred_value2']
                    ]
                    
                    # Calculate means (only if all values are not None)
                    if all(v is not None for v in pred_values):
                        affinity_values['affinity_pred_mean'] = sum(pred_values) / len(pred_values)
                    else:
                        affinity_values['affinity_pred_mean'] = None
                        
                    if all(v is not None for v in prob_values):
                        affinity_values['affinity_probability_mean'] = sum(prob_values) / len(prob_values)
                    else:
                        affinity_values['affinity_probability_mean'] = None
                    
                    # Calculate mean of IC50 values
                    if all(v is not None for v in ic50_pred_values):
                        affinity_values['ic50_pred_mean'] = sum(ic50_pred_values) / len(ic50_pred_values)
                    else:
                        affinity_values['ic50_pred_mean'] = None
                    
                    # Update the row with affinity values
                    for col, value in affinity_values.items():
                        df.at[index, col] = value
                    
                    updated_count += 1
                    
                except Exception as e:
                    print(f"Error reading JSON file {json_file_path}: {e}")
                    missing_files.append(str(json_file_path))
            else:
                missing_files.append(str(json_file_path))
        
        # Save the updated mutation metadata
        df.to_csv(metadata_file, index=False)
        print(f"Updated {updated_count} rows with affinity values")
        
        if missing_files:
            print(f"Missing {len(missing_files)} JSON files")
        
        return True
        
    except Exception as e:
        print(f"Error in add_mutation_affinity_values_to_metadata: {e}")
        return False

def process_all_available_mutation_combinations():
    """
    Automatically process all available protein-drug-mutation combinations.
    """
    base_dir = Path(".")
    boltz_output_dir = base_dir / "boltz_affinity_output"
    metadata_dir = base_dir / "affinity_metadata"
    
    if not boltz_output_dir.exists():
        print(f"Boltz output directory not found: {boltz_output_dir}")
        return
    
    # Find all protein-drug-mutation combinations
    combinations = []
    for item in boltz_output_dir.iterdir():
        if item.is_dir() and 'mut' in item.name:
            # Extract protein and mutation range from directory name (e.g., "EGFR_mut790-810")
            parts = item.name.split('_mut')
            if len(parts) == 2:
                protein_name = parts[0]
                mutation_range = parts[1]
                
                # Check for drug subdirectories
                for drug_dir in item.iterdir():
                    if drug_dir.is_dir() and drug_dir.name.startswith('boltz_results_'):
                        drug_name = drug_dir.name.replace('boltz_results_', '')
                        combinations.append((protein_name, drug_name, mutation_range))
    
    print(f"Found {len(combinations)} protein-drug-mutation combinations")
    
    # Process each combination
    successful = 0
    failed = 0
    
    for protein_name, drug_name, mutation_range in combinations:
        if add_mutation_affinity_values_to_metadata(protein_name, drug_name, mutation_range):
            successful += 1
        else:
            failed += 1
    
    print(f"Summary: {successful} successful, {failed} failed")

def main():
    """Main function to demonstrate usage"""
    import sys
    
    if len(sys.argv) > 3:
        # Command line usage: python add_mutation_affinity_values.py <protein> <drug> <mutation_range>
        protein_name = sys.argv[1]
        drug_name = sys.argv[2]
        mutation_range = sys.argv[3]
        
        success = add_mutation_affinity_values_to_metadata(protein_name, drug_name, mutation_range)
        
        if success:
            print("Successfully added mutation affinity values to metadata")
        else:
            print("Failed to add mutation affinity values to metadata")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Process all available combinations
        process_all_available_mutation_combinations()
    
    else:
        # Default example usage
        protein_name = "EGFR"
        drug_name = "osimertinib"
        mutation_range = "790-810"
        
        success = add_mutation_affinity_values_to_metadata(protein_name, drug_name, mutation_range)
        
        if success:
            print("Successfully added mutation affinity values to metadata")
        else:
            print("Failed to add mutation affinity values to metadata")
        
        print("\nUsage examples:")
        print("   python add_mutation_affinity_values.py EGFR osimertinib 790-810")
        print("   python add_mutation_affinity_values.py --all")

if __name__ == "__main__":
    main() 
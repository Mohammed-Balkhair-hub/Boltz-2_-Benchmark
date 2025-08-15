import os
import json
import pandas as pd
from pathlib import Path

def add_affinity_values_to_metadata(protein_name, drug_name):
    """
    Add affinity values from JSON files to the protein metadata CSV file.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        boltz_output_dir = base_dir / "boltz_affinity_output"
        metadata_dir = base_dir / "affinity_metadata"
        
        # Path to the metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not metadata_file.exists():
            print(f"Metadata file not found: {metadata_file}")
            return False
        
        # Path to the boltz results directory
        boltz_results_dir = boltz_output_dir / f"{protein_name}_{drug_name}" / f"boltz_results_{drug_name}" / "predictions"
        
        if not boltz_results_dir.exists():
            print(f"Boltz results directory not found: {boltz_results_dir}")
            return False
        
        # Read the existing metadata CSV
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
        
        # Process each row in the metadata
        updated_count = 0
        missing_files = []
        
        for index, row in df.iterrows():
            truncation = row['truncation']
            drug = row['drug_name']
            
            # Skip if this row is for a different drug
            if drug != drug_name:
                continue
            
            # Construct the JSON file path
            if truncation == 'FULL':
                truncation_dir = f"{protein_name}-FULL_{drug}"
            else:
                truncation_dir = f"{protein_name}-{truncation}_{drug}"
            
            json_file_path = boltz_results_dir / truncation_dir / f"affinity_{truncation_dir}.json"
            
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
        
        # Save the updated metadata
        df.to_csv(metadata_file, index=False)
        print(f"Updated {updated_count} rows with affinity values")
        
        if missing_files:
            print(f"Missing {len(missing_files)} JSON files")
        
        return True
        
    except Exception as e:
        print(f"Error in add_affinity_values_to_metadata: {e}")
        return False

def process_all_available_combinations():
    """
    Automatically process all available protein-drug combinations.
    """
    base_dir = Path(".")
    boltz_output_dir = base_dir / "boltz_affinity_output"
    metadata_dir = base_dir / "affinity_metadata"
    
    if not boltz_output_dir.exists():
        print(f"Boltz output directory not found: {boltz_output_dir}")
        return
    
    # Find all protein-drug combinations
    combinations = []
    for item in boltz_output_dir.iterdir():
        if item.is_dir() and '_' in item.name:
            parts = item.name.split('_')
            if len(parts) == 2:
                protein_name, drug_name = parts
                combinations.append((protein_name, drug_name))
    
    print(f"Found {len(combinations)} protein-drug combinations")
    
    # Process each combination
    successful = 0
    failed = 0
    
    for protein_name, drug_name in combinations:
        if add_affinity_values_to_metadata(protein_name, drug_name):
            successful += 1
        else:
            failed += 1
    
    print(f"Summary: {successful} successful, {failed} failed")

def main():
    """Main function to demonstrate usage"""
    import sys
    
    if len(sys.argv) > 2:
        # Command line usage: python add_affinity_values.py <protein> <drug>
        protein_name = sys.argv[1]
        drug_name = sys.argv[2]
        
        success = add_affinity_values_to_metadata(protein_name, drug_name)
        
        if success:
            print("Successfully added affinity values to metadata")
        else:
            print("Failed to add affinity values to metadata")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Process all available combinations
        process_all_available_combinations()
    
    else:
        # Default example usage
        protein_name = "EGFR"
        drug_name = "afatinib"
        
        success = add_affinity_values_to_metadata(protein_name, drug_name)
        
        if success:
            print("Successfully added affinity values to metadata")
        else:
            print("Failed to add affinity values to metadata")
        
        print("\nUsage examples:")
        print("   python add_affinity_values.py EGFR osimertinib")
        print("   python add_affinity_values.py --all")

if __name__ == "__main__":
    main() 
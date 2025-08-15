#!/usr/bin/env python3
"""
Automated preparation of affinity prediction inputs for Boltz-2 experiments.
This script creates YAML files and CSV metadata for protein-drug interactions.
"""

import os
import yaml
import pandas as pd
import requests
from typing import List, Dict
from collections import OrderedDict
from parse_proteins import get_smiles_for_boltz

class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

def get_smiles_for_boltz(drug_name):
    """
    Get SMILES string for a compound using ChEMBL API.
    
    Args:
        drug_name (str): Name of the compound to search for
        
    Returns:
        str: Canonical SMILES string for the compound
    """
    search_url = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        f"?pref_name__iexact={drug_name}&limit=1"
    )
    res = requests.get(search_url)
    data = res.json()

    chembl_id = data["molecules"][0]["molecule_chembl_id"]
    details = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json").json()
    smiles = details["molecule_structures"]["canonical_smiles"]

    return smiles

def create_yaml_content(protein_name: str, protein_sequence: str, drug_name: str, smiles: str) -> Dict:
    """
    Create YAML content for Boltz-2 input with numeric ligand IDs.
    
    Args:
        protein_name (str): Name of the protein
        protein_sequence (str): Protein sequence
        drug_name (str): Name of the drug (used for mapping to numeric ID)
        smiles (str): SMILES string for the drug
    
    Returns:
        Dict: YAML content structure
    """
    # Map drug names to simple letter IDs
    drug_id_map = {
        'gefitinib': 'A',
        'sorafenib': 'B'
    }
    
    # Get letter ID for the drug, default to 'A' if not found
    ligand_id = drug_id_map.get(drug_name, 'A')
    
    yaml_data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_name,
                    "sequence": protein_sequence
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
    
    return yaml_data

def write_yaml_with_formatting(yaml_content: Dict, file_path: str):
    """
    Write YAML content with exact formatting including quotes and comments.
    
    Args:
        yaml_content (Dict): The YAML content dictionary
        file_path (str): Path to write the YAML file
    """
    protein_name = yaml_content["sequences"][0]["protein"]["id"]
    protein_sequence = yaml_content["sequences"][0]["protein"]["sequence"]
    drug_name = yaml_content["sequences"][1]["ligand"]["id"]
    smiles = yaml_content["sequences"][1]["ligand"]["smiles"]
    
    # Create the exact YAML structure with proper indentation
    yaml_data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_name,
                    "sequence": protein_sequence
                }
            },
            {
                "ligand": {
                    "id": drug_name,
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
                    "binder": drug_name
                }
            }
        ]
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(yaml_data, f, Dumper=CustomDumper, default_flow_style=False, sort_keys=False, indent=2)

def prepare_affinity_inputs(proteins: List[Dict], 
                          drugs: List[Dict], 
                          output_root: str = "affinity_inputs", 
                          force_overwrite: bool = False,
                          min_length: int = 80,
                          step_size: int = 20):
    """
    Prepare structured input files for Boltz-2 affinity prediction experiments.
    
    Args:
        proteins (List[Dict]): List of protein dictionaries with 'name' and 'sequence'
        drugs (List[Dict]): List of drug dictionaries with 'name' and 'smiles'
        output_root (str): Root folder for output files
        force_overwrite (bool): Whether to overwrite existing files
        min_length (int): Minimum sequence length for truncation (default: 80)
        step_size (int): Number of amino acids to truncate each step (default: 20)
    """
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Process each protein-drug combination
    for protein in proteins:
        protein_name = protein['name']
        protein_sequence = protein['sequence']
        sequence_length = len(protein_sequence)
        
        # Initialize metadata list for this protein
        protein_metadata_rows = []
        
        # Create protein directory
        protein_dir = os.path.join(output_root, protein_name)
        os.makedirs(protein_dir, exist_ok=True)
        
        for drug in drugs:
            drug_name = drug['name']
            smiles = drug['smiles']
            
            # Create drug directory
            drug_dir = os.path.join(protein_dir, drug_name)
            os.makedirs(drug_dir, exist_ok=True)
            
            # Generate truncation lengths using correct C-terminal truncation logic
            truncation_lengths = []
            if sequence_length > min_length:
                # Cut by step_size each time until we can't cut more due to min_length
                step = 1
                while True:
                    trunc_len = sequence_length - (step * step_size)
                    if trunc_len >= min_length:
                        truncation_lengths.append(trunc_len)
                        step += 1
                    else:
                        break
            
            # Create YAML files
            yaml_files = []
            
            # Full sequence
            full_yaml_path = os.path.join(drug_dir, f"{protein_name}-FULL_{drug_name}.yaml")
            if not os.path.exists(full_yaml_path) or force_overwrite:
                yaml_content = create_yaml_content(protein_name, protein_sequence, drug_name, smiles)
                write_yaml_with_formatting(yaml_content, full_yaml_path)
                yaml_files.append(full_yaml_path)
            
            # Truncated sequences
            for trunc_len in truncation_lengths:
                truncated_sequence = protein_sequence[:trunc_len]
                trunc_yaml_path = os.path.join(drug_dir, f"{protein_name}-TRUNC{trunc_len}_{drug_name}.yaml")
                
                if not os.path.exists(trunc_yaml_path) or force_overwrite:
                    yaml_content = create_yaml_content(protein_name, truncated_sequence, drug_name, smiles)
                    write_yaml_with_formatting(yaml_content, trunc_yaml_path)
                    yaml_files.append(trunc_yaml_path)
            
            # Add metadata rows for this protein
            for yaml_path in yaml_files:
                if 'FULL' in yaml_path:
                    truncation_type = 'FULL'
                    seq_len = len(protein_sequence)
                    seq_content = protein_sequence
                else:
                    # Extract truncation length from filename
                    filename = os.path.basename(yaml_path)
                    trunc_len = int(filename.split('TRUNC')[1].split('_')[0])
                    truncation_type = f'TRUNC{trunc_len}'
                    seq_len = trunc_len
                    seq_content = protein_sequence[:trunc_len]
                
                protein_metadata_rows.append({
                    'protein_name': protein_name,
                    'truncation': truncation_type,
                    'drug_name': drug_name,
                    'smiles': smiles,
                    'sequence_length': seq_len,
                    'protein_sequence': seq_content,
                    'yaml_path': yaml_path,
                    'affinity_prediction': ''  # Placeholder for predictions
                })
        
        # Create CSV metadata file for this protein
        csv_path = f"affinity_metadata/affinity_{protein_name}.csv"
        os.makedirs("affinity_metadata", exist_ok=True)
        
        if os.path.exists(csv_path):
            # Read existing CSV and append new rows
            existing_df = pd.read_csv(csv_path)
            new_df = pd.DataFrame(protein_metadata_rows)
            
            # Remove duplicates based on yaml_path
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['yaml_path'], keep='last')
        else:
            combined_df = pd.DataFrame(protein_metadata_rows)
        
        # Save CSV for this protein
        combined_df.to_csv(csv_path, index=False)
        print(f"Created {len(protein_metadata_rows)} YAML files for {protein_name}")
    
    print(f"Output directory: {output_root}")

def get_proteins_from_csv(csv_path: str) -> List[Dict]:
    """
    Extract proteins from the CSV file used in parse_proteins.py
    
    Args:
        csv_path (str): Path to the protein CSV file
        
    Returns:
        List[Dict]: List of protein dictionaries
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    proteins = []
    
    for _, row in df.iterrows():
        proteins.append({
            'name': row['name'],
            'sequence': row['seqres']
        })
    
    return proteins

def main():
    """Main function to demonstrate the automation."""
    
    # Example usage with proteins from your dataset
    csv_file = "/home/ubuntu/SUDS-MutDTA/data/davis/alphaflow_io/input.csv"
    
    # Get proteins from CSV
    proteins = get_proteins_from_csv(csv_file)
    
    # Select specific proteins for testing (you can modify this list)
    selected_proteins = [
        protein for protein in proteins 
        if protein['name'] in ['EGFR']
    ]
    
    # Define drugs with SMILES
    drugs = [
        {'name': 'cetuximab', 'smiles': get_smiles_for_boltz('Cetuximab')},
    ]
    
    # Prepare affinity inputs
    prepare_affinity_inputs(
        proteins=selected_proteins,
        drugs=drugs,
        output_root="affinity_inputs",
        force_overwrite=True,
        min_length=80,  # Minimum sequence length
        step_size=20    # Truncate by 20 amino acids each step
    )

if __name__ == "__main__":
    main() 
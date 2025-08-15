import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Standard amino acid codes
AMINO_ACIDS = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
}

def generate_mutation_sequences(protein_name, drug_name, mutation_start, mutation_end, save_csv=True):
    """
    Generate mutated protein sequences by replacing each amino acid in the specified range
    with all 19 other amino acids.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_start (int): Start position of mutation range (1-based)
        mutation_end (int): End position of mutation range (1-based)
        save_csv (bool): Whether to save the results to CSV
    
    Returns:
        pd.DataFrame: DataFrame containing all mutated sequences
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Path to the original metadata CSV file
        original_csv = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not original_csv.exists():
            print(f"Original metadata file not found: {original_csv}")
            return None
        
        # Read the original CSV file
        df = pd.read_csv(original_csv)
        
        # Get the original protein sequence (from FULL sequence)
        original_row = df[(df['protein_name'] == protein_name) & 
                         (df['drug_name'] == drug_name) & 
                         (df['truncation'] == 'FULL')]
        
        if original_row.empty:
            print(f"No FULL sequence found for {protein_name} + {drug_name}")
            return None
        
        original_sequence = original_row.iloc[0]['protein_sequence']
        original_smiles = original_row.iloc[0]['smiles']
        sequence_length = len(original_sequence)
        
        print(f"Original sequence length: {sequence_length}")
        print(f"Mutation range: {mutation_start}-{mutation_end}")
        
        # Validate mutation range
        if mutation_start < 1 or mutation_end > sequence_length:
            print(f"Invalid mutation range. Sequence length is {sequence_length}")
            return None
        
        if mutation_start > mutation_end:
            print("Invalid range: start > end")
            return None
        
        # Convert to 0-based indexing for string operations
        start_idx = mutation_start - 1
        end_idx = mutation_end
        
        # Get the original amino acids in the mutation range
        original_range = original_sequence[start_idx:end_idx]
        print(f"Original amino acids in range: {original_range}")
        
        # Generate all mutations
        mutations_data = []
        mutation_count = 0
        
        # Add original sequence as first row
        original_data = {
            'protein_name': protein_name,
            'drug_name': drug_name,
            'mutation_position': 0,
            'original_amino_acid': '-',
            'new_amino_acid': '-',
            'mutation_id': 'ORIGINAL',
            'protein_sequence': original_sequence,
            'sequence_length': len(original_sequence),
            'smiles': original_smiles,
            'mutation_range': f"{mutation_start}-{mutation_end}",
            'is_mutation': False
        }
        mutations_data.append(original_data)
        
        for pos in range(mutation_start, mutation_end + 1):
            original_aa = original_sequence[pos - 1]  # Convert to 0-based
            
            # Generate mutations for this position
            for new_aa in AMINO_ACIDS.keys():
                if new_aa != original_aa:  # Skip the original amino acid
                    # Create mutated sequence
                    mutated_sequence = (
                        original_sequence[:pos-1] + 
                        new_aa + 
                        original_sequence[pos:]
                    )
                    
                    # Create mutation identifier
                    mutation_id = f"{original_aa}{pos}{new_aa}"
                    
                    # Create mutated protein name
                    mutated_protein_name = f"{protein_name}-{original_aa}{pos}{new_aa}"
                    
                    # Create row data
                    mutation_data = {
                        'protein_name': mutated_protein_name,
                        'drug_name': drug_name,
                        'mutation_position': pos,
                        'original_amino_acid': original_aa,
                        'new_amino_acid': new_aa,
                        'mutation_id': mutation_id,
                        'protein_sequence': mutated_sequence,
                        'sequence_length': len(mutated_sequence),
                        'smiles': original_smiles,
                        'mutation_range': f"{mutation_start}-{mutation_end}",
                        'is_mutation': True
                    }
                    
                    mutations_data.append(mutation_data)
                    mutation_count += 1
        
        # Create DataFrame
        mutations_df = pd.DataFrame(mutations_data)
        
        print(f"Generated {len(mutations_df)} sequences:")
        print(f"   Original sequence: 1")
        print(f"   Positions mutated: {mutation_end - mutation_start + 1}")
        print(f"   Mutations per position: 19")
        print(f"   Total mutations: {len(mutations_df) - 1}")
        
        # Save to CSV
        if save_csv:
            output_filename = f"affinity_{protein_name}_mutation.csv"
            output_path = metadata_dir / output_filename
            
            # Check if file already exists (same protein)
            if output_path.exists():
                print(f"Found existing mutation file for {protein_name}: {output_path}")
                existing_df = pd.read_csv(output_path)
                
                # Check if this drug already exists
                existing_drugs = existing_df[existing_df['drug_name'] == drug_name]
                if not existing_drugs.empty:
                    print(f"Drug {drug_name} already exists in the file. Overwriting...")
                    # Remove existing data for this drug
                    existing_df = existing_df[existing_df['drug_name'] != drug_name]
                
                # Append new mutations
                combined_df = pd.concat([existing_df, mutations_df], ignore_index=True)
                combined_df.to_csv(output_path, index=False)
                print(f"Updated mutations file: {output_path}")
                print(f"   Total sequences in file: {len(combined_df)}")
                print(f"   Drugs in file: {list(combined_df['drug_name'].unique())}")
            else:
                # Create new file for this protein
                mutations_df.to_csv(output_path, index=False)
                print(f"Created new mutations file for {protein_name}: {output_path}")
                print(f"   Total sequences in file: {len(mutations_df)}")
                print(f"   Drugs in file: [{drug_name}]")
        
        # Print summary statistics
        print(f"Mutation Summary:")
        print(f"   Protein: {protein_name}")
        print(f"   Drug: {drug_name}")
        print(f"   Mutation range: {mutation_start}-{mutation_end}")
        print(f"   Original amino acids: {original_range}")
        print(f"   Total mutations generated: {len(mutations_df) - 1}")
        
        return mutations_df
        
    except Exception as e:
        print(f"Error generating mutations: {str(e)}")
        return None

def analyze_mutation_impact(protein_name, drug_name, mutation_start, mutation_end):
    """
    Analyze the potential impact of mutations in the specified range.
    
    Args:
        protein_name (str): Name of the protein
        drug_name (str): Name of the drug
        mutation_start (int): Start position of mutation range
        mutation_end (int): End position of mutation range
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        
        # Path to the original metadata CSV file
        original_csv = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not original_csv.exists():
            print(f"Original metadata file not found: {original_csv}")
            return
        
        # Read the original CSV file
        df = pd.read_csv(original_csv)
        
        # Get the original protein sequence
        original_row = df[(df['protein_name'] == protein_name) & 
                         (df['drug_name'] == drug_name) & 
                         (df['truncation'] == 'FULL')]
        
        if original_row.empty:
            print(f"No FULL sequence found for {protein_name} + {drug_name}")
            return
        
        original_sequence = original_row.iloc[0]['protein_sequence']
        
        # Analyze the mutation range
        start_idx = mutation_start - 1
        end_idx = mutation_end
        mutation_range = original_sequence[start_idx:end_idx]
        
        print(f"Mutation Range Analysis:")
        print(f"   Protein: {protein_name}")
        print(f"   Drug: {drug_name}")
        print(f"   Range: {mutation_start}-{mutation_end}")
        print(f"   Amino acids in range: {mutation_range}")
        print(f"   Range length: {len(mutation_range)}")
        
        # Count amino acid types in the range
        aa_counts = {}
        for aa in mutation_range:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        print(f"Amino Acid Distribution in Range:")
        for aa, count in sorted(aa_counts.items()):
            full_name = AMINO_ACIDS.get(aa, aa)
            print(f"   {aa} ({full_name}): {count} occurrences")
        
        # Calculate total mutations
        total_positions = mutation_end - mutation_start + 1
        total_mutations = total_positions * 19
        
        print(f"Mutation Statistics:")
        print(f"   Positions to mutate: {total_positions}")
        print(f"   Mutations per position: 19")
        print(f"   Total mutations to generate: {total_mutations}")
        
    except Exception as e:
        print(f"Error analyzing mutations: {str(e)}")

def main():
    """Main function to run the mutation generation."""
    parser = argparse.ArgumentParser(description='Generate mutated protein sequences for affinity analysis')
    parser.add_argument('protein', help='Protein name (e.g., EGFR)')
    parser.add_argument('drug', help='Drug name (e.g., osimertinib)')
    parser.add_argument('start', type=int, help='Start position of mutation range (1-based)')
    parser.add_argument('end', type=int, help='End position of mutation range (1-based)')
    parser.add_argument('--analyze', action='store_true', help='Analyze mutation range before generating')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("Analyzing mutation range...")
        analyze_mutation_impact(args.protein, args.drug, args.start, args.end)
        print("\n" + "="*50 + "\n")
    
    print("Generating mutated protein sequences...")
    mutations_df = generate_mutation_sequences(
        args.protein, 
        args.drug, 
        args.start, 
        args.end
    )
    
    if mutations_df is not None:
        print("Mutation generation completed successfully!")
        print(f"Check affinity_metadata/affinity_{args.protein}_mutation.csv")
    else:
        print("Mutation generation failed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script to parse protein data from CSV file and find the longest and shortest proteins.
"""

import pandas as pd
import sys
import requests

def parse_proteins(csv_file):
    """
    Parse the CSV file and return protein data with lengths.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with protein names and their lengths
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if 'name' not in df.columns or 'seqres' not in df.columns:
            print("Error: CSV file must contain 'name' and 'seqres' columns")
            return None
            
        # Calculate sequence lengths
        df['length'] = df['seqres'].str.len()
        
        # Sort by length in descending order
        df_sorted = df.sort_values('length', ascending=False)
        
        return df_sorted
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def print_results(df):
    """
    Print the top 10 longest and bottom 10 shortest proteins.
    
    Args:
        df (pandas.DataFrame): DataFrame with protein data
    """
    if df is None or df.empty:
        print("No data to display")
        return
    
    print("=" * 80)
    print("TOP 10 LONGEST PROTEINS")
    print("=" * 80)
    print(f"{'Rank':<5} {'Protein Name':<30} {'Length':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        print(f"{i:<5} {row['name']:<30} {row['length']:<10}")
    
    print("\n" + "=" * 80)
    print("BOTTOM 10 SHORTEST PROTEINS")
    print("=" * 80)
    print(f"{'Rank':<5} {'Protein Name':<30} {'Length':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(df.tail(10).iterrows(), 1):
        print(f"{i:<5} {row['name']:<30} {row['length']:<10}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total number of proteins: {len(df)}")
    print(f"Average protein length: {df['length'].mean():.2f}")
    print(f"Median protein length: {df['length'].median():.2f}")
    print(f"Longest protein: {df.iloc[0]['name']} ({df.iloc[0]['length']} amino acids)")
    print(f"Shortest protein: {df.iloc[-1]['name']} ({df.iloc[-1]['length']} amino acids)")
    print(f"Length range: {df['length'].max() - df['length'].min()} amino acids")

def search_protein(df, protein_name):
    """
    Search for a specific protein by name and display its information.
    
    Args:
        df (pandas.DataFrame): DataFrame with protein data
        protein_name (str): Name of the protein to search for
    """
    if df is None or df.empty:
        print("No data to search")
        return
    
    # Search for the protein (case-insensitive)
    protein_found = df[df['name'].str.contains(protein_name, case=False, na=False)]
    
    if protein_found.empty:
        print(f"Protein '{protein_name}' not found in the dataset.")
        print("Available proteins containing similar names:")
        # Show proteins with similar names
        similar_proteins = df[df['name'].str.contains(protein_name[:3], case=False, na=False)]
        if not similar_proteins.empty:
            for _, row in similar_proteins.head(5).iterrows():
                print(f"  - {row['name']} ({row['length']} amino acids)")
        return
    
    print("=" * 80)
    print(f"SEARCH RESULTS FOR '{protein_name.upper()}'")
    print("=" * 80)
    
    for _, row in protein_found.iterrows():
        print(f"Protein Name: {row['name']}")
        print(f"Sequence Length: {row['length']} amino acids")
        print(f"Rank by Length: {df[df['length'] >= row['length']].shape[0]} out of {len(df)}")
        print(f"Sequence Preview: {row['seqres'][:50]}...")
        print("-" * 80)

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



def main():
    """Main function to run the protein analysis."""
    csv_file = "/home/ubuntu/SUDS-MutDTA/data/davis/alphaflow_io/input.csv"
    
    print("Parsing protein data from CSV file...")
    print(f"File: {csv_file}")
    
    # Parse the CSV file
    df = parse_proteins(csv_file)
    
    if df is not None:
        # Print the original analysis results
        print_results(df)
        
        # Search for VEGFR2 protein
        print("\n" + "=" * 80)
        print("SEARCHING FOR PROTEIN")
        print("=" * 80)
        search_protein(df, "VEGFR2")
        
        # Test SMILES retrieval
        print("\n" + "=" * 80)
        print("TESTING SMILES RETRIEVAL")
        print("=" * 80)
        try:
            smiles = get_smiles_for_boltz("Sorafenib")
            print(f"SMILES for Drug: {smiles}")
        except Exception as e:
            print(f"Error retrieving SMILES: {e}")
        
    else:
        print("Failed to parse protein data.")

if __name__ == "__main__":
    main() 
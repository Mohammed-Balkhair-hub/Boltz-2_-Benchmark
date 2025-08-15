import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def plot_affinity_heatmap(protein_name, drug_name, save_plot=True, show_plot=False):
    """
    Create a heatmap visualization of mean affinity probability values across truncations.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        save_plot (bool): Whether to save the plot as PNG file
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots"
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(exist_ok=True)
        
        # Path to the metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not metadata_file.exists():
            print(f"Metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        print(f"Loaded metadata with {len(df)} rows")
        
        # Filter data for the specific protein and drug
        filtered_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
        
        if filtered_df.empty:
            print(f"No data found for {protein_name} and {drug_name}")
            return False
        
        # Check if affinity_probability_mean column exists
        if 'affinity_probability_mean' not in filtered_df.columns:
            print("'affinity_probability_mean' column not found. Run add_affinity_values.py first.")
            return False
        
        # Remove rows where mean probability is None or NaN
        filtered_df = filtered_df.dropna(subset=['affinity_probability_mean'])
        
        if filtered_df.empty:
            print("No valid probability data found")
            return False
        
        # Sort by sequence length (descending) to have FULL first, then truncations
        filtered_df = filtered_df.sort_values('sequence_length', ascending=False)
        
        # Create the heatmap data
        truncations = filtered_df['truncation'].tolist()
        probabilities = filtered_df['affinity_probability_mean'].tolist()
        
        # Create a 1D array for heatmap (we'll reshape it)
        heatmap_data = np.array(probabilities).reshape(1, -1)
        
        # Set up the plot
        plt.figure(figsize=(max(12, len(truncations) * 0.8), 6))
        
        # Create the heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=truncations,
                   yticklabels=[f'{protein_name}\n{drug_name}'],
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu',  # Red-Yellow-Blue (blue=high probability, red=low probability)
                   cbar_kws={'label': 'Mean Affinity Probability'},
                   linewidths=0.5,
                   linecolor='white')
        
        # Customize the plot
        plt.title(f'Binding Affinity Probability Heatmap\n{protein_name} + {drug_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Protein Truncation', fontsize=12, fontweight='bold')
        plt.ylabel('Protein-Drug Combination', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add statistics as text
        min_prob = min(probabilities)
        max_prob = max(probabilities)
        avg_prob = np.mean(probabilities)
        
        stats_text = f'Min: {min_prob:.3f} | Max: {max_prob:.3f} | Avg: {avg_prob:.3f}'
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            plot_filename = f"{protein_name}_{drug_name}_affinity_heatmap.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to: {plot_path}")
        
        # Show the plot (only if explicitly requested)
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
        
        # Find best and worst truncations
        best_trunc = truncations[np.argmax(probabilities)]
        worst_trunc = truncations[np.argmin(probabilities)]
        
        return True
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        return False

def plot_protein_comparison_heatmap(protein_name, selected_drugs=None, save_plot=True, show_plot=False):
    """
    Create a heatmap comparing all drugs for a specific protein in one image.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        save_plot (bool): Whether to save the plot
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots" / protein_name
        
        # Create protein-specific plots directory
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not metadata_file.exists():
            print(f"Error: Metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Filter data for the specific protein
        filtered_df = df[df['protein_name'] == protein_name]
        
        if filtered_df.empty:
            print(f"Error: No data found for {protein_name}")
            return False
        
        # Get all available drugs for this protein
        available_drugs = filtered_df['drug_name'].unique()
        
        # Filter drugs if specified
        if selected_drugs:
            available_drugs = [drug for drug in available_drugs if drug in selected_drugs]
        
        if len(available_drugs) == 0:
            print(f"Error: No drugs found for {protein_name}")
            return False
        
        all_data = []
        labels = []
        
        for drug_name in available_drugs:
            # Filter data for this specific drug
            drug_df = filtered_df[filtered_df['drug_name'] == drug_name]
            
            # Check if affinity_probability_mean column exists
            if 'affinity_probability_mean' not in drug_df.columns:
                print(f"Warning:  Skipping {drug_name}: No affinity data")
                continue
            
            # Remove rows where mean probability is None or NaN
            drug_df = drug_df.dropna(subset=['affinity_probability_mean'])
            
            if drug_df.empty:
                print(f"Warning:  Skipping {drug_name}: No valid data")
                continue
            
            # Sort by sequence length (descending)
            drug_df = drug_df.sort_values('sequence_length', ascending=False)
            
            # Get truncations and probabilities
            truncations = drug_df['truncation'].tolist()
            probabilities = drug_df['affinity_probability_mean'].tolist()
            
            all_data.append(probabilities)
            labels.append(drug_name)
        
        if not all_data:
            print("Error: No valid data found for any drug")
            return False
        
        # Create the heatmap with smaller figure size
        plt.figure(figsize=(max(8, min(30, len(all_data[0])) * 0.6), max(4, len(all_data) * 1.0)))
        
        # Convert to numpy array and limit to columns 10-30
        heatmap_data = np.array(all_data)[:, 9:30]  # Columns 10-30 (0-indexed: 9-29)
        
        # Get truncation labels from the first dataset (columns 10-30)
        first_drug_df = filtered_df[filtered_df['drug_name'] == available_drugs[0]]
        first_drug_df = first_drug_df.sort_values('sequence_length', ascending=False)
        truncations_raw = first_drug_df['truncation'].tolist()[9:30]  # Columns 10-30 (0-indexed: 9-29)
        
        # Reverse the order so 693 -> 700 -> FULL
        heatmap_data = heatmap_data[:, ::-1]  # Reverse the data columns
        truncations_raw = truncations_raw[::-1]  # Reverse the labels
        
        # Simplify truncation labels: if TRUNC797, show as 797; if FULL, keep as FULL
        truncations = []
        for trunc in truncations_raw:
            if trunc.startswith('TRUNC'):
                truncations.append(trunc[5:])  # Remove 'TRUNC' prefix
            else:
                truncations.append(trunc)  # Keep as is (e.g., 'FULL')
        
        # Calculate column averages
        column_averages = np.mean(heatmap_data, axis=0)
        
        # Calculate differences between consecutive column averages
        column_differences = []
        max_diff_index = 0
        max_diff_value = 0
        
        for i in range(len(column_averages)):
            if i == 0:
                # For the first column, show the mean itself
                column_differences.append(column_averages[i])
            else:
                # For other columns, calculate the absolute difference from previous column
                diff = abs(column_averages[i] - column_averages[i-1])
                column_differences.append(diff)
                
                # Track the maximum difference
                if diff > max_diff_value:
                    max_diff_value = diff
                    max_diff_index = i
        
        # Create the mean probability heatmap
        plt.figure(figsize=(max(8, min(30, len(all_data[0])) * 0.6), max(4, len(all_data) * 1.0)))
        
        sns.heatmap(heatmap_data, 
                   xticklabels=truncations,
                   yticklabels=labels,
                   annot=False,  # Remove numerical values
                   cmap='RdYlBu',  # Blue=high probability, red=low probability
                   cbar=True,  # Show colorbar on the right
                   cbar_kws={'label': 'Mean Affinity Probability'},
                   linewidths=0.2,  # Thinner lines for smaller boxes
                   linecolor='white',
                   square=True)  # Make boxes square
        
        # Remove title and customize the plot with labels moved up
        plt.xlabel('Protein Truncation', fontsize=10, fontweight='bold')
        plt.ylabel('Drug', fontsize=10, fontweight='bold', labelpad=15)
        
        # Move x-axis labels to top and make horizontal
        plt.gca().xaxis.set_ticks_position('top')
        plt.xticks(rotation=0, ha='center', fontsize=8)
        plt.yticks(fontsize=8)
        
        # Adjust layout for smaller, more compact plot with colorbar at bottom
        plt.subplots_adjust(top=0.98, bottom=0.18, left=0.12, right=0.95)
        
        # Save the mean probability plot
        if save_plot:
            plot_filename = f"{protein_name}_all_drugs_affinity_heatmap.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved mean probability heatmap to: {plot_path}")
        
        # Show the plot (only if explicitly requested)
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
        
        # Create the mean value heatmap
        plt.figure(figsize=(max(8, min(30, len(all_data[0])) * 0.6), max(4, len(all_data) * 1.0)))
        
        # Prepare mean value data
        all_value_data = []
        for drug_name in available_drugs:
            # Filter data for this specific drug
            drug_df = filtered_df[filtered_df['drug_name'] == drug_name]
            
            # Check if ic50_pred_mean column exists
            if 'ic50_pred_mean' not in drug_df.columns:
                print(f"Warning:  Skipping {drug_name}: No IC50 value data")
                continue
            
            # Remove rows where mean value is None or NaN
            drug_df = drug_df.dropna(subset=['ic50_pred_mean'])
            
            if drug_df.empty:
                print(f"Warning:  Skipping {drug_name}: No valid value data")
                continue
            
            # Sort by sequence length (descending)
            drug_df = drug_df.sort_values('sequence_length', ascending=False)
            
            # Get truncations and values
            values = drug_df['ic50_pred_mean'].tolist()
            all_value_data.append(values)
        
        if not all_value_data:
            print("Error: No valid value data found for any drug")
            return False
        
        # Convert to numpy array and limit to columns 10-30
        value_heatmap_data = np.array(all_value_data)[:, 9:30]  # Columns 10-30 (0-indexed: 9-29)
        
        # Reverse the order so 693 -> 700 -> FULL
        value_heatmap_data = value_heatmap_data[:, ::-1]  # Reverse the data columns
        
        sns.heatmap(value_heatmap_data, 
                   xticklabels=truncations,
                   yticklabels=labels,
                   annot=False,  # Remove numerical values
                   cmap='RdYlBu_r',  # Reversed: Red=high value, blue=low value
                   cbar=True,  # Show colorbar on the right
                   cbar_kws={'label': 'Mean pIC50 Score'},
                   linewidths=0.2,  # Thinner lines for smaller boxes
                   linecolor='white',
                   square=True)  # Make boxes square
        
        # Remove title and customize the plot with labels moved up
        plt.xlabel('Protein Truncation', fontsize=10, fontweight='bold')
        plt.ylabel('Drug', fontsize=10, fontweight='bold', labelpad=15)
        
        # Move x-axis labels to top and make horizontal
        plt.gca().xaxis.set_ticks_position('top')
        plt.xticks(rotation=0, ha='center', fontsize=8)
        plt.yticks(fontsize=8)
        
        # Adjust layout for smaller, more compact plot with colorbar at bottom
        plt.subplots_adjust(top=0.98, bottom=0.18, left=0.12, right=0.95)
        
        # Save the mean value plot
        if save_plot:
            plot_filename = f"{protein_name}_all_drugs_value_heatmap.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved mean value heatmap to: {plot_path}")
        
        # Show the plot (only if explicitly requested)
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating comparison heatmap: {str(e)}")
        return False

def plot_multiple_proteins_heatmap(protein_drug_pairs, save_plot=True, show_plot=False):
    """
    Create a heatmap comparing multiple protein-drug combinations.
    
    Args:
        protein_drug_pairs (list): List of tuples [(protein, drug), ...]
        save_plot (bool): Whether to save the plot
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots"
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(exist_ok=True)
        
        all_data = []
        labels = []
        
        for protein_name, drug_name in protein_drug_pairs:
            # Path to the metadata CSV file
            metadata_file = metadata_dir / f"affinity_{protein_name}.csv"
            
            if not metadata_file.exists():
                print(f"Warning:  Skipping {protein_name} + {drug_name}: CSV file not found")
                continue
            
            # Read the CSV file
            df = pd.read_csv(metadata_file)
            
            # Filter data for the specific protein and drug
            filtered_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
            
            if filtered_df.empty:
                print(f"Warning:  Skipping {protein_name} + {drug_name}: No data found")
                continue
            
            # Check if affinity_probability_mean column exists
            if 'affinity_probability_mean' not in filtered_df.columns:
                print(f"Warning:  Skipping {protein_name} + {drug_name}: No affinity data")
                continue
            
            # Remove rows where mean probability is None or NaN
            filtered_df = filtered_df.dropna(subset=['affinity_probability_mean'])
            
            if filtered_df.empty:
                print(f"Warning:  Skipping {protein_name} + {drug_name}: No valid data")
                continue
            
            # Sort by sequence length (descending)
            filtered_df = filtered_df.sort_values('sequence_length', ascending=False)
            
            # Get truncations and probabilities
            truncations = filtered_df['truncation'].tolist()
            probabilities = filtered_df['affinity_probability_mean'].tolist()
            
            all_data.append(probabilities)
            labels.append(f"{protein_name}\n{drug_name}")
        
        if not all_data:
            print("Error: No valid data found for any protein-drug combination")
            return False
        
        # Create the heatmap
        plt.figure(figsize=(max(12, len(all_data[0]) * 0.8), max(6, len(all_data) * 1.5)))
        
        # Convert to numpy array
        heatmap_data = np.array(all_data)
        
        # Get truncation labels from the first dataset
        metadata_file = metadata_dir / f"affinity_{protein_drug_pairs[0][0]}.csv"
        df = pd.read_csv(metadata_file)
        filtered_df = df[(df['protein_name'] == protein_drug_pairs[0][0]) & 
                        (df['drug_name'] == protein_drug_pairs[0][1])]
        filtered_df = filtered_df.sort_values('sequence_length', ascending=False)
        truncations = filtered_df['truncation'].tolist()
        
        # Create the heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=truncations,
                   yticklabels=labels,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu',  # Blue=high probability, red=low probability
                   cbar_kws={'label': 'Mean Affinity Probability'},
                   linewidths=0.5,
                   linecolor='white')
        
        # Customize the plot
        plt.title('Binding Affinity Probability Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Protein Truncation', fontsize=12, fontweight='bold')
        plt.ylabel('Protein-Drug Combination', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            plot_filename = "multiple_proteins_affinity_heatmap.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved comparison heatmap to: {plot_path}")
        
        # Show the plot (only if explicitly requested)
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating comparison heatmap: {str(e)}")
        return False

def plot_column_differences(protein_name, selected_drugs=None, save_plot=True, show_plot=False):
    """
    Create a separate plot showing the differences between consecutive column averages.
    
    Args:
        protein_name (str): Name of the protein
        save_plot (bool): Whether to save the plot
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots" / protein_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not metadata_file.exists():
            print(f"Error: Metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Get available drugs
        available_drugs = df['drug_name'].unique()
        
        # Filter drugs if specified
        if selected_drugs:
            available_drugs = [drug for drug in available_drugs if drug in selected_drugs]
        
        # Filter data for drugs with valid affinity data
        all_data = []
        labels = []
        
        for drug_name in available_drugs:
            filtered_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
            
            if filtered_df.empty:
                print(f"Warning:  Skipping {drug_name}: No data found")
                continue
            
            # Check if affinity_probability_mean column exists
            if 'affinity_probability_mean' not in filtered_df.columns:
                print(f"Warning:  Skipping {drug_name}: No affinity data")
                continue
            
            # Remove rows where mean probability is None or NaN
            filtered_df = filtered_df.dropna(subset=['affinity_probability_mean'])
            
            if filtered_df.empty:
                print(f"Warning:  Skipping {drug_name}: No valid data")
                continue
            
            # Sort by sequence length (descending)
            filtered_df = filtered_df.sort_values('sequence_length', ascending=False)
            
            # Get truncations and probabilities
            truncations = filtered_df['truncation'].tolist()
            probabilities = filtered_df['affinity_probability_mean'].tolist()
            
            all_data.append(probabilities)
            labels.append(drug_name)
        
        if not all_data:
            print(f"Error: No valid data found for {protein_name}")
            return False
        
        # Convert to numpy array
        heatmap_data = np.array(all_data)
        
        # Get truncation labels from the first dataset
        first_drug_name = available_drugs[0]
        first_drug_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == first_drug_name)]
        first_drug_df = first_drug_df.sort_values('sequence_length', ascending=False)
        truncations = first_drug_df['truncation'].tolist()
        
        # ===== FIRST PLOT: CONSECUTIVE DIFFERENCES (PROBABILITY) =====
        
        # Create consecutive differences plot with individual drug lines
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        max_consecutive_differences = []
        
        # Get the drugs that actually have data
        valid_drugs = []
        for drug_name in available_drugs:
            filtered_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
            if not filtered_df.empty and 'affinity_probability_mean' in filtered_df.columns:
                filtered_df = filtered_df.dropna(subset=['affinity_probability_mean'])
                if not filtered_df.empty:
                    valid_drugs.append(drug_name)
        
        for idx, drug_name in enumerate(valid_drugs):
            # Get data for this specific drug
            drug_data = heatmap_data[idx]
            
            # Calculate consecutive differences for this drug
            consecutive_differences = []
            for i in range(1, len(drug_data)):
                diff = abs(drug_data[i] - drug_data[i-1])
                consecutive_differences.append(diff)
            
            max_consecutive_differences.append(max(consecutive_differences))
            
            # Plot line for this drug
            x_positions = range(len(consecutive_differences))
            plt.plot(x_positions, consecutive_differences, 'o-', linewidth=2, markersize=4, 
                    color=colors[idx % len(colors)], alpha=0.8, label=drug_name)
        
        # Create transition labels
        transition_labels = []
        for i in range(1, len(truncations)):
            transition_labels.append(f'{truncations[i-1]}→{truncations[i]}')
        
        # Customize the plot
        plt.title(f'{protein_name} - Consecutive Differences (Probability)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Transition Index', fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Difference', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(transition_labels) // 20)
        visible_positions = range(0, len(transition_labels), step)
        visible_labels = [transition_labels[i] for i in visible_positions]
        plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
        
        plt.tight_layout(pad=1.0)
        
        # Save consecutive differences plot
        if save_plot:
            plot_filename = f"{protein_name}_consecutive_differences_prob.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved consecutive differences (probability) plot to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # ===== SECOND PLOT: DELTA FROM ORIGINAL (PROBABILITY) =====
        
        # Create delta from original plot with individual drug lines
        plt.figure(figsize=(12, 8))
        
        max_differences = []
        
        for idx, drug_name in enumerate(valid_drugs):
            # Get data for this specific drug
            drug_data = heatmap_data[idx]
            
            # Find the original/full protein value (first column)
            original_value = drug_data[0]
            
            # Calculate absolute differences from original/full protein for this drug
            differences = []
            for i in range(len(drug_data)):
                diff = abs(drug_data[i] - original_value)  # Absolute difference
                differences.append(diff)
            
            max_differences.append(max(differences))
            
            # Plot line for this drug
            x_positions = range(len(differences))
            plt.plot(x_positions, differences, 'o-', linewidth=2, markersize=4, 
                    color=colors[idx % len(colors)], alpha=0.8, label=f'{drug_name} (orig: {original_value:.3f})')
        
        # Customize the plot
        plt.title(f'{protein_name} - Absolute Delta from Original Protein (Probability)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Truncation Position', fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Delta (Magnitude)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels - show truncation names
        step = max(1, len(truncations) // 15)  # Show ~15 labels
        visible_positions = range(0, len(truncations), step)
        visible_labels = [truncations[i] for i in visible_positions]
        plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
        
        # Adjust layout - make it more compact
        plt.tight_layout(pad=1.0)
        
        # Save the delta plot
        if save_plot:
            plot_filename = f"{protein_name}_delta_from_original_prob.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved delta (probability) plot to: {plot_path}")
        
        # Show the plot (only if explicitly requested)
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
        
        # ===== THIRD PLOT: CONSECUTIVE DIFFERENCES (MEAN VALUE) =====
        
        # Prepare mean value data
        all_value_data = []
        for drug_name in available_drugs:
            # Filter data for this specific drug from original dataframe
            drug_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
            
            # Check if ic50_pred_mean column exists
            if 'ic50_pred_mean' not in drug_df.columns:
                print(f"Warning:  Skipping {drug_name}: No IC50 value data")
                continue
            
            # Remove rows where mean value is None or NaN
            drug_df = drug_df.dropna(subset=['ic50_pred_mean'])
            
            if drug_df.empty:
                print(f"Warning:  Skipping {drug_name}: No valid value data")
                continue
            
            # Sort by sequence length (descending)
            drug_df = drug_df.sort_values('sequence_length', ascending=False)
            
            # Get truncations and values
            values = drug_df['ic50_pred_mean'].tolist()
            all_value_data.append(values)
        
        if not all_value_data:
            print("Error: No valid value data found for any drug")
            return False
        
        # Convert to numpy array (use all columns like probability data)
        value_heatmap_data = np.array(all_value_data)
        
        # ===== THIRD PLOT: CONSECUTIVE DIFFERENCES (MEAN VALUE) =====
        
        # Create consecutive differences plot for values with individual drug lines
        plt.figure(figsize=(12, 8))
        
        max_value_consecutive_differences = []
        
        for idx, drug_name in enumerate(valid_drugs):
            # Get data for this specific drug
            drug_value_data = value_heatmap_data[idx]
            
            # Calculate consecutive differences for this drug
            value_consecutive_differences = []
            for i in range(1, len(drug_value_data)):
                diff = abs(drug_value_data[i] - drug_value_data[i-1])
                value_consecutive_differences.append(diff)
            
            max_value_consecutive_differences.append(max(value_consecutive_differences))
            
            # Plot line for this drug
            x_positions = range(len(value_consecutive_differences))
            plt.plot(x_positions, value_consecutive_differences, 'o-', linewidth=2, markersize=4, 
                    color=colors[idx % len(colors)], alpha=0.8, label=drug_name)
        
        # Customize the plot
        plt.title(f'{protein_name} - Consecutive Differences (Mean Value)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Transition Index', fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Difference', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(transition_labels) // 20)
        visible_positions = range(0, len(transition_labels), step)
        visible_labels = [transition_labels[i] for i in visible_positions]
        plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
        
        plt.tight_layout(pad=1.0)
        
        # Save consecutive differences plot for values
        if save_plot:
            plot_filename = f"{protein_name}_consecutive_differences_value.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved consecutive differences (value) plot to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # ===== FOURTH PLOT: DELTA FROM ORIGINAL (MEAN VALUE) =====
        
        # Create delta from original plot for values with individual drug lines
        plt.figure(figsize=(12, 8))
        
        max_value_differences = []
        
        for idx, drug_name in enumerate(valid_drugs):
            # Get data for this specific drug
            drug_value_data = value_heatmap_data[idx]
            
            # Find the original/full protein value (first column)
            value_original_value = drug_value_data[0]
            
            # Calculate absolute differences from original/full protein for this drug
            value_differences = []
            for i in range(len(drug_value_data)):
                diff = abs(drug_value_data[i] - value_original_value)  # Absolute difference
                value_differences.append(diff)
            
            max_value_differences.append(max(value_differences))
            
            # Plot line for this drug
            x_positions = range(len(value_differences))
            plt.plot(x_positions, value_differences, 'o-', linewidth=2, markersize=4, 
                    color=colors[idx % len(colors)], alpha=0.8, label=f'{drug_name} (orig: {value_original_value:.3f})')
        
        # Customize the plot
        plt.title(f'{protein_name} - Absolute Delta from Original Protein (Mean Value)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Truncation Position', fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Delta (Magnitude)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels - show truncation names
        step = max(1, len(truncations) // 15)  # Show ~15 labels
        visible_positions = range(0, len(truncations), step)
        visible_labels = [truncations[i] for i in visible_positions]
        plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
        
        # Adjust layout - make it more compact
        plt.tight_layout(pad=1.0)
        
        # Save the delta plot for values
        if save_plot:
            plot_filename = f"{protein_name}_delta_from_original_value.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved delta (value) plot to: {plot_path}")
        
        # Show the plot (only if explicitly requested)
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
        
        # ===== FIFTH PLOT: AVERAGE CONSECUTIVE DIFFERENCES (ALL DRUGS COMBINED) =====
        
        # Calculate average consecutive differences across all drugs
        if len(valid_drugs) > 1:
            # Calculate average consecutive differences for probability
            avg_prob_consecutive_diffs = []
            for i in range(1, len(truncations)):
                drug_diffs = []
                for drug_name in valid_drugs:
                    drug_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
                    drug_df = drug_df.sort_values('sequence_length', ascending=False)
                    probabilities = drug_df['affinity_probability_mean'].tolist()
                    if len(probabilities) > i:
                        diff = abs(probabilities[i] - probabilities[i-1])
                        drug_diffs.append(diff)
                if drug_diffs:
                    avg_prob_consecutive_diffs.append(np.mean(drug_diffs))
            
            # Calculate average consecutive differences for mean values
            avg_value_consecutive_diffs = []
            for i in range(1, len(truncations)):
                drug_diffs = []
                for drug_name in valid_drugs:
                    drug_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
                    drug_df = drug_df.sort_values('sequence_length', ascending=False)
                    values = drug_df['ic50_pred_mean'].tolist()
                    if len(values) > i:
                        diff = abs(values[i] - values[i-1])
                        drug_diffs.append(diff)
                if drug_diffs:
                    avg_value_consecutive_diffs.append(np.mean(drug_diffs))
            
            # Plot 5: Average consecutive differences (probability) - all drugs combined
            plt.figure(figsize=(12, 8))
            x_positions = range(len(avg_prob_consecutive_diffs))
            plt.plot(x_positions, avg_prob_consecutive_diffs, 'o-', linewidth=3, markersize=6, 
                    color='purple', alpha=0.8, label='Average (All Drugs)')
            
            plt.title(f'{protein_name} - Average Consecutive Differences (Probability, All Drugs)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Transition Index', fontsize=12, fontweight='bold')
            plt.ylabel('Average Absolute Difference', fontsize=12, fontweight='bold')
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Set x-axis labels
            step = max(1, len(transition_labels) // 20)
            visible_positions = range(0, len(transition_labels), step)
            visible_labels = [transition_labels[i] for i in visible_positions]
            plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
            
            plt.tight_layout(pad=1.0)
            
            if save_plot:
                plot_filename = f"{protein_name}_average_consecutive_differences_prob.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved: Saved average consecutive differences (probability) plot to: {plot_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            # Plot 6: Average consecutive differences (mean value) - all drugs combined
            plt.figure(figsize=(12, 8))
            x_positions = range(len(avg_value_consecutive_diffs))
            plt.plot(x_positions, avg_value_consecutive_diffs, 'o-', linewidth=3, markersize=6, 
                    color='orange', alpha=0.8, label='Average (All Drugs)')
            
            plt.title(f'{protein_name} - Average Consecutive Differences (Mean Value, All Drugs)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Transition Index', fontsize=12, fontweight='bold')
            plt.ylabel('Average Absolute Difference', fontsize=12, fontweight='bold')
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Set x-axis labels
            plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
            
            plt.tight_layout(pad=1.0)
            
            if save_plot:
                plot_filename = f"{protein_name}_average_consecutive_differences_value.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved: Saved average consecutive differences (mean value) plot to: {plot_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            # ===== SEVENTH PLOT: AVERAGE DELTA FROM ORIGINAL (ALL DRUGS COMBINED) =====
            
            # Calculate average delta from original across all drugs
            avg_prob_deltas = []
            avg_value_deltas = []
            
            for i in range(len(truncations)):
                drug_prob_deltas = []
                drug_value_deltas = []
                
                for drug_name in valid_drugs:
                    drug_df = df[(df['protein_name'] == protein_name) & (df['drug_name'] == drug_name)]
                    drug_df = drug_df.sort_values('sequence_length', ascending=False)
                    
                    # Get original values (usually the first or last entry)
                    original_prob = drug_df['affinity_probability_mean'].iloc[0]  # First entry is usually original
                    original_value = drug_df['ic50_pred_mean'].iloc[0]
                    
                    # Get current truncation values
                    if i < len(drug_df):
                        current_prob = drug_df['affinity_probability_mean'].iloc[i]
                        current_value = drug_df['ic50_pred_mean'].iloc[i]
                        
                        prob_delta = abs(current_prob - original_prob)
                        value_delta = abs(current_value - original_value)
                        
                        drug_prob_deltas.append(prob_delta)
                        drug_value_deltas.append(value_delta)
                
                if drug_prob_deltas:
                    avg_prob_deltas.append(np.mean(drug_prob_deltas))
                if drug_value_deltas:
                    avg_value_deltas.append(np.mean(drug_value_deltas))
            
            # Plot 7: Average delta from original (probability) - all drugs combined
            plt.figure(figsize=(12, 8))
            x_positions = range(len(avg_prob_deltas))
            plt.plot(x_positions, avg_prob_deltas, 'o-', linewidth=3, markersize=6, 
                    color='darkred', alpha=0.8, label='Average (All Drugs)')
            
            plt.title(f'{protein_name} - Average Delta from Original (Probability, All Drugs)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Truncation Position', fontsize=12, fontweight='bold')
            plt.ylabel('Average |Probability - Original|', fontsize=12, fontweight='bold')
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Set x-axis labels
            step = max(1, len(truncations) // 15)
            visible_positions = range(0, len(truncations), step)
            visible_labels = [truncations[i] for i in visible_positions]
            plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
            
            plt.tight_layout(pad=1.0)
            
            if save_plot:
                plot_filename = f"{protein_name}_average_delta_from_original_prob.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved: Saved average delta from original (probability) plot to: {plot_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            # Plot 8: Average delta from original (mean value) - all drugs combined
            plt.figure(figsize=(12, 8))
            x_positions = range(len(avg_value_deltas))
            plt.plot(x_positions, avg_value_deltas, 'o-', linewidth=3, markersize=6, 
                    color='darkblue', alpha=0.8, label='Average (All Drugs)')
            
            plt.title(f'{protein_name} - Average Delta from Original (Mean Value, All Drugs)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Truncation Position', fontsize=12, fontweight='bold')
            plt.ylabel('Average |Mean Value - Original|', fontsize=12, fontweight='bold')
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Set x-axis labels
            plt.xticks(visible_positions, visible_labels, fontsize=8, rotation=45, ha='right')
            
            plt.tight_layout(pad=1.0)
            
            if save_plot:
                plot_filename = f"{protein_name}_average_delta_from_original_value.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved: Saved average delta from original (mean value) plot to: {plot_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating differences plot: {str(e)}")
        return False

def plot_mutation_affinity_heatmap(protein_name, drug_name, mutation_range, save_plot=True, show_plot=False):
    """
    Create a heatmap visualization of mutation affinity probability values.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_range (str): Mutation range (e.g., '790-810')
        save_plot (bool): Whether to save the plot as PNG file
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots"
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(exist_ok=True)
        
        # Path to the mutation metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not metadata_file.exists():
            print(f"Error: Mutation metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Filter data for the specific protein, drug, and mutation range
        # For mutations, protein_name can be either the base name (EGFR) or mutated name (EGFR-T790A)
        filtered_df = df[((df['protein_name'] == protein_name) | (df['protein_name'].str.startswith(protein_name + '-'))) & 
                        (df['drug_name'] == drug_name) & 
                        (df['mutation_range'] == mutation_range)]
        
        if filtered_df.empty:
            print(f"Error: No data found for {protein_name}, {drug_name}, and {mutation_range}")
            return False
        
        # Check if affinity_probability_mean column exists
        if 'affinity_probability_mean' not in filtered_df.columns:
            print("Error: 'affinity_probability_mean' column not found. Run add_mutation_affinity_values.py first.")
            return False
        
        # Remove rows where mean probability is None or NaN
        filtered_df = filtered_df.dropna(subset=['affinity_probability_mean'])
        
        if filtered_df.empty:
            print("Error: No valid probability data found")
            return False
        
        # Parse mutation range to get start and end positions
        start_pos, end_pos = map(int, mutation_range.split('-'))
        positions = list(range(start_pos, end_pos + 1))
        
        # Define the 20 amino acids
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Create a 2D matrix for the heatmap
        # Rows: amino acids (20), Columns: positions
        heatmap_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        
        # Get original amino acid for each position
        original_aa_dict = {}
        for _, row in filtered_df.iterrows():
            if row['is_mutation']:
                pos = row['mutation_position']
                if pos in positions:
                    original_aa = row['original_amino_acid']
                    original_aa_dict[pos] = original_aa
            elif row['mutation_id'] == 'ORIGINAL':
                # For ORIGINAL entries, we need to get the original amino acid from the sequence
                # We'll use the first mutation row for each position to get the original AA
                pass
        
        # Fill the heatmap matrix
        for pos_idx, position in enumerate(positions):
            # Get mutations for this position
            pos_mutations = filtered_df[filtered_df['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                # Find the mutation for this amino acid at this position
                mutation_row = pos_mutations[pos_mutations['new_amino_acid'] == amino_acid]
                
                if not mutation_row.empty:
                    # This is a mutation that exists
                    prob_value = mutation_row.iloc[0]['affinity_probability_mean']
                    heatmap_matrix[aa_idx, pos_idx] = prob_value
                else:
                    # Check if this is the original amino acid
                    original_aa = original_aa_dict.get(position)
                    if amino_acid == original_aa:
                        # This is the original amino acid (no mutation)
                        # Find the ORIGINAL row for this position
                        original_row = filtered_df[(filtered_df['mutation_position'] == position) & 
                                                 (filtered_df['mutation_id'] == 'ORIGINAL')]
                        if not original_row.empty:
                            prob_value = original_row.iloc[0]['affinity_probability_mean']
                            heatmap_matrix[aa_idx, pos_idx] = prob_value
                        else:
                            # If no ORIGINAL row found, try to get from any mutation row for this position
                            any_mutation_row = pos_mutations.iloc[0] if not pos_mutations.empty else None
                            if any_mutation_row is not None:
                                prob_value = any_mutation_row['affinity_probability_mean']
                                heatmap_matrix[aa_idx, pos_idx] = prob_value
        
        # Create the plot with smaller dimensions
        plt.figure(figsize=(max(12, len(positions) * 0.8), 6))
        
        # Create a custom colormap with green for original values
        # We'll use a mask to show original values in green
        original_mask = np.zeros_like(heatmap_matrix, dtype=bool)
        
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(heatmap_matrix[aa_idx, pos_idx]):
                    original_mask[aa_idx, pos_idx] = True
        
        # Create subplots: colorbar at top, heatmap below with more space
        gs = plt.GridSpec(2, 1, height_ratios=[0.1, 0.9], hspace=0.5)
        
        # Create colorbar in top subplot with red=small values, blue=large values, flipped stick
        cbar_ax = plt.subplot(gs[0])
        min_val = np.nanmin(heatmap_matrix)
        max_val = np.nanmax(heatmap_matrix)
        norm = plt.Normalize(max_val, min_val)  # Reversed normalization to flip the stick
        sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)  # Normal colormap
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Mean Affinity Probability', fontsize=10, fontweight='bold')
        cbar_ax.set_position([0.1, 0.95, 0.8, 0.03])
        
        # Create heatmap in bottom subplot
        ax = plt.subplot(gs[1])
        sns.heatmap(heatmap_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,  # Remove numerical annotations
                   cmap='RdYlBu',  # Red-Yellow-Blue (dark blue=high probability, red=low probability)
                   cbar=False,  # Don't show colorbar yet
                   linewidths=0.1,  # Very thin lines for bigger boxes
                   linecolor='white',
                   square=False,  # Don't force square to allow bigger boxes
                   ax=ax)
        
        # Add green circles to annotate original amino acid boxes
        if np.any(original_mask):
            for pos_idx, position in enumerate(positions):
                original_aa = original_aa_dict.get(position)
                if original_aa:
                    aa_idx = amino_acids.index(original_aa)
                    if not np.isnan(heatmap_matrix[aa_idx, pos_idx]):
                        # Add a green circle to mark original values
                        plt.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                                ha='center', va='center', fontsize=10, 
                                color='green', weight='bold')
        
        # Customize the plot with title positioned lower
        plt.title(f'Mutation Affinity for {protein_name}', 
                 fontsize=16, fontweight='bold', pad=40, y=0.85)
        plt.xlabel('Residue Position', fontsize=12, fontweight='bold')
        plt.ylabel('Amino Acid', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability and spacing
        plt.xticks(rotation=0, fontsize=8, ha='center')
        plt.yticks(fontsize=8)
        
        # Adjust layout to push matrix down
        plt.subplots_adjust(bottom=0.12, top=0.85, left=0.08, right=0.92)
        
        # Save the plot
        if save_plot:
            plot_filename = f"mutation_heatmap_{protein_name}_{drug_name}_{mutation_range}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved mutation heatmap to {plot_path}")
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating mutation heatmap: {str(e)}")
        return False

def plot_mutation_affinity_heatmap_variants(protein_name, drug_name, mutation_range, save_plot=True, show_plot=False):
    """
    Create 4 different heatmap visualizations of mutation affinity values:
    1. Probability values (current)
    2. Mean values (affinity_pred_mean)
    3. Absolute difference from original using probability values
    4. Absolute difference from original using mean values
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_range (str): Mutation range (e.g., '790-810')
        save_plot (bool): Whether to save the plot as PNG file
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots"
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(exist_ok=True)
        
        # Path to the mutation metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not metadata_file.exists():
            print(f"Error: Mutation metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Filter data for the specific protein, drug, and mutation range
        filtered_df = df[((df['protein_name'] == protein_name) | (df['protein_name'].str.startswith(protein_name + '-'))) & 
                        (df['drug_name'] == drug_name) & 
                        (df['mutation_range'] == mutation_range)]
        
        if filtered_df.empty:
            print(f"Error: No data found for {protein_name}, {drug_name}, and {mutation_range}")
            return False
        
        
        # Check if required columns exist
        required_columns = ['affinity_probability_mean', 'ic50_pred_mean']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}. Run add_mutation_affinity_values.py first.")
            return False
        
        # Remove rows where mean values are None or NaN
        filtered_df = filtered_df.dropna(subset=['affinity_probability_mean', 'ic50_pred_mean'])
        
        if filtered_df.empty:
            print("Error: No valid data found")
            return False
        
        # Parse mutation range to get start and end positions
        start_pos, end_pos = map(int, mutation_range.split('-'))
        positions = list(range(start_pos, end_pos + 1))
        
        # Define the 20 amino acids
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Create 4 different matrices for the heatmaps
        prob_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        mean_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        prob_diff_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        mean_diff_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        
        # Get original amino acid for each position
        original_aa_dict = {}
        original_prob_dict = {}
        original_mean_dict = {}
        
        # Get original values from the ORIGINAL entry (which has mutation_position=0)
        original_entry = filtered_df[filtered_df['mutation_id'] == 'ORIGINAL'].iloc[0]
        original_prob_value = original_entry['affinity_probability_mean']
        original_mean_value = original_entry['ic50_pred_mean']
        
        # For each position, get the original amino acid from mutation entries
        for _, row in filtered_df.iterrows():
            if row['is_mutation']:
                pos = row['mutation_position']
                if pos in positions:
                    original_aa = row['original_amino_acid']
                    original_aa_dict[pos] = original_aa
                    # Use the original values for all positions
                    original_prob_dict[pos] = original_prob_value
                    original_mean_dict[pos] = original_mean_value
        
        # Fill the matrices
        for pos_idx, position in enumerate(positions):
            # Get mutations for this position
            pos_mutations = filtered_df[filtered_df['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                # Find the mutation for this amino acid at this position
                mutation_row = pos_mutations[pos_mutations['new_amino_acid'] == amino_acid]
                
                if not mutation_row.empty:
                    # This is a mutation that exists
                    prob_value = mutation_row.iloc[0]['affinity_probability_mean']
                    mean_value = mutation_row.iloc[0]['ic50_pred_mean']
                    
                    prob_matrix[aa_idx, pos_idx] = prob_value
                    mean_matrix[aa_idx, pos_idx] = mean_value
                    
                    # Calculate differences from original
                    original_prob = original_prob_dict.get(position)
                    original_mean = original_mean_dict.get(position)
                    
                    if original_prob is not None:
                        prob_diff_matrix[aa_idx, pos_idx] = abs(prob_value - original_prob)
                    
                    if original_mean is not None:
                        mean_diff_matrix[aa_idx, pos_idx] = abs(mean_value - original_mean)
                    

                else:
                    # Check if this is the original amino acid
                    original_aa = original_aa_dict.get(position)
                    if amino_acid == original_aa:
                        # This is the original amino acid (no mutation)
                        original_row = filtered_df[(filtered_df['mutation_position'] == position) & 
                                                 (filtered_df['mutation_id'] == 'ORIGINAL')]
                        if not original_row.empty:
                            prob_value = original_row.iloc[0]['affinity_probability_mean']
                            mean_value = original_row.iloc[0]['ic50_pred_mean']
                            
                            prob_matrix[aa_idx, pos_idx] = prob_value
                            mean_matrix[aa_idx, pos_idx] = mean_value
                            
                            # Original values have zero difference
                            prob_diff_matrix[aa_idx, pos_idx] = 0.0
                            mean_diff_matrix[aa_idx, pos_idx] = 0.0
        
        # Create the plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Mutation Affinity Analysis for {protein_name} + {drug_name} ({mutation_range})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Probability values (current)
        ax1 = axes[0, 0]
        sns.heatmap(prob_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',
                   cbar_kws={'label': 'Mean Affinity Probability'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax1)
        ax1.set_title('Probability Values', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Residue Position')
        ax1.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(prob_matrix[aa_idx, pos_idx]):
                    ax1.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                            ha='center', va='center', fontsize=8, 
                            color='green', weight='bold')
        
        # Plot 2: Mean values
        ax2 = axes[0, 1]
        sns.heatmap(mean_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',
                   cbar_kws={'label': 'Mean Affinity Value'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax2)
        ax2.set_title('Mean Values', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Residue Position')
        ax2.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(mean_matrix[aa_idx, pos_idx]):
                    ax2.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                            ha='center', va='center', fontsize=8, 
                            color='green', weight='bold')
        
        # Plot 3: Absolute difference from original (probability)
        ax3 = axes[1, 0]
        
        # Replace NaN with 0 for better visualization
        prob_diff_matrix_vis = prob_diff_matrix.copy()
        prob_diff_matrix_vis = np.nan_to_num(prob_diff_matrix_vis, nan=0.0)
        
        sns.heatmap(prob_diff_matrix_vis, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='Reds',  # Red for differences
                   cbar_kws={'label': '|Probability - Original|'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax3)
        ax3.set_title('Absolute Difference (Probability)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Residue Position')
        ax3.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids (zero difference)
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                ax3.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                        ha='center', va='center', fontsize=8, 
                        color='green', weight='bold')
        
        # Plot 4: Absolute difference from original (mean)
        ax4 = axes[1, 1]
        
        # Replace NaN with 0 for better visualization
        mean_diff_matrix_vis = mean_diff_matrix.copy()
        mean_diff_matrix_vis = np.nan_to_num(mean_diff_matrix_vis, nan=0.0)
        
        sns.heatmap(mean_diff_matrix_vis, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='Reds',  # Red for differences
                   cbar_kws={'label': '|Mean - Original|'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax4)
        ax4.set_title('Absolute Difference (Mean)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Residue Position')
        ax4.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids (zero difference)
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                ax4.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                        ha='center', va='center', fontsize=8, 
                        color='green', weight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            plot_filename = f"mutation_heatmap_variants_{protein_name}_{drug_name}_{mutation_range}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved mutation heatmap variants to {plot_path}")
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return True
        
    except Exception as e:
        print(f"Error: Error creating mutation heatmap variants: {str(e)}")
        return False

def plot_combined_heatmap(protein_name, drug_name, mutation_range, selected_drugs=None, save_plot=True, show_plot=False):
    """
    Create a combined heatmap with comparison heatmap above and mutation heatmap below.
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug for mutation heatmap
        mutation_range (str): Mutation range (e.g., '790-810')
        selected_drugs (list): List of drugs for comparison heatmap
        save_plot (bool): Whether to save the plot
        show_plot (bool): Whether to show the plot
    """
    try:
        
        # Create the combined plot
        plt.figure(figsize=(max(16, 21 * 1.2), 12))  # Wider and taller for both plots
        
        # Create subplots: comparison at top, mutation at bottom
        gs = plt.GridSpec(2, 1, height_ratios=[0.5, 0.5], hspace=0.4)
        
        # ===== TOP PLOT: COMPARISON HEATMAP =====
        ax1 = plt.subplot(gs[0])
        
        # Load comparison data
        metadata_dir = Path("affinity_metadata")
        csv_file = metadata_dir / f"affinity_{protein_name}.csv"
        
        if not csv_file.exists():
            print(f"Error: No metadata file found for {protein_name}")
            return False
        
        df = pd.read_csv(csv_file)
        
        # Filter for selected drugs or use all available
        if selected_drugs:
            filtered_df = df[df['drug_name'].isin(selected_drugs)]
            available_drugs = selected_drugs
        else:
            available_drugs = df['drug_name'].unique()
            filtered_df = df
        
        if len(available_drugs) == 0:
            print(f"Error: No drugs found for {protein_name}")
            return False
        
        # Prepare comparison data
        all_data = []
        for drug in available_drugs:
            drug_df = filtered_df[filtered_df['drug_name'] == drug]
            drug_df = drug_df.sort_values('sequence_length', ascending=False)  # Descending order
            # Limit to columns 10-30
            probabilities = drug_df['affinity_probability_mean'].tolist()[9:30]
            all_data.append(probabilities)
        
        # Convert to numpy array
        heatmap_data = np.array(all_data)[:, 9:30]  # Columns 10-30 (0-indexed: 9-29)
        
        # Get truncation labels
        first_drug_df = filtered_df[filtered_df['drug_name'] == available_drugs[0]]
        first_drug_df = first_drug_df.sort_values('sequence_length', ascending=False)  # Descending order
        truncations_raw = first_drug_df['truncation'].tolist()[9:30]  # Columns 10-30
        
        # Reverse the order so 693 -> 700 -> FULL
        heatmap_data = heatmap_data[:, ::-1]  # Reverse the data columns
        truncations_raw = truncations_raw[::-1]  # Reverse the labels
        
        # Simplify truncation labels
        truncations = []
        for trunc in truncations_raw:
            if trunc.startswith('TRUNC'):
                truncations.append(trunc[5:])  # Remove 'TRUNC' prefix
            else:
                truncations.append(trunc)  # Keep as is (e.g., 'FULL')
        
        # Create comparison heatmap
        sns.heatmap(heatmap_data,
                   xticklabels=truncations,
                   yticklabels=available_drugs,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu',
                   cbar=False,  # Remove colorbar
                   linewidths=0.2,
                   linecolor='white',
                   square=True,
                   ax=ax1)
        
        # Customize comparison plot
        ax1.set_title(f'Protein Comparison Heatmap - {protein_name}', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Truncation Position', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Drug', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=0, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        
        # ===== BOTTOM PLOT: MUTATION HEATMAP =====
        ax2 = plt.subplot(gs[1])
        
        # Load mutation data
        mutation_csv_file = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not mutation_csv_file.exists():
            print(f"Error: No mutation metadata file found for {protein_name}")
            return False
        
        mutation_df = pd.read_csv(mutation_csv_file)
        
        # Parse mutation range
        start_pos, end_pos = map(int, mutation_range.split('-'))
        positions = list(range(start_pos, end_pos + 1))
        
        # Filter for specific drug and mutation range
        filtered_df = mutation_df[
            (mutation_df['protein_name'] == protein_name) &
            (mutation_df['drug_name'] == drug_name) &
            (mutation_df['mutation_range'] == mutation_range)
        ]
        
        if filtered_df.empty:
            print(f"Error: No data found for {protein_name} + {drug_name} + {mutation_range}")
            return False
        
        # Prepare mutation data
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        heatmap_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        original_aa_dict = {}
        
        # Get original amino acid for each position
        for _, row in filtered_df.iterrows():
            if row['is_mutation']:
                pos = row['mutation_position']
                if pos in positions:
                    original_aa = row['original_amino_acid']
                    original_aa_dict[pos] = original_aa
            elif row['mutation_id'] == 'ORIGINAL':
                # For ORIGINAL entries, we need to get the original amino acid from the sequence
                # We'll use the first mutation row for each position to get the original AA
                pass
        
        # Fill the heatmap matrix
        for pos_idx, position in enumerate(positions):
            # Get mutations for this position
            pos_mutations = filtered_df[filtered_df['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                # Find the mutation for this amino acid at this position
                mutation_row = pos_mutations[pos_mutations['new_amino_acid'] == amino_acid]
                
                if not mutation_row.empty:
                    # This is a mutation that exists
                    prob_value = mutation_row.iloc[0]['affinity_probability_mean']
                    heatmap_matrix[aa_idx, pos_idx] = prob_value
                else:
                    # Check if this is the original amino acid
                    original_aa = original_aa_dict.get(position)
                    if amino_acid == original_aa:
                        # This is the original amino acid (no mutation)
                        # Find the ORIGINAL row for this position
                        original_row = filtered_df[(filtered_df['mutation_position'] == position) & 
                                                 (filtered_df['mutation_id'] == 'ORIGINAL')]
                        if not original_row.empty:
                            prob_value = original_row.iloc[0]['affinity_probability_mean']
                            heatmap_matrix[aa_idx, pos_idx] = prob_value
                        else:
                            # If no ORIGINAL row found, try to get from any mutation row for this position
                            any_mutation_row = pos_mutations.iloc[0] if not pos_mutations.empty else None
                            if any_mutation_row is not None:
                                prob_value = any_mutation_row['affinity_probability_mean']
                                heatmap_matrix[aa_idx, pos_idx] = prob_value
        
        # Create mutation heatmap with red=small values, blue=large values
        # Create a modified matrix where original amino acids have a special value
        modified_matrix = heatmap_matrix.copy()
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(heatmap_matrix[aa_idx, pos_idx]):
                    # Set original amino acids to a very low value to make them appear red
                    modified_matrix[aa_idx, pos_idx] = np.nanmin(heatmap_matrix) - 0.2
        
        # Create the heatmap
        sns.heatmap(modified_matrix,
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',  # Normal colormap
                   cbar=False,
                   linewidths=0.1,
                   linecolor='white',
                   square=False,
                   ax=ax2)
        
        # Add green circles for original amino acids
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(heatmap_matrix[aa_idx, pos_idx]):
                    ax2.text(pos_idx + 0.5, aa_idx + 0.5, '●',
                            ha='center', va='center', fontsize=10,
                            color='green', weight='bold')
        
        # Customize mutation plot
        ax2.set_title(f'Mutation Affinity for {protein_name} + {drug_name} (● = Original)', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Residue Position', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Amino Acid', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=0, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        
        # Add overall title
        plt.suptitle(f'Combined Analysis: {protein_name}', fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.08, top=0.92, left=0.08, right=0.92)
        
        # Save the plot
        if save_plot:
            plots_dir = Path("affinity_plots")
            plots_dir.mkdir(exist_ok=True)
            plot_filename = f"combined_heatmap_{protein_name}_{drug_name}_{mutation_range}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved combined heatmap to {plot_path}")
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating combined heatmap: {str(e)}")
        return False

def plot_mutation_probability_heatmaps(protein_name, drug_name, mutation_range, save_plot=True, show_plot=False):
    """
    Create 2 probability-based heatmap visualizations of mutation affinity values:
    1. Probability values (current)
    2. Absolute difference from original using probability values
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_range (str): Mutation range (e.g., '790-810')
        save_plot (bool): Whether to save the plot as PNG file
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots"
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(exist_ok=True)
        
        # Path to the mutation metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not metadata_file.exists():
            print(f"Error: Mutation metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Filter data for the specific protein, drug, and mutation range
        filtered_df = df[((df['protein_name'] == protein_name) | (df['protein_name'].str.startswith(protein_name + '-'))) & 
                        (df['drug_name'] == drug_name) & 
                        (df['mutation_range'] == mutation_range)]
        
        if filtered_df.empty:
            print(f"Error: No data found for {protein_name}, {drug_name}, and {mutation_range}")
            return False
        
        
        # Check if required columns exist
        required_columns = ['affinity_probability_mean', 'ic50_pred_mean']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}. Run add_mutation_affinity_values.py first.")
            return False
        
        # Remove rows where mean values are None or NaN
        filtered_df = filtered_df.dropna(subset=['affinity_probability_mean', 'ic50_pred_mean'])
        
        if filtered_df.empty:
            print("Error: No valid data found")
            return False
        
        # Parse mutation range to get start and end positions
        start_pos, end_pos = map(int, mutation_range.split('-'))
        positions = list(range(start_pos, end_pos + 1))
        
        # Define the 20 amino acids
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Create 2 different matrices for the heatmaps
        prob_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        prob_diff_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        
        # Get original values from the ORIGINAL entry (which has mutation_position=0)
        original_entry = filtered_df[filtered_df['mutation_id'] == 'ORIGINAL'].iloc[0]
        original_prob_value = original_entry['affinity_probability_mean']
        
        # Get original amino acid for each position
        original_aa_dict = {}
        original_prob_dict = {}
        
        # For each position, get the original amino acid from mutation entries
        for _, row in filtered_df.iterrows():
            if row['is_mutation']:
                pos = row['mutation_position']
                if pos in positions:
                    original_aa = row['original_amino_acid']
                    original_aa_dict[pos] = original_aa
                    # Use the original values for all positions
                    original_prob_dict[pos] = original_prob_value
        
        # Fill the matrices
        for pos_idx, position in enumerate(positions):
            # Get mutations for this position
            pos_mutations = filtered_df[filtered_df['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                # Find the mutation for this amino acid at this position
                mutation_row = pos_mutations[pos_mutations['new_amino_acid'] == amino_acid]
                
                if not mutation_row.empty:
                    # This is a mutation that exists
                    prob_value = mutation_row.iloc[0]['affinity_probability_mean']
                    
                    prob_matrix[aa_idx, pos_idx] = prob_value
                    
                    # Calculate differences from original
                    original_prob = original_prob_dict.get(position)
                    
                    if original_prob is not None:
                        prob_diff_matrix[aa_idx, pos_idx] = abs(prob_value - original_prob)
                else:
                    # Check if this is the original amino acid
                    original_aa = original_aa_dict.get(position)
                    if amino_acid == original_aa:
                        # This is the original amino acid (no mutation)
                        original_row = filtered_df[(filtered_df['mutation_position'] == position) & 
                                                 (filtered_df['mutation_id'] == 'ORIGINAL')]
                        if not original_row.empty:
                            prob_value = original_row.iloc[0]['affinity_probability_mean']
                            
                            prob_matrix[aa_idx, pos_idx] = prob_value
                            
                            # Original values have zero difference
                            prob_diff_matrix[aa_idx, pos_idx] = 0.0
        
        # Create the plots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Mutation Probability Analysis for {protein_name} + {drug_name} ({mutation_range})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Probability values
        ax1 = axes[0]
        sns.heatmap(prob_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',
                   cbar_kws={'label': 'Mean Affinity Probability'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax1)
        ax1.set_title('Probability Values', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Residue Position')
        ax1.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(prob_matrix[aa_idx, pos_idx]):
                    ax1.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                            ha='center', va='center', fontsize=8, 
                            color='green', weight='bold')
        
        # Plot 2: Absolute difference from original (probability)
        ax2 = axes[1]
        
        # Replace NaN with 0 for better visualization
        prob_diff_matrix_vis = prob_diff_matrix.copy()
        prob_diff_matrix_vis = np.nan_to_num(prob_diff_matrix_vis, nan=0.0)
        
        sns.heatmap(prob_diff_matrix_vis, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='Reds',  # Red for differences
                   cbar_kws={'label': '|Probability - Original|'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax2)
        ax2.set_title('Absolute Difference (Probability)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Residue Position')
        ax2.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids (zero difference)
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                ax2.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                        ha='center', va='center', fontsize=8, 
                        color='green', weight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            plot_filename = f"mutation_probability_heatmaps_{protein_name}_{drug_name}_{mutation_range}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved mutation probability heatmaps to {plot_path}")
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating mutation probability heatmaps: {str(e)}")
        return False

def plot_mutation_affinity_score_heatmaps(protein_name, drug_name, mutation_range, save_plot=True, show_plot=False):
    """
    Create 4 affinity score-based heatmap visualizations of mutation affinity values:
    1. Mean IC50 values (ic50_pred_mean) with reversed colors
    2. Delta mean IC50 with wild type (absolute difference from original)
    3. Mean affinity values (affinity_pred_mean) with reversed colors  
    4. Delta mean affinity score values with wild type (absolute difference from original)
    
    Args:
        protein_name (str): Name of the protein (e.g., 'EGFR')
        drug_name (str): Name of the drug (e.g., 'osimertinib')
        mutation_range (str): Mutation range (e.g., '790-810')
        save_plot (bool): Whether to save the plot as PNG file
        show_plot (bool): Whether to display the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define paths
        base_dir = Path(".")
        metadata_dir = base_dir / "affinity_metadata"
        plots_dir = base_dir / "affinity_plots"
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(exist_ok=True)
        
        # Path to the mutation metadata CSV file
        metadata_file = metadata_dir / f"affinity_{protein_name}_mutation.csv"
        
        if not metadata_file.exists():
            print(f"Error: Mutation metadata file not found: {metadata_file}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Filter data for the specific protein, drug, and mutation range
        filtered_df = df[((df['protein_name'] == protein_name) | (df['protein_name'].str.startswith(protein_name + '-'))) & 
                        (df['drug_name'] == drug_name) & 
                        (df['mutation_range'] == mutation_range)]
        
        if filtered_df.empty:
            print(f"Error: No data found for {protein_name}, {drug_name}, and {mutation_range}")
            return False
        
        
        # Check if required columns exist
        required_columns = ['affinity_probability_mean', 'ic50_pred_mean', 'affinity_pred_mean']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}. Run add_mutation_affinity_values.py first.")
            return False
        
        # Remove rows where mean values are None or NaN
        filtered_df = filtered_df.dropna(subset=['affinity_probability_mean', 'ic50_pred_mean', 'affinity_pred_mean'])
        
        if filtered_df.empty:
            print("Error: No valid data found")
            return False
        
        # Parse mutation range to get start and end positions
        start_pos, end_pos = map(int, mutation_range.split('-'))
        positions = list(range(start_pos, end_pos + 1))
        
        # Define the 20 amino acids
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Create 4 different matrices for the heatmaps
        ic50_mean_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        ic50_diff_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        affinity_mean_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        affinity_diff_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        
        # Get original values from the ORIGINAL entry (which has mutation_position=0)
        original_entry = filtered_df[filtered_df['mutation_id'] == 'ORIGINAL'].iloc[0]
        original_ic50_value = original_entry['ic50_pred_mean']
        original_affinity_value = original_entry['affinity_pred_mean']
        
        # Get original amino acid for each position
        original_aa_dict = {}
        original_ic50_dict = {}
        original_affinity_dict = {}
        
        # For each position, get the original amino acid from mutation entries
        for _, row in filtered_df.iterrows():
            if row['is_mutation']:
                pos = row['mutation_position']
                if pos in positions:
                    original_aa = row['original_amino_acid']
                    original_aa_dict[pos] = original_aa
                    # Use the original values for all positions
                    original_ic50_dict[pos] = original_ic50_value
                    original_affinity_dict[pos] = original_affinity_value
        
        # Fill the matrices
        for pos_idx, position in enumerate(positions):
            # Get mutations for this position
            pos_mutations = filtered_df[filtered_df['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                # Find the mutation for this amino acid at this position
                mutation_row = pos_mutations[pos_mutations['new_amino_acid'] == amino_acid]
                
                if not mutation_row.empty:
                    # This is a mutation that exists
                    ic50_value = mutation_row.iloc[0]['ic50_pred_mean']
                    affinity_value = mutation_row.iloc[0]['affinity_pred_mean']
                    
                    ic50_mean_matrix[aa_idx, pos_idx] = ic50_value
                    affinity_mean_matrix[aa_idx, pos_idx] = affinity_value
                    
                    # Calculate differences from original
                    original_ic50 = original_ic50_dict.get(position)
                    original_affinity = original_affinity_dict.get(position)
                    
                    if original_ic50 is not None:
                        ic50_diff_matrix[aa_idx, pos_idx] = abs(ic50_value - original_ic50)
                    if original_affinity is not None:
                        affinity_diff_matrix[aa_idx, pos_idx] = abs(affinity_value - original_affinity)
                else:
                    # Check if this is the original amino acid
                    original_aa = original_aa_dict.get(position)
                    if amino_acid == original_aa:
                        # This is the original amino acid (no mutation)
                        original_row = filtered_df[(filtered_df['mutation_position'] == position) & 
                                                 (filtered_df['mutation_id'] == 'ORIGINAL')]
                        if not original_row.empty:
                            ic50_value = original_row.iloc[0]['ic50_pred_mean']
                            affinity_value = original_row.iloc[0]['affinity_pred_mean']
                            
                            ic50_mean_matrix[aa_idx, pos_idx] = ic50_value
                            affinity_mean_matrix[aa_idx, pos_idx] = affinity_value
                            
                            # Original values have zero difference
                            ic50_diff_matrix[aa_idx, pos_idx] = 0.0
                            affinity_diff_matrix[aa_idx, pos_idx] = 0.0
        
        # Create the plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Mean IC50 values with reversed colors
        ax1 = axes[0, 0]
        sns.heatmap(ic50_mean_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',  # Match truncation colormap
                   cbar_kws={'label': 'Mean pIC50 Score'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax1)
        ax1.set_title(f'{drug_name} - Mean pIC50 Values ({mutation_range})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Residue Position')
        ax1.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(ic50_mean_matrix[aa_idx, pos_idx]):
                    ax1.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                            ha='center', va='center', fontsize=8, 
                            color='green', weight='bold')
        
        # Plot 2: Delta mean pIC50 with wild type
        ax2 = axes[0, 1]
        
        # Replace NaN with 0 for better visualization
        ic50_diff_matrix_vis = ic50_diff_matrix.copy()
        ic50_diff_matrix_vis = np.nan_to_num(ic50_diff_matrix_vis, nan=0.0)
        
        sns.heatmap(ic50_diff_matrix_vis, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu_r',  # Blue low -> Red high
                   vmin=0,
                   cbar_kws={'label': '|IC50 Score - Wild Type|'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax2)
        ax2.set_title(f'{drug_name} - Δ pIC50 from Wild Type ({mutation_range})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Residue Position')
        ax2.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids (zero difference)
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                ax2.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                        ha='center', va='center', fontsize=8, 
                        color='green', weight='bold')
        
        # Plot 3: Mean affinity values with reversed colors
        ax3 = axes[1, 0]
        sns.heatmap(affinity_mean_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',  # Match truncation colormap
                   cbar_kws={'label': 'Mean Affinity Score'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax3)
        ax3.set_title(f'{drug_name} - Mean Affinity Values ({mutation_range})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Residue Position')
        ax3.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                if not np.isnan(affinity_mean_matrix[aa_idx, pos_idx]):
                    ax3.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                            ha='center', va='center', fontsize=8, 
                            color='green', weight='bold')
        
        # Plot 4: Delta mean affinity score values with wild type
        ax4 = axes[1, 1]
        
        # Replace NaN with 0 for better visualization
        affinity_diff_matrix_vis = affinity_diff_matrix.copy()
        affinity_diff_matrix_vis = np.nan_to_num(affinity_diff_matrix_vis, nan=0.0)
        
        sns.heatmap(affinity_diff_matrix_vis, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu_r',  # Blue low -> Red high
                   vmin=0,
                   cbar_kws={'label': '|Affinity Score - Wild Type|'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax4)
        ax4.set_title(f'{drug_name} - Δ Affinity from Wild Type ({mutation_range})', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Residue Position')
        ax4.set_ylabel('Amino Acid')
        
        # Add green circles to mark original amino acids (zero difference)
        for pos_idx, position in enumerate(positions):
            original_aa = original_aa_dict.get(position)
            if original_aa:
                aa_idx = amino_acids.index(original_aa)
                ax4.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                        ha='center', va='center', fontsize=8, 
                        color='green', weight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            plot_filename = f"mutation_affinity_score_heatmaps_{protein_name}_{drug_name}_{mutation_range}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved mutation affinity score heatmaps to {plot_path}")
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Print summary
        
        # Calculate statistics for each matrix
        matrices = [
            ("IC50 Mean", ic50_mean_matrix),
            ("IC50 Difference", ic50_diff_matrix),
            ("Affinity Mean", affinity_mean_matrix),
            ("Affinity Difference", affinity_diff_matrix)
        ]
        
        for name, matrix in matrices:
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                avg_val = np.mean(valid_values)
            
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating mutation affinity score heatmaps: {str(e)}")
        return False

def plot_drug_comparison_delta_heatmap(protein_name, drug1_name, drug2_name, mutation_range, save_plot=True, show_plot=False):
    """
    Create a 2D heatmap comparing delta pIC50 values between two drugs.
    
    Args:
        protein_name (str): Name of the protein
        drug1_name (str): Name of the first drug
        drug2_name (str): Name of the second drug
        mutation_range (str): Mutation range (e.g., "790-810")
        save_plot (bool): Whether to save the plot
        show_plot (bool): Whether to show the plot
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load mutation metadata
        metadata_file = Path("affinity_metadata") / f"affinity_{protein_name}_mutation.csv"
        if not metadata_file.exists():
            print(f"Error: Metadata file not found: {metadata_file}")
            return False
            
        df = pd.read_csv(metadata_file)
        
        # Filter data for the specified protein and mutation range
        start_pos, end_pos = map(int, mutation_range.split('-'))
        positions = list(range(start_pos, end_pos + 1))
        
        # Define the 20 amino acids
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Filter data for both drugs with the same approach as the working function
        drug1_data = df[((df['protein_name'] == protein_name) | (df['protein_name'].str.startswith(protein_name + '-'))) & 
                        (df['drug_name'] == drug1_name) & 
                        (df['mutation_range'] == mutation_range)]
        
        drug2_data = df[((df['protein_name'] == protein_name) | (df['protein_name'].str.startswith(protein_name + '-'))) & 
                        (df['drug_name'] == drug2_name) & 
                        (df['mutation_range'] == mutation_range)]
        
        if drug1_data.empty or drug2_data.empty:
            print(f"Error: Missing data for one or both drugs: {drug1_name}, {drug2_name}")
            return False
        
        # Create matrices for delta values
        drug1_delta_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        drug2_delta_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        
        # Get original values from the ORIGINAL entry for both drugs
        original_entry1 = drug1_data[drug1_data['mutation_id'] == 'ORIGINAL']
        original_entry2 = drug2_data[drug2_data['mutation_id'] == 'ORIGINAL']
        
        if original_entry1.empty or original_entry2.empty:
            print(f"Error: Missing ORIGINAL entries for one or both drugs")
            return False
            
        original_ic50_1 = original_entry1.iloc[0]['ic50_pred_mean']
        original_ic50_2 = original_entry2.iloc[0]['ic50_pred_mean']
        
        # Get original amino acid for each position
        original_aa_dict = {}
        for _, row in drug1_data.iterrows():
            if row['is_mutation']:
                pos = row['mutation_position']
                if pos in positions:
                    original_aa = row['original_amino_acid']
                    original_aa_dict[pos] = original_aa
        
        # Fill the matrices using the same approach as the working function
        for pos_idx, position in enumerate(positions):
            # Get mutations for this position for both drugs
            pos_mutations1 = drug1_data[drug1_data['mutation_position'] == position]
            pos_mutations2 = drug2_data[drug2_data['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                # Find the mutation for this amino acid at this position for drug1
                mutation_row1 = pos_mutations1[pos_mutations1['new_amino_acid'] == amino_acid]
                mutation_row2 = pos_mutations2[pos_mutations2['new_amino_acid'] == amino_acid]
                
                if not mutation_row1.empty:
                    # Calculate delta for drug1
                    ic50_value1 = mutation_row1.iloc[0]['ic50_pred_mean']
                    drug1_delta_matrix[aa_idx, pos_idx] = abs(ic50_value1 - original_ic50_1)
                
                if not mutation_row2.empty:
                    # Calculate delta for drug2
                    ic50_value2 = mutation_row2.iloc[0]['ic50_pred_mean']
                    drug2_delta_matrix[aa_idx, pos_idx] = abs(ic50_value2 - original_ic50_2)
        
        # Calculate the difference: drug1_delta - drug2_delta
        # Positive = drug2 is better (favoring drug2 binding)
        # Negative = drug1 is better (favoring drug1 binding)
        delta_diff_matrix = drug1_delta_matrix - drug2_delta_matrix
        
        # Create matrices for direct IC50 values
        drug1_ic50_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        drug2_ic50_matrix = np.full((len(amino_acids), len(positions)), np.nan)
        
        # Fill IC50 matrices
        for pos_idx, position in enumerate(positions):
            pos_mutations1 = drug1_data[drug1_data['mutation_position'] == position]
            pos_mutations2 = drug2_data[drug2_data['mutation_position'] == position]
            
            for aa_idx, amino_acid in enumerate(amino_acids):
                mutation_row1 = pos_mutations1[pos_mutations1['new_amino_acid'] == amino_acid]
                mutation_row2 = pos_mutations2[pos_mutations2['new_amino_acid'] == amino_acid]
                
                if not mutation_row1.empty:
                    drug1_ic50_matrix[aa_idx, pos_idx] = mutation_row1.iloc[0]['ic50_pred_mean']
                if not mutation_row2.empty:
                    drug2_ic50_matrix[aa_idx, pos_idx] = mutation_row2.iloc[0]['ic50_pred_mean']
        
        # Calculate the difference: drug1_ic50 - drug2_ic50
        ic50_diff_matrix = drug1_ic50_matrix - drug2_ic50_matrix
        
        # Create the plot with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Delta comparison (drug1_delta - drug2_delta)
        sns.heatmap(delta_diff_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',  # Same as other plots: Red (negative) to Blue (positive)
                   center=0,  # Center at zero
                   cbar_kws={'label': f'Δ({drug1_name}) - Δ({drug2_name})'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax1)
        
        # Add green circles for zero differences (no preference)
        for pos_idx, position in enumerate(positions):
            for aa_idx, amino_acid in enumerate(amino_acids):
                if not np.isnan(delta_diff_matrix[aa_idx, pos_idx]):
                    if abs(delta_diff_matrix[aa_idx, pos_idx]) < 0.01:  # Very small difference
                        ax1.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                                ha='center', va='center', fontsize=8, 
                                color='green', weight='bold')
        
        # Set title and labels for left plot
        ax1.set_title(f'Δ(pIC50, {drug1_name}) - Δ(pIC50, {drug2_name}) ({mutation_range})', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Residue Position')
        ax1.set_ylabel('Amino Acid')
        
        # Right plot: Direct IC50 comparison (drug1_ic50 - drug2_ic50)
        sns.heatmap(ic50_diff_matrix, 
                   xticklabels=positions,
                   yticklabels=amino_acids,
                   annot=False,
                   cmap='RdYlBu',  # Same as other plots: Red (negative) to Blue (positive)
                   center=0,  # Center at zero
                   cbar_kws={'label': f'{drug1_name} pIC50 - {drug2_name} pIC50'},
                   linewidths=0.1,
                   linecolor='white',
                   ax=ax2)
        
        # Add green circles for zero differences (no preference)
        for pos_idx, position in enumerate(positions):
            for aa_idx, amino_acid in enumerate(amino_acids):
                if not np.isnan(ic50_diff_matrix[aa_idx, pos_idx]):
                    if abs(ic50_diff_matrix[aa_idx, pos_idx]) < 0.01:  # Very small difference
                        ax2.text(pos_idx + 0.5, aa_idx + 0.5, '●', 
                                ha='center', va='center', fontsize=8, 
                                color='green', weight='bold')
        
        # Set title and labels for right plot
        ax2.set_title(f'{drug1_name} pIC50 - {drug2_name} pIC50 ({mutation_range})', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Residue Position')
        ax2.set_ylabel('Amino Acid')
        
        # Main title for the entire figure
        fig.suptitle(f'{protein_name} - Drug Comparison Analysis ({mutation_range})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            output_dir = Path("affinity_plots")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"drug_comparison_delta_{protein_name}_{drug1_name}_vs_{drug2_name}_{mutation_range}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: Saved drug comparison delta heatmap to {output_file}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        # Print summary statistics
        
        # Calculate statistics excluding NaN values
        valid_diffs = delta_diff_matrix[~np.isnan(delta_diff_matrix)]
        valid_ic50_diffs = ic50_diff_matrix[~np.isnan(ic50_diff_matrix)]
        
        
        return True
        
    except Exception as e:
        print(f"Error: Error creating drug comparison delta heatmap: {str(e)}")
        return False

def main():
    """Main function to run the plotting functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot affinity probability heatmaps')
    parser.add_argument('protein', nargs='?', help='Protein name (e.g., EGFR)')
    parser.add_argument('drug', nargs='?', help='Drug name (e.g., osimertinib)')
    parser.add_argument('--all', action='store_true', help='Plot all available combinations')
    parser.add_argument('--protein-compare', action='store_true', help='Plot all drugs for a protein in one image')
    parser.add_argument('--differences', action='store_true', help='Plot column differences for a protein')
    parser.add_argument('--mutation', action='store_true', help='Plot mutation heatmap')
    parser.add_argument('--mutation-variants', action='store_true', help='Plot mutation heatmap variants (4 plots)')
    parser.add_argument('--mutation-probability', action='store_true', help='Plot mutation probability heatmaps (2 plots)')
    parser.add_argument('--mutation-affinity-score', action='store_true', help='Plot mutation affinity score heatmaps (2 plots, reversed colors)')
    parser.add_argument('--mutation-range', help='Mutation range (e.g., 790-810)')
    parser.add_argument('--drugs', nargs='+', help='Specific drugs to include (e.g., --drugs gefitinib osimertinib)')
    parser.add_argument('--combined', action='store_true', help='Plot combined comparison and mutation heatmap')
    parser.add_argument('--drug-comparison-delta', action='store_true', help='Plot drug comparison delta heatmap between two drugs')
    
    args = parser.parse_args()
    
    if args.combined and args.protein and args.drug and args.mutation_range:
        # Plot combined heatmap with comparison above and mutation below
        success = plot_combined_heatmap(args.protein, args.drug, args.mutation_range, args.drugs)
        
        if success:
            print("Success: Combined heatmap created successfully!")
        else:
            print("Error: Failed to create combined heatmap")
            
    elif args.mutation_probability and args.protein and args.drug and args.mutation_range:
        # Plot mutation probability heatmaps (2 plots) for a specific protein-drug-mutation combination
        success = plot_mutation_probability_heatmaps(args.protein, args.drug, args.mutation_range)
        
        if success:
            print("Success: Mutation probability heatmaps created successfully!")
        else:
            print("Error: Failed to create mutation probability heatmaps")
            
    elif args.mutation_affinity_score and args.protein and args.drug and args.mutation_range:
        # Plot mutation affinity score heatmaps (2 plots) for a specific protein-drug-mutation combination
        success = plot_mutation_affinity_score_heatmaps(args.protein, args.drug, args.mutation_range)
        
        if success:
            print("Success: Mutation affinity score heatmaps created successfully!")
        else:
            print("Error: Failed to create mutation affinity score heatmaps")
            
    elif args.drug_comparison_delta and args.protein and args.drugs and len(args.drugs) == 2 and args.mutation_range:
        # Plot drug comparison delta heatmap between two drugs
        drug1, drug2 = args.drugs[0], args.drugs[1]
        success = plot_drug_comparison_delta_heatmap(args.protein, drug1, drug2, args.mutation_range)
        
        if success:
            print("Success: Drug comparison delta heatmap created successfully!")
        else:
            print("Error: Failed to create drug comparison delta heatmap")
            
    elif args.mutation_variants and args.protein and args.drug and args.mutation_range:
        # Plot mutation heatmap variants (4 plots) for a specific protein-drug-mutation combination
        success = plot_mutation_affinity_heatmap_variants(args.protein, args.drug, args.mutation_range)
        
        if success:
            print("Success: Mutation heatmap variants created successfully!")
        else:
            print("Error: Failed to create mutation heatmap variants")
            
    elif args.mutation and args.protein and args.drug and args.mutation_range:
        # Plot mutation heatmap for a specific protein-drug-mutation combination
        success = plot_mutation_affinity_heatmap(args.protein, args.drug, args.mutation_range)
        
        if success:
            print("Success: Mutation heatmap created successfully!")
        else:
            print("Error: Failed to create mutation heatmap")
            
    elif args.differences and args.protein:
        # Plot column differences for a specific protein
        success = plot_column_differences(args.protein, args.drugs)
        
        if success:
            print("Success: Column differences plot created successfully!")
        else:
            print("Error: Failed to create column differences plot")
            
    elif args.protein_compare and args.protein:
        # Plot all drugs for a specific protein in one image
        success = plot_protein_comparison_heatmap(args.protein, args.drugs)
        
        if success:
            print("Success: Protein comparison heatmap created successfully!")
        else:
            print("Error: Failed to create protein comparison heatmap")
            
    elif args.all:
        # Plot all available combinations
        print("🎨 Creating heatmaps for all available combinations...")
        
        # Get all available CSV files
        metadata_dir = Path("affinity_metadata")
        csv_files = list(metadata_dir.glob("affinity_*.csv"))
        
        for csv_file in csv_files:
            protein_name = csv_file.stem.replace("affinity_", "")
            
            # Read the CSV to find available drugs
            df = pd.read_csv(csv_file)
            available_drugs = df['drug_name'].unique()
            
            for drug_name in available_drugs:
                plot_affinity_heatmap(protein_name, drug_name, show_plot=False)
        
        print("\nSuccess: All heatmaps created successfully!")
        
    elif args.protein and args.drug:
        # Plot specific protein-drug combination
        success = plot_affinity_heatmap(args.protein, args.drug)
        
        if success:
            print("Success: Heatmap created successfully!")
        else:
            print("Error: Failed to create heatmap")
    else:
        print("Usage:")
        print("  python plot_affinity_heatmap.py EGFR osimertinib")
        print("  python plot_affinity_heatmap.py EGFR --protein-compare")
        print("  python plot_affinity_heatmap.py EGFR --differences")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --mutation --mutation-range 790-810")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --mutation-variants --mutation-range 790-810")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --combined --mutation-range 790-810")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --mutation-variants --mutation-range 790-810")
        print("  python plot_affinity_heatmap.py EGFR --protein-compare --drugs gefitinib osimertinib")
        print("  python plot_affinity_heatmap.py EGFR --differences --drugs gefitinib osimertinib")
        print("  python plot_affinity_heatmap.py --all")
        
        print("\nExamples:")
        print("  python plot_affinity_heatmap.py EGFR osimertinib")
        print("  python plot_affinity_heatmap.py EGFR gefitinib")
        print("  python plot_affinity_heatmap.py EGFR --protein-compare")
        print("  python plot_affinity_heatmap.py EGFR --differences")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --mutation --mutation-range 790-810")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --combined --mutation-range 790-810")
        print("  python plot_affinity_heatmap.py EGFR osimertinib --combined --mutation-range 790-810 --drugs gefitinib osimertinib")
        print("  python plot_affinity_heatmap.py EGFR --protein-compare --drugs gefitinib osimertinib")
        print("  python plot_affinity_heatmap.py EGFR --differences --drugs gefitinib osimertinib")
        print("  python plot_affinity_heatmap.py --all")

if __name__ == "__main__":
    main() 
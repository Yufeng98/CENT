#!/usr/bin/env python3
"""
Script to compare GPU utilization across multiple NCU profiling CSV files
Averages all "avg" columns across the 32 kernel launches in each file
"""
import sys
import csv
import os
import glob
from typing import Dict, List
import pandas as pd

def extract_avg_metrics(csv_file: str) -> Dict[str, float]:
    """
    Extract and average all columns containing 'avg' from NCU CSV file
    
    Args:
        csv_file: Path to the NCU CSV file
        
    Returns:
        Dictionary mapping column names to averaged values
    """
    avg_metrics = {}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            
            # Read header row
            header = next(reader)
            
            # Skip units row
            next(reader)
            
            # Find all columns containing 'avg'
            avg_columns = {}
            for i, col_name in enumerate(header):
                if 'avg' in col_name.lower():
                    avg_columns[i] = col_name
            
            if not avg_columns:
                print(f"Warning: No 'avg' columns found in {csv_file}")
                return avg_metrics
            
            # Collect values for each avg column
            column_values = {col_name: [] for col_name in avg_columns.values()}
            
            # Read data rows
            for row_num, row in enumerate(reader):
                if len(row) > max(avg_columns.keys()):
                    for col_idx, col_name in avg_columns.items():
                        try:
                            value = float(row[col_idx])
                            column_values[col_name].append(value)
                        except (ValueError, IndexError):
                            # Skip invalid values
                            continue
            
            # Calculate averages
            for col_name, values in column_values.items():
                if values:
                    avg_metrics[col_name] = sum(values) / len(values)
                    print(f"  {col_name}: {len(values)} values, avg = {avg_metrics[col_name]:.6f}")
                else:
                    avg_metrics[col_name] = 0.0
                    print(f"  {col_name}: No valid values found")
                    
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
    
    return avg_metrics

def process_multiple_files(file_pattern: str) -> pd.DataFrame:
    """
    Process multiple CSV files and create comparison DataFrame
    
    Args:
        file_pattern: Glob pattern to match CSV files
        
    Returns:
        DataFrame with averaged metrics for comparison
    """
    # Find all matching CSV files
    csv_files = sorted(glob.glob(file_pattern))
    
    if not csv_files:
        print(f"No files found matching pattern: {file_pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")
    
    # Process each file
    all_metrics = {}
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        filename = os.path.basename(csv_file)
        
        # Extract averaged metrics
        metrics = extract_avg_metrics(csv_file)
        
        if metrics:
            all_metrics[filename] = metrics
            print(f"  Extracted {len(metrics)} averaged metrics")
        else:
            print(f"  No metrics extracted from {csv_file}")
    
    if not all_metrics:
        print("No metrics extracted from any files")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(all_metrics, orient='index')
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    return df

def save_comparison_results(df: pd.DataFrame, output_file: str = "gpu_utilization_comparison.csv"):
    """
    Save comparison results to CSV file
    
    Args:
        df: DataFrame with comparison data
        output_file: Output filename
    """
    if df.empty:
        print("No data to save")
        return
    
    # Sort columns by name for better organization
    df = df.reindex(sorted(df.columns), axis=1)

    # Sort rows by batch size number first
    def extract_batch_number(filename):
        try:
            # Extract number after "batch-" in filename
            parts = filename.split('-')
            if len(parts) > 1:
                return int(parts[-1].split('.')[0])  # Remove .csv extension
            return 0
        except:
            return 0
    
    # Sort DataFrame rows by batch size
    df = df.iloc[df.index.to_series().apply(extract_batch_number).argsort()]
    
    
    # Save to CSV
    # df.to_csv(output_file)
    print(f"\nComparison results saved to: {output_file}")
    print(f"Shape: {df.shape[0]} files x {df.shape[1]} metrics")
    
    # Display key metrics for quick comparison
    key_metrics = [
        'sm__issue_active.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__bytes.avg',
        'dram__bytes_read.avg',
        'dram__bytes_write.avg',
    ]
    
    print(f"\nKey GPU Utilization Metrics Summary:")
    print("=" * 60)

    for metric in df.columns:
        if metric in key_metrics:
            print(f"\n{metric}:")
            # Since df is already sorted by batch size, iterate in order
            for filename in df.index:
                value = df.loc[filename, metric]
                print(f"  {filename:<40} {value:>10.4f}%")

def main():
    """Main function"""
    # You can modify this pattern to match your files
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    file_pattern = folder+"/ncu_profile-1gpu-batch-*.csv"
    
    print("GPU Utilization Comparison Tool")
    print("=" * 50)
    
    # Process files
    comparison_df = process_multiple_files(file_pattern)
    
    if not comparison_df.empty:
        # Save results
        save_comparison_results(comparison_df)
        
        # Also create a transposed version (metrics as rows, files as columns)
        transposed_df = comparison_df.transpose()
        transposed_output = "gpu_utilization_comparison_transposed.csv"
        # transposed_df.to_csv(transposed_output)
        print(f"Transposed results saved to: {transposed_output}")
        
        # Print basic statistics
        print(f"\nBasic Statistics:")
        print(f"Number of files processed: {comparison_df.shape[0]}")
        print(f"Number of metrics compared: {comparison_df.shape[1]}")
        
        # # Show files with highest/lowest overall utilization
        # if 'sm__issue_active.avg.pct_of_peak_sustained_elapsed' in comparison_df.columns:
        #     utilization_col = 'sm__issue_active.avg.pct_of_peak_sustained_elapsed'
        #     highest = comparison_df[utilization_col].idxmax()
        #     lowest = comparison_df[utilization_col].idxmin()
            
        #     print(f"\nHighest SM Issue Active: {highest} ({comparison_df.loc[highest, utilization_col]:.4f}%)")
        #     print(f"Lowest SM Issue Active:  {lowest} ({comparison_df.loc[lowest, utilization_col]:.4f}%)")

if __name__ == "__main__":
    main()
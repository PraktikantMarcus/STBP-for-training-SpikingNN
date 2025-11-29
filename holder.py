def create_detailed_latex_table(csv_path, output_path="results_table_detailed.tex"):
    """
    Create LaTeX table showing best configuration for each bit budget.
    Shows: Bits | m | n | Rounding | Overflow | Accuracy
    """
    df = pd.read_csv(csv_path)
    
    # Get best configuration for each bit budget
    best_per_bit = df.loc[df.groupby('bits')['acc'].idxmax()]
    
    # Select and reorder columns
    result_table = best_per_bit[['bits', 'm', 'n', 'rounding', 'overflow', 'acc']].copy()
    result_table.columns = ['Bits', 'm', 'n', 'Rounding', 'Overflow', 'Accuracy (%)']
    
    # Convert to integers
    result_table['Bits'] = result_table['Bits'].astype(int)
    result_table['m'] = result_table['m'].astype(int)
    result_table['n'] = result_table['n'].astype(int)
    
    # Sort by bits
    result_table = result_table.sort_values('Bits').reset_index(drop=True)
    
    # Style: format only the Accuracy column with 2 decimals
    styled = result_table.style.format({
        'Bits': '{:d}',
        'm': '{:d}',
        'n': '{:d}',
        'Accuracy (%)': '{:.2f}'
    })
    
    latex_table = styled.to_latex(
        caption="Best quantization configuration for each bit budget",
        label="tab:best_configs",
        position="htbp",
        position_float="centering",
        hrules=True,
        column_format="cccccc",
        index=False  # This removes the first numbering column
    )
    
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ Detailed LaTeX table saved to: {output_path}")
    print(f"\nPreview:")
    print(result_table)
    
    return result_table
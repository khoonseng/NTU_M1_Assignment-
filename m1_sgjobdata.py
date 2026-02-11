
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from DuckDB database.
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        Tuple of (jobs_df, categories_df, jobcategories_df)
    """
    print("Loading data from database...")
    
    # Connect to DuckDB
    con = duckdb.connect(db_path)
    
    # Import all 3 tables into dataframes
    jobs_df = con.execute("SELECT * FROM Jobs").fetchdf()
    categories_df = con.execute("SELECT * FROM Categories").fetchdf()
    jobcategories_df = con.execute("SELECT * FROM JobCategories").fetchdf()
    
    # Close connection
    con.close()
    
    # Display basic info
    print("\n=== DATA LOADED ===")
    print(f"Jobs DataFrame: Shape = {jobs_df.shape}")
    print(f"Categories DataFrame: Shape = {categories_df.shape}")
    print(f"JobCategories DataFrame: Shape = {jobcategories_df.shape}")
    
    return jobs_df, categories_df, jobcategories_df


def clean_and_prepare_data(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the jobs data.
    
    Args:
        jobs_df: Raw jobs dataframe
        
    Returns:
        Cleaned and prepared dataframe
    """
    print("\n=== CLEANING AND PREPARING DATA ===")
    
    # Create a copy to avoid modifying original
    filtered_jobs = jobs_df.copy()
    
    # Remove redundant columns
    filtered_jobs = filtered_jobs.drop(columns=['occupationId', 'status_id'])
    print(f"Dropped redundant columns. New shape: {filtered_jobs.shape}")
    
    # Convert columns to appropriate data types
    numeric_cols = ['minimumYearsExperience', 'salary_maximum', 'salary_minimum']
    for col in numeric_cols:
        filtered_jobs[col] = pd.to_numeric(filtered_jobs[col], errors='coerce')
        print(f"Converted {col} to numeric. Missing values: {filtered_jobs[col].isna().sum()}")
    
    # Clean company names: convert to title case
    filtered_jobs['postedCompany_name'] = filtered_jobs['postedCompany_name'].str.title()
    
    return filtered_jobs


def create_experience_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create experience level categories from minimumYearsExperience.
    
    Args:
        df: DataFrame with minimumYearsExperience column
        
    Returns:
        DataFrame with new 'experience_level' column
    """
    print("\n=== CREATING EXPERIENCE LEVEL CATEGORIES ===")
    
    # Define bins and labels for experience levels in 5-year intervals
    bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, np.inf]
    labels = [
        '0-5 years',
        '6-10 years',
        '11-15 years',
        '16-20 years',
        '21-25 years',
        '26-30 years',
        '31-35 years',
        '36-40 years',
        '40+ years'
    ]
    
    # Create the new column 'experience_level' using pandas.cut
    df['experience_level'] = pd.cut(
        df['minimumYearsExperience'], 
        bins=bins, 
        labels=labels, 
        right=True
    )
    
    # Display distribution
    exp_counts = df['experience_level'].value_counts().sort_index()
    print("Experience level distribution:")
    for level, count in exp_counts.items():
        print(f"  {level}: {count:,} jobs ({count/len(df)*100:.1f}%)")
    
    return df


def create_industry_categories(categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create broader industry categories from detailed job categories.
    
    Args:
        categories_df: DataFrame with detailed job categories
        
    Returns:
        DataFrame with new 'Industry_Category' column
    """
    print("\n=== CREATING INDUSTRY CATEGORIES ===")
    
    industry_mapping = {
        # Business & Professional Services
        'Accounting / Auditing / Taxation': 'Business & Professional Services',
        'Admin / Secretarial': 'Business & Professional Services',
        'Banking and Finance': 'Business & Professional Services',
        'Consulting': 'Business & Professional Services',
        'Insurance': 'Business & Professional Services',
        'Legal': 'Business & Professional Services',
        'Professional Services': 'Business & Professional Services',
        'Risk Management': 'Business & Professional Services',
        
        # Engineering & Technical
        'Engineering': 'Engineering & Technical',
        'Information Technology': 'Engineering & Technical',
        'Precision Engineering': 'Engineering & Technical',
        'Repair and Maintenance': 'Engineering & Technical',
        'Telecommunications': 'Engineering & Technical',
        
        # Sales, Marketing & Retail
        'Advertising / Media': 'Sales, Marketing & Retail',
        'Events / Promotions': 'Sales, Marketing & Retail',
        'Marketing / Public Relations': 'Sales, Marketing & Retail',
        'Sales / Retail': 'Sales, Marketing & Retail',
        'Wholesale Trade': 'Sales, Marketing & Retail',
        
        # Healthcare & Life Sciences
        'Healthcare / Pharmaceutical': 'Healthcare & Life Sciences',
        'Medical / Therapy Services': 'Healthcare & Life Sciences',
        'Environment / Health': 'Healthcare & Life Sciences',
        
        # Hospitality & Services
        'Customer Service': 'Hospitality & Services',
        'F&B': 'Hospitality & Services',
        'Hospitality': 'Hospitality & Services',
        'Personal Care / Beauty': 'Hospitality & Services',
        'Travel / Tourism': 'Hospitality & Services',
        
        # Creative & Design
        'Architecture / Interior Design': 'Creative & Design',
        'Design': 'Creative & Design',
        'Entertainment': 'Creative & Design',
        
        # Manufacturing & Logistics
        'Logistics / Supply Chain': 'Manufacturing & Logistics',
        'Manufacturing': 'Manufacturing & Logistics',
        'Purchasing / Merchandising': 'Manufacturing & Logistics',
        
        # Construction & Real Estate
        'Building and Construction': 'Construction & Real Estate',
        'Real Estate / Property Management': 'Construction & Real Estate',
        
        # Management & HR
        'General Management': 'Management & HR',
        'Human Resources': 'Management & HR',
        
        # Education & Training
        'Education and Training': 'Education & Training',
        
        # Research & Development
        'Sciences / Laboratory / R&D': 'Research & Development',
        
        # Public Sector & Social Services
        'Public / Civil Service': 'Public Sector & Social Services',
        'Social Services': 'Public Sector & Social Services',
        
        # General & Miscellaneous
        'General Work': 'General & Support Services',
        'Security and Investigation': 'General & Support Services',
        'Others': 'Other / Miscellaneous'
    }
    
    # Apply mapping to categories dataframe
    categories_df['Industry_Category'] = categories_df['Cat_Name'].map(industry_mapping)
    
    # Handle any unmapped categories
    unmapped = categories_df[categories_df['Industry_Category'].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} categories not mapped to industry")
        categories_df['Industry_Category'] = categories_df['Industry_Category'].fillna('Other / Miscellaneous')
    
    print(f"Created {categories_df['Industry_Category'].nunique()} industry categories")
    print("\nIndustry distribution:")
    industry_counts = categories_df['Industry_Category'].value_counts()
    for industry, count in industry_counts.items():
        print(f"  {industry}: {count} sub-categories")
    
    return categories_df


def join_datasets(jobs_df: pd.DataFrame, 
                  categories_df: pd.DataFrame, 
                  jobcategories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join all three datasets into one comprehensive dataframe.
    
    Args:
        jobs_df: Jobs dataframe
        categories_df: Categories dataframe with industry mapping
        jobcategories_df: Job-Categories mapping dataframe
        
    Returns:
        Joined dataframe
    """
    print("\n=== JOINING DATASETS ===")
    
    # First, join JobCategories with Categories to get Industry_Category for each job
    job_with_categories = pd.merge(
        jobcategories_df, 
        categories_df[['Cat_ID', 'Industry_Category']], 
        on='Cat_ID', 
        how='left'
    )
    
    # For jobs with multiple categories, take the first category
    job_industry = job_with_categories.groupby('metadata_jobPostId').first().reset_index()
    
    # Now join with the jobs dataframe
    final_df = pd.merge(
        jobs_df, 
        job_industry[['metadata_jobPostId', 'Industry_Category']], 
        left_on='metadata_jobPostId', 
        right_on='metadata_jobPostId', 
        how='left'
    )
    
    print(f"Final merged dataframe shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Check for missing values
    print("\nMissing values:")
    for col in ['Industry_Category', 'salary_minimum', 'salary_maximum', 'minimumYearsExperience']:
        missing = final_df[col].isna().sum()
        percent = missing / len(final_df) * 100
        print(f"  {col}: {missing:,} ({percent:.1f}%)")
    
    return final_df


def apply_realistic_salary_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply realistic salary filters based on Singapore market rates and experience levels.
    
    Args:
        df: DataFrame containing salary and experience data
        
    Returns:
        Filtered DataFrame with realistic salary ranges
    """
    print("\n=== APPLYING REALISTIC SALARY FILTERS ===")
    print(f"Initial data shape: {df.shape}")
    
    # Make a copy
    filtered_df = df.copy()
    
    # Step 1: Basic sanity checks
    # Remove salaries below reasonable minimum (SGD 1,000/month)
    min_reasonable = 1000
    mask_min = (filtered_df['salary_minimum'] >= min_reasonable)
    filtered_df = filtered_df[mask_min]
    print(f"After removing salaries < ${min_reasonable:,}: {filtered_df.shape}")
    
    # Remove salaries above reasonable maximum (SGD 50,000/month)
    max_reasonable = 50000
    mask_max = (filtered_df['salary_maximum'] <= max_reasonable)
    filtered_df = filtered_df[mask_max]
    print(f"After removing salaries > ${max_reasonable:,}: {filtered_df.shape}")
    
    # Ensure maximum >= minimum
    mask_valid_range = (filtered_df['salary_maximum'] >= filtered_df['salary_minimum'])
    filtered_df = filtered_df[mask_valid_range]
    print(f"After ensuring max >= min: {filtered_df.shape}")
    
    # Remove rows where difference is too large (likely error)
    salary_diff = filtered_df['salary_maximum'] - filtered_df['salary_minimum']
    mask_reasonable_diff = (salary_diff <= 30000)  # Max 30k difference
    filtered_df = filtered_df[mask_reasonable_diff]
    print(f"After removing extreme salary ranges (>30k diff): {filtered_df.shape}")
    
    # Step 2: Experience-level specific filtering
    print("\nApplying experience-level specific filters:")
    
    # Define reasonable ranges by experience level (based on Singapore market)
    salary_ranges = {
        '0-5 years': {'min': 2000, 'max': 12000},
        '6-10 years': {'min': 3500, 'max': 20000},
        '11-15 years': {'min': 5000, 'max': 30000},
        '16-20 years': {'min': 5000, 'max': 50000},
        '21-25 years': {'min': 5000, 'max': 50000},
        '26-30 years': {'min': 5000, 'max': 50000},
        '31-35 years': {'min': 5000, 'max': 50000},
        '36-40 years': {'min': 5000, 'max': 50000},
        '40+ years': {'min': 5000, 'max': 50000}
    }
    
    # Create mask for each experience level
    masks = []
    for exp_level, ranges in salary_ranges.items():
        # For this experience level, keep if within range
        # For other levels, keep regardless (will be filtered by their own mask)
        mask = (
            (filtered_df['experience_level'] == exp_level) & 
            (filtered_df['salary_minimum'] >= ranges['min']) & 
            (filtered_df['salary_maximum'] <= ranges['max'])
        ) | (filtered_df['experience_level'] != exp_level)
        
        # Count removals for this level
        level_data = filtered_df[filtered_df['experience_level'] == exp_level]
        removed = level_data[
            (level_data['salary_minimum'] < ranges['min']) | 
            (level_data['salary_maximum'] > ranges['max'])
        ].shape[0]
        
        if removed > 0:
            total_level = len(level_data)
            percent_removed = removed / total_level * 100 if total_level > 0 else 0
            print(f"  {exp_level}: Removed {removed:,}/{total_level:,} ({percent_removed:.1f}%)")
        
        masks.append(mask)
    
    # Combine all masks
    if masks:
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask & mask
        filtered_df = filtered_df[final_mask]
    
    print(f"\nFinal shape after realistic filters: {filtered_df.shape}")
    
    return filtered_df


def apply_iqr_outlier_filter(df: pd.DataFrame, salary_col: str = 'salary_average') -> pd.DataFrame:
    """
    Apply Interquartile Range (IQR) method to remove outliers for each experience level.
    
    Args:
        df: DataFrame containing salary and experience level data
        salary_col: Name of the salary column to filter
        
    Returns:
        DataFrame with outliers removed
    """
    print("\n=== APPLYING IQR OUTLIER FILTER ===")
    
    # Ensure we have the salary column
    if salary_col not in df.columns and 'salary_minimum' in df.columns and 'salary_maximum' in df.columns:
        df['salary_average'] = (df['salary_minimum'] + df['salary_maximum']) / 2
        salary_col = 'salary_average'
    
    filtered_dfs = []
    total_removed = 0
    
    for exp_level in df['experience_level'].dropna().unique():
        level_data = df[df['experience_level'] == exp_level].copy()
        
        if len(level_data) < 10:  # Skip if too few samples
            filtered_dfs.append(level_data)
            continue
        
        # Calculate IQR
        Q1 = level_data[salary_col].quantile(0.25)
        Q3 = level_data[salary_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (using 1.5 * IQR)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Ensure lower bound is reasonable (not negative)
        lower_bound = max(lower_bound, 1000)
        
        # Filter data
        mask = (level_data[salary_col] >= lower_bound) & (level_data[salary_col] <= upper_bound)
        filtered_level = level_data[mask]
        
        removed = len(level_data) - len(filtered_level)
        total_removed += removed
        
        if removed > 0:
            percent_removed = removed / len(level_data) * 100
            print(f"  {exp_level}: Removed {removed:,}/{len(level_data):,} ({percent_removed:.1f}%)")
        
        filtered_dfs.append(filtered_level)
    
    result = pd.concat(filtered_dfs, ignore_index=True)
    print(f"\nTotal rows removed by IQR: {total_removed:,}")
    print(f"Final shape after IQR: {result.shape}")
    
    return result


def analyze_salary_distribution(df: pd.DataFrame, title: str = "Data") -> pd.DataFrame:
    """
    Analyze salary distribution and generate summary statistics.
    
    Args:
        df: DataFrame to analyze
        title: Title for the analysis
        
    Returns:
        Summary statistics dataframe
    """
    print(f"\n=== SALARY ANALYSIS: {title} ===")
    
    # Calculate average salary if not present
    if 'salary_average' not in df.columns:
        df['salary_average'] = (df['salary_minimum'] + df['salary_maximum']) / 2
    
    # Generate summary statistics by experience level
    summary = df.groupby('experience_level')['salary_average'].agg([
        ('count', 'size'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75))
    ]).round(2)
    
    # Display summary
    print("\nSummary Statistics by Experience Level:")
    print(summary)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total jobs: {len(df):,}")
    print(f"  Overall mean salary: ${df['salary_average'].mean():,.2f}")
    print(f"  Overall median salary: ${df['salary_average'].median():,.2f}")
    print(f"  Overall salary range: ${df['salary_average'].min():,.0f} - ${df['salary_average'].max():,.0f}")
    
    return summary


def create_visualizations(original_df: pd.DataFrame, 
                         filtered_df: pd.DataFrame, 
                         iqr_df: pd.DataFrame) -> None:
    """
    Create visualizations comparing original, filtered, and IQR-cleaned data.
    
    Args:
        original_df: Original dataframe (with outliers)
        filtered_df: Realistically filtered dataframe
        iqr_df: IQR-filtered dataframe
    """
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Calculate average salaries for all datasets
    for df, name in [(original_df, 'original'), (filtered_df, 'filtered'), (iqr_df, 'iqr')]:
        if 'salary_average' not in df.columns:
            df['salary_average'] = (df['salary_minimum'] + df['salary_maximum']) / 2
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Salary Data Analysis: Before and After Filtering', fontsize=16, fontweight='bold')
    
    datasets = [
        (original_df, 'Original Data (With Outliers)', 'tab:blue'),
        (filtered_df, 'Realistically Filtered', 'tab:green'),
        (iqr_df, 'IQR Filtered (Conservative)', 'tab:orange')
    ]
    
    # Define experience levels order
    exp_levels = [
        '0-5 years', '6-10 years', '11-15 years', '16-20 years',
        '21-25 years', '26-30 years', '31-35 years', '36-40 years', '40+ years'
    ]
    
    # Row 1: Box plots
    for i, (df, title, color) in enumerate(datasets):
        ax = axes[0, i]
        
        # Prepare data for boxplot
        box_data = []
        valid_levels = []
        for level in exp_levels:
            level_data = df[df['experience_level'] == level]['salary_average'].dropna()
            if len(level_data) > 0:
                box_data.append(level_data)
                valid_levels.append(level)
        
        if box_data:
            bp = ax.boxplot(box_data, labels=valid_levels, patch_artist=True)
            # Set box colors
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.6)
            
            ax.set_title(title)
            ax.set_xlabel('Experience Level')
            ax.set_ylabel('Salary (SGD)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add sample size annotation
            total_samples = sum(len(data) for data in box_data)
            ax.text(0.02, 0.98, f'n = {total_samples:,}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Mean salary by experience level
    for i, (df, title, color) in enumerate(datasets):
        ax = axes[1, i]
        
        # Calculate mean salary by experience level
        mean_salaries = df.groupby('experience_level')['salary_average'].mean().reindex(exp_levels)
        
        # Plot
        bars = ax.bar(range(len(exp_levels)), mean_salaries, color=color, alpha=0.7)
        ax.set_title(f'{title}\nMean Salary by Experience')
        ax.set_xlabel('Experience Level')
        ax.set_ylabel('Mean Salary (SGD)')
        ax.set_xticks(range(len(exp_levels)))
        ax.set_xticklabels(exp_levels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, salary in zip(bars, mean_salaries):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                   f'${salary:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Row 3: Data retention and additional metrics
    # Subplot 1: Data retention rate
    ax1 = axes[2, 0]
    retention_rates = []
    for level in exp_levels:
        original_count = len(original_df[original_df['experience_level'] == level])
        final_count = len(iqr_df[iqr_df['experience_level'] == level])
        if original_count > 0:
            retention = (final_count / original_count) * 100
        else:
            retention = 0
        retention_rates.append(retention)
    
    bars = ax1.bar(range(len(exp_levels)), retention_rates, color='teal', alpha=0.7)
    ax1.set_title('Data Retention After Filtering')
    ax1.set_xlabel('Experience Level')
    ax1.set_ylabel('Data Retained (%)')
    ax1.set_xticks(range(len(exp_levels)))
    ax1.set_xticklabels(exp_levels, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, rate in zip(bars, retention_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Salary range by experience (IQR filtered data)
    ax2 = axes[2, 1]
    salary_ranges = []
    for level in exp_levels:
        level_data = iqr_df[iqr_df['experience_level'] == level]['salary_average']
        if len(level_data) > 0:
            salary_range = level_data.max() - level_data.min()
            salary_ranges.append(salary_range)
        else:
            salary_ranges.append(0)
    
    bars = ax2.bar(range(len(exp_levels)), salary_ranges, color='coral', alpha=0.7)
    ax2.set_title('Salary Range by Experience\n(IQR Filtered Data)')
    ax2.set_xlabel('Experience Level')
    ax2.set_ylabel('Salary Range (SGD)')
    ax2.set_xticks(range(len(exp_levels)))
    ax2.set_xticklabels(exp_levels, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Correlation heatmap (IQR filtered data)
    ax3 = axes[2, 2]
    
    # Prepare data for correlation
    corr_data = iqr_df[['minimumYearsExperience', 'salary_minimum', 'salary_maximum', 'salary_average']].copy()
    corr_matrix = corr_data.corr()
    
    # Create heatmap
    im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax3.set_title('Correlation Heatmap\n(IQR Filtered Data)')
    
    # Set ticks and labels
    tick_labels = ['Years Exp', 'Min Salary', 'Max Salary', 'Avg Salary']
    ax3.set_xticks(range(len(tick_labels)))
    ax3.set_yticks(range(len(tick_labels)))
    ax3.set_xticklabels(tick_labels, rotation=45)
    ax3.set_yticklabels(tick_labels)
    
    # Add correlation values
    for i in range(len(tick_labels)):
        for j in range(len(tick_labels)):
            value = corr_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax3.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig('salary_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def industry_salary_analysis(df: pd.DataFrame) -> None:
    """
    Analyze salary by industry category.
    
    Args:
        df: Cleaned dataframe with industry information
    """
    print("\n=== INDUSTRY SALARY ANALYSIS ===")
    
    # Ensure we have average salary
    if 'salary_average' not in df.columns:
        df['salary_average'] = (df['salary_minimum'] + df['salary_maximum']) / 2
    
    # Top industries by average salary
    industry_salaries = df.groupby('Industry_Category')['salary_average'].agg([
        ('count', 'size'),
        ('mean_salary', 'mean'),
        ('median_salary', 'median'),
        ('min_salary', 'min'),
        ('max_salary', 'max')
    ]).round(2)
    
    # Sort by mean salary
    industry_salaries = industry_salaries.sort_values('mean_salary', ascending=False)
    
    print("\nTop 10 Industries by Average Salary:")
    print(industry_salaries.head(10))
    
    print("\nBottom 10 Industries by Average Salary:")
    print(industry_salaries.tail(10))
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Top 15 industries by average salary
    top_15 = industry_salaries.head(15)
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(top_15)), top_15['mean_salary'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15.index)
    ax1.set_xlabel('Average Salary (SGD)')
    ax1.set_title('Top 15 Industries by Average Salary')
    ax1.invert_yaxis()  # Highest at top
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(top_15.iterrows()):
        ax1.text(row['mean_salary'] + 100, i, f'${row["mean_salary"]:,.0f}', 
                va='center', fontsize=9)
        ax1.text(row['mean_salary'] + 100, i + 0.2, f'n={row["count"]:,}', 
                va='center', fontsize=8, alpha=0.7)
    
    # Plot 2: Industry salary by experience level (heatmap)
    ax2 = axes[1]
    
    # Get top 10 industries by count
    top_10_industries = df['Industry_Category'].value_counts().head(10).index
    
    # Create pivot table
    pivot_data = df[df['Industry_Category'].isin(top_10_industries)]
    pivot_table = pivot_data.pivot_table(
        values='salary_average',
        index='Industry_Category',
        columns='experience_level',
        aggfunc='mean'
    )
    
    # Reorder columns by experience
    exp_order = [
        '0-5 years', '6-10 years', '11-15 years', '16-20 years',
        '21-25 years', '26-30 years', '31-35 years', '36-40 years', '40+ years'
    ]
    pivot_table = pivot_table.reindex(columns=[col for col in exp_order if col in pivot_table.columns])
    
    # Create heatmap
    im = ax2.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
    ax2.set_title('Salary Heatmap: Industry vs Experience Level')
    ax2.set_xlabel('Experience Level')
    ax2.set_ylabel('Industry')
    
    # Set ticks
    ax2.set_xticks(range(len(pivot_table.columns)))
    ax2.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(pivot_table.index)))
    ax2.set_yticklabels(pivot_table.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Average Salary (SGD)')
    
    plt.tight_layout()
    plt.savefig('industry_salary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print industry insights
    print("\n=== INDUSTRY INSIGHTS ===")
    print(f"Highest paying industry: {industry_salaries.index[0]} (${industry_salaries.iloc[0]['mean_salary']:,.2f})")
    print(f"Lowest paying industry: {industry_salaries.index[-1]} (${industry_salaries.iloc[-1]['mean_salary']:,.2f})")
    print(f"Industry count: {len(industry_salaries)}")
    print(f"Industries with >1000 jobs: {(industry_salaries['count'] > 1000).sum()}")


def save_datasets(filtered_df: pd.DataFrame) -> None:

    Save the filtered job dataset to a CSV file.
    
    Args:
        filtered_df: The processed and filtered dataframe
    """
    filename = "filtered_sgjobdata.csv"
    
    print("\n=== SAVING DATASET ===")
    
    # Save the dataframe
    filtered_df.to_csv(filename, index=False)
    
    print(f"Successfully saved {filename} ({len(filtered_df):,} rows)")


def main():
    """Main function to run the complete analysis pipeline."""
  
    
    # Database path
    db_path = "/Users/simgsr/Documents/GitHub/NTU_M1_Assignment--main/data/SGJobData_Normalized.db"
    
    try:
        # Step 1: Load data
        jobs_df, categories_df, jobcategories_df = load_data(db_path)
        
        # Step 2: Clean and prepare data
        cleaned_jobs = clean_and_prepare_data(jobs_df)
        
        # Step 3: Create experience levels
        cleaned_jobs = create_experience_levels(cleaned_jobs)
        
        # Step 4: Create industry categories
        categories_df = create_industry_categories(categories_df)
        
        # Step 5: Join all datasets
        merged_df = join_datasets(cleaned_jobs, categories_df, jobcategories_df)
        
        # Step 6: Analyze original data (before filtering)
        print("\n" + "=" * 80)
        print("ANALYZING ORIGINAL DATA (BEFORE FILTERING)")
        print("=" * 80)
        original_summary = analyze_salary_distribution(merged_df, "ORIGINAL DATA")
        
        # Step 7: Apply realistic salary filters
        realistically_filtered = apply_realistic_salary_filters(merged_df)
        filtered_summary = analyze_salary_distribution(realistically_filtered, "REALISTICALLY FILTERED")
        
        # Step 8: Apply IQR outlier filter
        iqr_filtered = apply_iqr_outlier_filter(realistically_filtered)
        iqr_summary = analyze_salary_distribution(iqr_filtered, "IQR FILTERED")
        
        # Step 9: Create visualizations
        create_visualizations(merged_df, realistically_filtered, iqr_filtered)
        
        # Step 10: Industry analysis (using IQR filtered data)
        industry_salary_analysis(iqr_filtered)
        
        # Step 11: Save datasets
        save_datasets(merged_df, realistically_filtered, iqr_filtered)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nRecommendations:")
        print("1. Use 'iqr_cleaned.csv' for statistical modeling (most conservative)")
        print("2. Use 'realistically_filtered.csv' for general trend analysis")
        print("3. Use 'original_with_outliers.csv' for reference only")
        print("\nVisualizations saved as PNG files.")
        
    except FileNotFoundError:
        print(f"Error: Database file not found at {db_path}")
        print("Please update the db_path variable with the correct path.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
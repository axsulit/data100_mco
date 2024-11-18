import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import os

class HousePrices:
    def __init__(self, data):
        """
        Class constructor for HousePrices.

        Parameters:
            self: This HousePrices class instantiation.
            data: The dataset containing house price information.
        """
        
        self.data = data.copy()
        self.transaction_columns = ['date', 'price']
        self.house_attributes = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
            'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
        ]
        self.identifier_column = 'id'


    
    @classmethod
    def identify_outliers(cls, df, threshold):
        """
        Identifies outliers using the Z-score method for specified columns.

        Parameters:
        cls: Accessor for this class method.
        df: The dataframe containing house price information.
        threshold: The z-score threshold for what will be considered an outlier.
        
        Returns:
        A dictionary with column names as keys and outlier indices as values.
        """
        
        SELECTED_COLUMNS = [
            'price', 'sqft_living', 'sqft_lot', 'sqft_above',
            'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
        ]
        outliers_dict = {}

        # Calculate z-scores
        for col in SELECTED_COLUMNS:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers_idx = np.where(z_scores > threshold)[0]
            outliers_dict[col] = outliers_idx
        return outliers_dict


    
    @classmethod
    def create_zscore_boxplot(cls, df, z_threshold):
        """
        Creates boxplots for the specified columns, using outlier data.

        Parameters:
        cls: Accessor for this class method.
        df: The dataframe containing house price information.
        threshold: The z-score threshold for what will be considered an outlier.
        
        Returns:
        The consolidated boxplot figure and a dictionary with column names as keys 
        and outlier indices as values.
        """
        
        SELECTED_COLUMNS = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 
                            'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
        
        outliers_dict = cls.identify_outliers(df, z_threshold)
        
        # Create subplots
        n_cols = 3
        n_rows = (len(SELECTED_COLUMNS) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle(f'Boxplots with Z-Score Outliers (Z > {z_threshold})', fontsize=16, y=1.02)
        axes_flat = axes.flatten() if n_rows > 1 else [axes]
        
        # Create boxplots
        for i, col in enumerate(SELECTED_COLUMNS):
            ax = axes_flat[i]
            data = df[col].dropna()
            bp = ax.boxplot(data, notch=True)
            outlier_indices = outliers_dict[col]
            
            if len(outlier_indices) > 0:
                outlier_values = data.iloc[outlier_indices]
                ax.scatter([1] * len(outlier_values), outlier_values, color='red', 
                           alpha=0.5, label='Z-score outliers')
            
            n_outliers = len(outliers_dict[col])
            ax.set_title(f'{col}\n(Z-score outliers: {n_outliers})')
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax.yaxis.grid(True, linestyle='-', alpha=0.2)
            if col == 'price':
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            if n_outliers > 0:
                ax.legend(fontsize='small')
        
        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])
        plt.tight_layout()

        print("\nZ-Score Outlier Summary:")
        print("-" * 40)
        for col in SELECTED_COLUMNS:
            n_outliers = len(outliers_dict[col])
            if n_outliers > 0:
                percent_outliers = (n_outliers / len(df[col].dropna())) * 100
                print(f"{col}: {n_outliers} outliers ({percent_outliers:.2f}% of values)")
        
        return fig, outliers_dict

    
    
    def print_outlier_details(df, outliers_dict):
        """
        Print detailed information about outliers for a specific column.

        Parameters:
        df: The dataframe containing house price information.
        outliers_dict: The dictionary containing the outlier information.
        """
        
        print("\n" + "="*80)
        print("OUTLIER ANALYSIS SUMMARY".center(80))
        print("="*80 )
        
        for col in outliers_dict.keys():
            outlier_indices = outliers_dict[col]
            n_outliers = len(outlier_indices)
            
            if n_outliers > 0:
                percent_outliers = (n_outliers / len(df[col].dropna())) * 100
                
                # Print statistics
                print("\n" + "-"*80)
                print(f" {col.upper()} ".center(80, "-"))
                print("-"*80)
                
                print(f"\nOutlier Count: {n_outliers:,} ({percent_outliers:.2f}% of values)")
                outlier_values = df[col].iloc[outlier_indices]
                stats = outlier_values.describe()
                
                print(f"{'Mean:':<15} {stats['mean']:,.2f}")
                print(f"{'Median:':<15} {stats['50%']:,.2f}")
                print(f"{'Std Dev:':<15} {stats['std']:,.2f}")
                
                extreme_values = pd.concat([outlier_values.nsmallest(5), 
                                            outlier_values.nlargest(5)]).sort_values()
                
                print("\nSmallest outliers:")
                for val in extreme_values.head(5):
                    print(f"  • {val:,.2f}")
                    
                print("\nLargest outliers:")
                for val in extreme_values.tail(5):
                    print(f"  • {val:,.2f}")

    

    def check_special_variables(df):
        """
        Print detailed information about each of the specified special variables.

        Parameters:
        df: The dataframe containing house price information.
        """
        
        print("\n=== Year Built Analysis ===")
        print(f"Unique values: {sorted(df['yr_built'].unique())}")
        print("\nValue counts:")
        print(df['yr_built'].value_counts().head())
        print(f"\nTotal unique years: {df['yr_built'].nunique()}")
        
        print("\n=== Year Renovated Analysis ===")
        print(f"Unique values: {sorted(df['yr_renovated'].unique())}")
        print("\nValue counts:")
        print(df['yr_renovated'].value_counts().head())
        print(f"\nTotal unique years: {df['yr_renovated'].nunique()}")
        
        print("\n=== Zipcode Analysis ===")
        print(f"Unique values: {sorted(df['zipcode'].unique())}")
        print("\nValue counts:")
        print(df['zipcode'].value_counts().head())
        print(f"\nTotal unique zipcodes: {df['zipcode'].nunique()}")
        
        print("\n=== Date Analysis ===")
        df['date_temp'] = pd.to_datetime(df['date'])
        print(f"Date range: {df['date_temp'].min()} - {df['date_temp'].max()}")
        print(f"Invalid dates (future): {
            df[df['date_temp'] > datetime.datetime.now()].shape[0]}")
        df.drop('date_temp', axis=1, inplace=True)
        
        # Additional logical checks
        print("\n=== Logical Checks ===")
        print(f"Renovation year before build year (excluding non-renovated): {
            df[(df['yr_renovated'] != 0) & (df['yr_renovated'] < df['yr_built'])].shape[0]}")



    def save_dataframe(df, filepath, index=False):
        """
        Save DataFrame to CSV.
        
        Parameters:
        df: The dataframe containing house price information.
        filepath: Path to save the CSV file.
        index: Whether to save the index as a column, default False.
        
        Returns:
        Whether the save was successful, True or False.
        """
        
        try:
            df.to_csv(filepath, index=index)
            return True
            
        except PermissionError:
            print("Permission denied. Try saving to the current directory instead.")
            try:
                # Try saving to current directory with just the filename
                new_filepath = os.path.basename(filepath)
                df.to_csv(new_filepath, index=index)
                print(f"Successfully saved DataFrame to {os.path.abspath(new_filepath)}")
                return True
            except Exception as e:
                print(f"Error saving to current directory: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error saving DataFrame: {str(e)}")
            return False





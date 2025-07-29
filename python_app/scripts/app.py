import pandas as pd
import numpy as np
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import urllib
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION SECTION - Extract all cleaning rules into structured format
# ==============================================================================

class CleaningConfig:
    """
    Centralized configuration for data cleaning operations.
    This replaces hardcoded cleaning logic with reusable rules.
    """
    
    def __init__(self):
        # Define cleaning rules for each dataset
        self.datasets = {
            'books': {
                'file_path': os.path.join("data", "03_Library Systembook.csv"),
                'output_table': 'SystemBook',
                'columns': {
                    'Id': {
                        'dtype': 'int',           # Convert to integer
                        'required': True,         # Cannot be null/invalid
                        'on_invalid': 'drop_row'  # Drop entire row if invalid
                    },
                    'Customer ID': {
                        'dtype': 'int',
                        'required': True,
                        'on_invalid': 'drop_row'
                    },
                    'Book checkout': {
                        'dtype': 'datetime',
                        'required': True,
                        'clean_quotes': True,     # Remove quote characters
                        'date_format': 'dayfirst', # Parse with day first
                        'on_invalid': 'drop_row'
                    },
                    'Book Returned': {
                        'dtype': 'datetime', 
                        'required': True,
                        'clean_quotes': True,
                        'date_format': 'dayfirst',
                        'on_invalid': 'drop_row'
                    }
                },
                # Special business rules
                'custom_rules': [
                    'calculate_loan_duration',  # Add LoanDurationDays column
                    'validate_return_after_checkout'  # Ensure return >= checkout
                ]
            },
            
            'customers': {
                'file_path': os.path.join("data", "03_Library SystemCustomers.csv"),
                'output_table': 'SystemCustomer',
                'columns': {
                    'Customer ID': {
                        'dtype': 'int',
                        'required': True,
                        'on_invalid': 'drop_row'
                    }
                    # Add other customer columns as needed
                },
                'custom_rules': []
            }
        }
        
        # Global cleaning settings
        self.global_settings = {
            'encoding_fallback': ['utf-8', 'latin1'],  # Try these encodings in order
            'empty_string_to_nan': True,               # Convert empty strings to NaN
            'drop_fully_empty_rows': True,             # Remove completely empty rows
            'drop_duplicates': True,                   # Remove duplicate rows
            'create_dated_folders': True,              # Organize output by date
        }
        
        # Database connection settings (loaded from .env)
        self.db_config = {
            'server_env': 'DB_SERVER',
            'database_env': 'DB_NAME', 
            'driver_env': 'DB_DRIVER',
            'trusted_connection_env': 'DB_TRUSTED_CONNECTION'
        }
        
        # Logging configuration
        self.logging_config = {
            'log_file': os.path.join("data", "cleaned", "cleaning_log.txt"),
            'sql_log_table': 'CleaningLog',
            'track_metrics': [
                'rows_initial', 'rows_after_empty_drop', 'rows_after_dupes',
                'rows_after_invalid_ids', 'quotes_removed', 'invalid_dates_fixed'
            ]
        }

    def get_dataset_config(self, dataset_name):
        """Get configuration for a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        return self.datasets[dataset_name]
    
    def get_column_config(self, dataset_name, column_name):
        """Get configuration for a specific column in a dataset"""
        dataset = self.get_dataset_config(dataset_name)
        if column_name not in dataset['columns']:
            return None  # Column not configured, use defaults
        return dataset['columns'][column_name]

# ==============================================================================
# BASE CLEANING RULE SYSTEM - Turn each operation into reusable classes
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class CleaningRule(ABC):
    """
    Abstract base class for all cleaning operations.
    Each rule is responsible for one specific cleaning task.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}  # Track what this rule changed
    
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning rule to the dataframe"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics about what this rule changed"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Clear metrics for next run"""
        self.metrics = {}

class EmptyStringToNaNRule(CleaningRule):
    """Convert empty strings and whitespace-only strings to NaN"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_nulls = df.isnull().sum().sum()
        df_cleaned = df.replace(r'^\s*

# Initialize configuration and cleaning pipeline
config = CleaningConfig()
pipeline = CleaningPipeline(config)

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, "..", ".env")
dotenv_path = os.path.abspath(dotenv_path)

load_dotenv(dotenv_path)

# Load DB credentials from .env using config
server = os.getenv(config.db_config['server_env'])
database = os.getenv(config.db_config['database_env'])
driver = os.getenv(config.db_config['driver_env'])
trusted_connection = os.getenv(config.db_config['trusted_connection_env'], "yes")

# Get absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build file paths using configuration
books_config = config.get_dataset_config('books')
customers_config = config.get_dataset_config('customers')

books_file = os.path.join(script_dir, books_config['file_path'])
customers_file = os.path.join(script_dir, customers_config['file_path'])

# Load with fallback encoding using config
encoding_options = config.global_settings['encoding_fallback']

def load_csv_with_fallback(file_path, encodings):
    """Load CSV with multiple encoding attempts - now configurable"""
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"{encoding} failed for {file_path}, trying next encoding...")
            continue
    raise Exception(f"All encoding options failed for {file_path}")

try:
    books_df_raw = load_csv_with_fallback(books_file, encoding_options)
    customers_df_raw = load_csv_with_fallback(customers_file, encoding_options)
except Exception as e:
    print(f"🚨 Error loading CSVs: {e}")

# Store initial row counts for logging
books_rows_initial = len(books_df_raw)
customers_rows_initial = len(customers_df_raw)

# Apply cleaning pipeline to each dataset
print("[INFO] Applying cleaning pipeline to books dataset...")
books_df = pipeline.clean_dataset(books_df_raw, 'books')

print("[INFO] Applying cleaning pipeline to customers dataset...")
customers_df = pipeline.clean_dataset(customers_df_raw, 'customers')

# Get detailed metrics from pipeline
all_metrics = pipeline.get_all_metrics()

# Extract key metrics for backwards compatibility with logging
def extract_legacy_metrics(metrics_dict, dataset_name):
    """Extract metrics in the format expected by the existing logging system"""
    dataset_metrics = metrics_dict.get(dataset_name, {})
    
    # Extract metrics from individual rules
    empty_rows_dropped = dataset_metrics.get('DropEmptyRowsRule', {}).get('empty_rows_dropped', 0)
    duplicate_rows_dropped = dataset_metrics.get('DropDuplicatesRule', {}).get('duplicate_rows_dropped', 0)
    invalid_ids_dropped = dataset_metrics.get('DropInvalidRequiredRule', {}).get('rows_dropped', 0)
    quotes_removed = dataset_metrics.get('CleanQuotesRule', {}).get('quotes_removed', 0)
    
    # Calculate final row counts (work backwards from drops)
    initial_rows = books_rows_initial if dataset_name == 'books' else customers_rows_initial
    after_empty_drop = initial_rows - empty_rows_dropped
    after_dupes_drop = after_empty_drop - duplicate_rows_dropped
    after_invalid_drop = after_dupes_drop - invalid_ids_dropped
    
    return {
        'rows_initial': initial_rows,
        'rows_after_empty_drop': after_empty_drop,
        'rows_after_dupes_drop': after_dupes_drop, 
        'rows_after_invalid_drop': after_invalid_drop,
        'dropped_invalid_ids': invalid_ids_dropped,
        'quotes_removed': quotes_removed
    }

books_legacy_metrics = extract_legacy_metrics(all_metrics, 'books')
customers_legacy_metrics = extract_legacy_metrics(all_metrics, 'customers')

# Legacy variables for logging compatibility
books_rows_after_dropna = books_legacy_metrics['rows_after_empty_drop']
books_rows_after_dupes = books_legacy_metrics['rows_after_dupes_drop']
books_rows_after_dropna_ids = books_legacy_metrics['rows_after_invalid_drop']
books_dropped_invalid_ids = books_legacy_metrics['dropped_invalid_ids']

customers_rows_after_dropna = customers_legacy_metrics['rows_after_empty_drop']
customers_rows_after_dupes = customers_legacy_metrics['rows_after_dupes_drop']
customers_rows_after_dropna_ids = customers_legacy_metrics['rows_after_invalid_drop']
customers_dropped_invalid_ids = customers_legacy_metrics['dropped_invalid_ids']

quotes_removed = (books_legacy_metrics['quotes_removed'] + 
                 customers_legacy_metrics['quotes_removed'])

# Calculate invalid dates fixed from conversion metrics
books_conversion_metrics = all_metrics.get('books', {}).get('DataTypeConversionRule', {})
customers_conversion_metrics = all_metrics.get('customers', {}).get('DataTypeConversionRule', {})

books_invalid_dates = sum(books_conversion_metrics.get('invalid_values_found', {}).values())
customers_invalid_dates = sum(customers_conversion_metrics.get('invalid_values_found', {}).values())
invalid_dates_fixed = books_invalid_dates + customers_invalid_dates

print(f"[INFO] Books cleaning complete: {books_rows_initial} → {len(books_df)} rows")
print(f"[INFO] Customers cleaning complete: {customers_rows_initial} → {len(customers_df)} rows")

# Build connection string (unchanged)
params = urllib.parse.quote_plus(
    f"DRIVER={driver};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"Trusted_Connection={trusted_connection};"
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Write to database using configured table names
books_df.to_sql(books_config['output_table'], con=engine, if_exists='replace', index=False)
customers_df.to_sql(customers_config['output_table'], con=engine, if_exists='replace', index=False)

print(f"[INFO] Data written to SQL Server tables {books_config['output_table']} and {customers_config['output_table']}")

# Create cleaned folder using configuration
if config.global_settings['create_dated_folders']:
    base_cleaned_dir = os.path.join(script_dir, "data", "cleaned")
    today_str = datetime.today().strftime("%Y%m%d")
    dated_folder = os.path.join(base_cleaned_dir, today_str)
    os.makedirs(dated_folder, exist_ok=True)
    
    # Save cleaned files with configured names
    clean_books_file = os.path.join(dated_folder, "03_Library Systembook_cleaned.csv")
    clean_customers_file = os.path.join(dated_folder, "03_Library SystemCustomers_cleaned.csv")
    
    books_df.to_csv(clean_books_file, index=False)
    customers_df.to_csv(clean_customers_file, index=False)
    
    print(f"[INFO] Cleaned books data saved to {clean_books_file}")
    print(f"[INFO] Cleaned customers data saved to {clean_customers_file}")

# Logging using configuration
log_file = config.logging_config['log_file']
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "a") as log:
    log_entry = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Books rows: initial={books_rows_initial}, after empty drop={books_rows_after_dropna}, "
        f"after duplicates drop={books_rows_after_dupes}, after invalid ID drop={books_rows_after_dropna_ids}, dropped_invalid_ids={books_dropped_invalid_ids} | "
        f"Customers rows: initial={customers_rows_initial}, after empty drop={customers_rows_after_dropna}, "
        f"after duplicates drop={customers_rows_after_dupes}, after invalid ID drop={customers_rows_after_dropna_ids}, dropped_invalid_ids={customers_dropped_invalid_ids} | "
        f"Quotes removed from dates: {quotes_removed} | Invalid dates fixed: {invalid_dates_fixed}\n"
    )
    log.write(log_entry)

print(f"[INFO] Cleaning log .txt updated at {log_file}")

# SQL logging using configuration
log_df = pd.DataFrame([{
    "Timestamp": datetime.now(),
    "BooksRowsInitial": books_rows_initial,
    "BooksRowsAfterEmptyDrop": books_rows_after_dropna,
    "BooksRowsAfterDupesDrop": books_rows_after_dupes,
    "BooksRowsAfterInvalidIDDrop": books_rows_after_dropna_ids,
    "BooksDroppedInvalidIDs": books_dropped_invalid_ids,
    "CustomersRowsInitial": customers_rows_initial,
    "CustomersRowsAfterEmptyDrop": customers_rows_after_dropna,
    "CustomersRowsAfterDupesDrop": customers_rows_after_dupes,
    "CustomersRowsAfterInvalidIDDrop": customers_rows_after_dropna_ids,
    "CustomersDroppedInvalidIDs": customers_dropped_invalid_ids,
    "QuotesRemoved": quotes_removed,
    "InvalidDatesFixed": invalid_dates_fixed
}])

# Create CleaningLog table using configured name
log_table_name = config.logging_config['sql_log_table']
with engine.connect() as conn:
    conn.execute(text(f"""
        IF NOT EXISTS (
            SELECT * FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{log_table_name}'
        )
        CREATE TABLE {log_table_name} (
            LogID INT IDENTITY(1,1) PRIMARY KEY,
            Timestamp DATETIME,
            BooksRowsInitial INT,
            BooksRowsAfterEmptyDrop INT,
            BooksRowsAfterDupesDrop INT,
            BooksRowsAfterInvalidIDDrop INT,
            BooksDroppedInvalidIDs INT,
            CustomersRowsInitial INT,
            CustomersRowsAfterEmptyDrop INT,
            CustomersRowsAfterDupesDrop INT,
            CustomersRowsAfterInvalidIDDrop INT,
            CustomersDroppedInvalidIDs INT,
            QuotesRemoved INT,
            InvalidDatesFixed INT
        )
    """))

log_df.to_sql(log_table_name, con=engine, if_exists='append', index=False), np.nan, regex=True)
        final_nulls = df_cleaned.isnull().sum().sum()
        
        self.metrics['empty_strings_converted'] = final_nulls - initial_nulls
        return df_cleaned

class DropEmptyRowsRule(CleaningRule):
    """Drop rows that are completely empty (all NaN)"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        df_cleaned = df.dropna(how='all')
        final_rows = len(df_cleaned)
        
        self.metrics['empty_rows_dropped'] = initial_rows - final_rows
        return df_cleaned

class DropDuplicatesRule(CleaningRule):
    """Remove duplicate rows"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        final_rows = len(df_cleaned)
        
        self.metrics['duplicate_rows_dropped'] = initial_rows - final_rows
        return df_cleaned

class CleanQuotesRule(CleaningRule):
    """Remove quote characters from specified columns"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        total_quotes_removed = 0
        
        # Get columns that need quote cleaning from config
        columns_to_clean = []
        if 'columns' in self.config:
            for col_name, col_config in self.config['columns'].items():
                if col_config.get('clean_quotes', False) and col_name in df_cleaned.columns:
                    columns_to_clean.append(col_name)
        
        for col in columns_to_clean:
            quotes_before = df_cleaned[col].astype(str).str.count('"').sum()
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace('"', '')
            quotes_after = df_cleaned[col].astype(str).str.count('"').sum()
            total_quotes_removed += (quotes_before - quotes_after)
        
        self.metrics['quotes_removed'] = total_quotes_removed
        self.metrics['columns_cleaned'] = columns_to_clean
        return df_cleaned

class DataTypeConversionRule(CleaningRule):
    """Convert columns to specified data types based on configuration"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        conversions_applied = {}
        invalid_values_found = {}
        
        if 'columns' not in self.config:
            return df_cleaned
        
        for col_name, col_config in self.config['columns'].items():
            if col_name not in df_cleaned.columns:
                continue
                
            dtype = col_config.get('dtype')
            if not dtype:
                continue
            
            initial_valid = df_cleaned[col_name].notna().sum()
            
            if dtype == 'int':
                df_cleaned[col_name] = pd.to_numeric(df_cleaned[col_name], errors='coerce')
            elif dtype == 'datetime':
                date_format = col_config.get('date_format', 'infer')
                dayfirst = (date_format == 'dayfirst')
                df_cleaned[col_name] = pd.to_datetime(df_cleaned[col_name], 
                                                   dayfirst=dayfirst, 
                                                   errors='coerce')
            
            final_valid = df_cleaned[col_name].notna().sum()
            invalid_count = initial_valid - final_valid
            
            conversions_applied[col_name] = dtype
            if invalid_count > 0:
                invalid_values_found[col_name] = invalid_count
        
        self.metrics['conversions_applied'] = conversions_applied
        self.metrics['invalid_values_found'] = invalid_values_found
        return df_cleaned

class DropInvalidRequiredRule(CleaningRule):
    """Drop rows where required columns have invalid/null values"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'columns' not in self.config:
            return df
            
        required_columns = []
        for col_name, col_config in self.config['columns'].items():
            if (col_config.get('required', False) and 
                col_config.get('on_invalid') == 'drop_row' and 
                col_name in df.columns):
                required_columns.append(col_name)
        
        if not required_columns:
            self.metrics['required_columns'] = []
            self.metrics['rows_dropped'] = 0
            return df
        
        initial_rows = len(df)
        df_cleaned = df.dropna(subset=required_columns)
        final_rows = len(df_cleaned)
        
        self.metrics['required_columns'] = required_columns
        self.metrics['rows_dropped'] = initial_rows - final_rows
        return df_cleaned

class FinalDataTypeRule(CleaningRule):
    """Convert columns to final data types after cleaning (no more coercion)"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        final_conversions = {}
        
        if 'columns' not in self.config:
            return df_cleaned
        
        for col_name, col_config in self.config['columns'].items():
            if col_name not in df_cleaned.columns:
                continue
                
            dtype = col_config.get('dtype')
            if dtype == 'int':
                # Only convert if all values are valid (no NaN after cleaning)
                if df_cleaned[col_name].notna().all():
                    df_cleaned[col_name] = df_cleaned[col_name].astype(int)
                    final_conversions[col_name] = 'int'
        
        self.metrics['final_conversions'] = final_conversions
        return df_cleaned

class CustomBusinessRulesRule(CleaningRule):
    """Apply dataset-specific custom business logic"""
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        rules_applied = []
        
        custom_rules = self.config.get('custom_rules', [])
        
        for rule in custom_rules:
            if rule == 'calculate_loan_duration':
                if 'Book Returned' in df_cleaned.columns and 'Book checkout' in df_cleaned.columns:
                    df_cleaned["LoanDurationDays"] = (df_cleaned["Book Returned"] - 
                                                    df_cleaned["Book checkout"]).dt.days
                    rules_applied.append('calculate_loan_duration')
                    
            elif rule == 'validate_return_after_checkout':
                if 'LoanDurationDays' in df_cleaned.columns:
                    initial_rows = len(df_cleaned)
                    df_cleaned = df_cleaned[df_cleaned["LoanDurationDays"] >= 0]
                    final_rows = len(df_cleaned)
                    rules_applied.append('validate_return_after_checkout')
                    self.metrics['invalid_loan_periods_removed'] = initial_rows - final_rows
        
        self.metrics['rules_applied'] = rules_applied
        return df_cleaned

class CleaningPipeline:
    """
    Orchestrates the application of cleaning rules in the correct order.
    This replaces the scattered cleaning logic in the original script.
    """
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.global_rules = self._build_global_rules()
        self.all_metrics = {}
    
    def _build_global_rules(self) -> list:
        """Build the standard cleaning rules that apply to all datasets"""
        rules = []
        
        if self.config.global_settings.get('empty_string_to_nan', False):
            rules.append(EmptyStringToNaNRule({}))
            
        if self.config.global_settings.get('drop_fully_empty_rows', False):
            rules.append(DropEmptyRowsRule({}))
            
        if self.config.global_settings.get('drop_duplicates', False):
            rules.append(DropDuplicatesRule({}))
            
        return rules
    
    def clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Apply all cleaning rules to a specific dataset"""
        dataset_config = self.config.get_dataset_config(dataset_name)
        df_cleaned = df.copy()
        self.all_metrics[dataset_name] = {}
        
        # Apply global rules first
        for rule in self.global_rules:
            rule.reset_metrics()
            df_cleaned = rule.apply(df_cleaned)
            rule_name = rule.__class__.__name__
            self.all_metrics[dataset_name][rule_name] = rule.get_metrics()
        
        # Apply dataset-specific rules
        dataset_rules = [
            CleanQuotesRule(dataset_config),
            DataTypeConversionRule(dataset_config), 
            DropInvalidRequiredRule(dataset_config),
            FinalDataTypeRule(dataset_config),
            CustomBusinessRulesRule(dataset_config)
        ]
        
        for rule in dataset_rules:
            rule.reset_metrics()
            df_cleaned = rule.apply(df_cleaned)
            rule_name = rule.__class__.__name__
            self.all_metrics[dataset_name][rule_name] = rule.get_metrics()
        
        return df_cleaned
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive metrics from all applied rules"""
        return self.all_metrics.copy()

# ==============================================================================
# ORIGINAL SCRIPT WITH CONFIGURATION INTEGRATION
# ==============================================================================

# Initialize configuration
config = CleaningConfig()

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, "..", ".env")
dotenv_path = os.path.abspath(dotenv_path)

load_dotenv(dotenv_path)

# Load DB credentials from .env using config
server = os.getenv(config.db_config['server_env'])
database = os.getenv(config.db_config['database_env'])
driver = os.getenv(config.db_config['driver_env'])
trusted_connection = os.getenv(config.db_config['trusted_connection_env'], "yes")

# Get absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build file paths using configuration
books_config = config.get_dataset_config('books')
customers_config = config.get_dataset_config('customers')

books_file = os.path.join(script_dir, books_config['file_path'])
customers_file = os.path.join(script_dir, customers_config['file_path'])

# Load with fallback encoding using config
encoding_options = config.global_settings['encoding_fallback']

def load_csv_with_fallback(file_path, encodings):
    """Load CSV with multiple encoding attempts - now configurable"""
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"{encoding} failed for {file_path}, trying next encoding...")
            continue
    raise Exception(f"All encoding options failed for {file_path}")

try:
    books_df = load_csv_with_fallback(books_file, encoding_options)
    customers_df = load_csv_with_fallback(customers_file, encoding_options)
except Exception as e:
    print(f"🚨 Error loading CSVs: {e}")

# Initial row counts
books_rows_initial = len(books_df)
customers_rows_initial = len(customers_df)

# Apply global cleaning settings
if config.global_settings['empty_string_to_nan']:
    # Replace empty strings/whitespace with NaN
    books_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    customers_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

if config.global_settings['drop_fully_empty_rows']:
    # Drop fully empty rows
    books_df.dropna(how='all', inplace=True)
    customers_df.dropna(how='all', inplace=True)

books_rows_after_dropna = len(books_df)
customers_rows_after_dropna = len(customers_df)

if config.global_settings['drop_duplicates']:
    # Drop duplicates
    books_df.drop_duplicates(inplace=True)
    customers_df.drop_duplicates(inplace=True)

books_rows_after_dupes = len(books_df)
customers_rows_after_dupes = len(customers_df)

# Apply column-specific cleaning rules using configuration
def apply_column_cleaning(df, dataset_name):
    """Apply column-specific cleaning rules based on configuration"""
    dataset_config = config.get_dataset_config(dataset_name)
    quotes_removed = 0
    invalid_dates_fixed = 0
    
    for col_name, col_config in dataset_config['columns'].items():
        if col_name not in df.columns:
            continue
            
        # Clean quotes if specified
        if col_config.get('clean_quotes', False):
            quotes_before = df[col_name].astype(str).str.count('"').sum()
            df[col_name] = df[col_name].astype(str).str.replace('"', '')
            quotes_after = df[col_name].astype(str).str.count('"').sum()
            quotes_removed += (quotes_before - quotes_after)
        
        # Handle data type conversion
        if col_config['dtype'] == 'int':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        elif col_config['dtype'] == 'datetime':
            parsed_dates = pd.to_datetime(df[col_name], 
                                        dayfirst=(col_config.get('date_format') == 'dayfirst'),
                                        errors='coerce')
            invalid_dates = parsed_dates.isna().sum()
            invalid_dates_fixed += invalid_dates
            df[col_name] = parsed_dates
    
    return quotes_removed, invalid_dates_fixed

# Apply cleaning to both datasets
books_quotes_removed, books_invalid_dates = apply_column_cleaning(books_df, 'books')
customers_quotes_removed, customers_invalid_dates = apply_column_cleaning(customers_df, 'customers')

quotes_removed = books_quotes_removed + customers_quotes_removed
invalid_dates_fixed = books_invalid_dates + customers_invalid_dates

# Handle required columns and invalid data
def handle_required_columns(df, dataset_name):
    """Drop rows with invalid data in required columns"""
    dataset_config = config.get_dataset_config(dataset_name)
    required_cols = []
    
    for col_name, col_config in dataset_config['columns'].items():
        if col_config.get('required', False) and col_config.get('on_invalid') == 'drop_row':
            required_cols.append(col_name)
    
    if required_cols:
        rows_before = len(df)
        df.dropna(subset=required_cols, inplace=True)
        return rows_before - len(df)
    return 0

books_rows_before_dropna_ids = len(books_df)
customers_rows_before_dropna_ids = len(customers_df)

books_dropped_invalid_ids = handle_required_columns(books_df, 'books')
customers_dropped_invalid_ids = handle_required_columns(customers_df, 'customers')

books_rows_after_dropna_ids = len(books_df)
customers_rows_after_dropna_ids = len(customers_df)

# Convert to final data types for required columns
for col_name, col_config in books_config['columns'].items():
    if col_name in books_df.columns and col_config['dtype'] == 'int':
        books_df[col_name] = books_df[col_name].astype(int)

for col_name, col_config in customers_config['columns'].items():
    if col_name in customers_df.columns and col_config['dtype'] == 'int':
        customers_df[col_name] = customers_df[col_name].astype(int)

# Apply custom business rules using configuration
def apply_custom_rules(df, dataset_name):
    """Apply dataset-specific custom business rules"""
    dataset_config = config.get_dataset_config(dataset_name)
    
    for rule in dataset_config.get('custom_rules', []):
        if rule == 'calculate_loan_duration':
            # Add loan duration column (in days)
            df["LoanDurationDays"] = (df["Book Returned"] - df["Book checkout"]).dt.days
        elif rule == 'validate_return_after_checkout':
            # Remove rows where return date is before checkout
            df = df[df["LoanDurationDays"] >= 0]
    
    return df

books_df = apply_custom_rules(books_df, 'books')
customers_df = apply_custom_rules(customers_df, 'customers')

# Build connection string (unchanged)
params = urllib.parse.quote_plus(
    f"DRIVER={driver};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"Trusted_Connection={trusted_connection};"
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Write to database using configured table names
books_df.to_sql(books_config['output_table'], con=engine, if_exists='replace', index=False)
customers_df.to_sql(customers_config['output_table'], con=engine, if_exists='replace', index=False)

print(f"[INFO] Data written to SQL Server tables {books_config['output_table']} and {customers_config['output_table']}")

# Create cleaned folder using configuration
if config.global_settings['create_dated_folders']:
    base_cleaned_dir = os.path.join(script_dir, "data", "cleaned")
    today_str = datetime.today().strftime("%Y%m%d")
    dated_folder = os.path.join(base_cleaned_dir, today_str)
    os.makedirs(dated_folder, exist_ok=True)
    
    # Save cleaned files with configured names
    clean_books_file = os.path.join(dated_folder, "03_Library Systembook_cleaned.csv")
    clean_customers_file = os.path.join(dated_folder, "03_Library SystemCustomers_cleaned.csv")
    
    books_df.to_csv(clean_books_file, index=False)
    customers_df.to_csv(clean_customers_file, index=False)
    
    print(f"[INFO] Cleaned books data saved to {clean_books_file}")
    print(f"[INFO] Cleaned customers data saved to {clean_customers_file}")

# Logging using configuration
log_file = config.logging_config['log_file']
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "a") as log:
    log_entry = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Books rows: initial={books_rows_initial}, after empty drop={books_rows_after_dropna}, "
        f"after duplicates drop={books_rows_after_dupes}, after invalid ID drop={books_rows_after_dropna_ids}, dropped_invalid_ids={books_dropped_invalid_ids} | "
        f"Customers rows: initial={customers_rows_initial}, after empty drop={customers_rows_after_dropna}, "
        f"after duplicates drop={customers_rows_after_dupes}, after invalid ID drop={customers_rows_after_dropna_ids}, dropped_invalid_ids={customers_dropped_invalid_ids} | "
        f"Quotes removed from dates: {quotes_removed} | Invalid dates fixed: {invalid_dates_fixed}\n"
    )
    log.write(log_entry)

print(f"[INFO] Cleaning log .txt updated at {log_file}")

# SQL logging using configuration
log_df = pd.DataFrame([{
    "Timestamp": datetime.now(),
    "BooksRowsInitial": books_rows_initial,
    "BooksRowsAfterEmptyDrop": books_rows_after_dropna,
    "BooksRowsAfterDupesDrop": books_rows_after_dupes,
    "BooksRowsAfterInvalidIDDrop": books_rows_after_dropna_ids,
    "BooksDroppedInvalidIDs": books_dropped_invalid_ids,
    "CustomersRowsInitial": customers_rows_initial,
    "CustomersRowsAfterEmptyDrop": customers_rows_after_dropna,
    "CustomersRowsAfterDupesDrop": customers_rows_after_dupes,
    "CustomersRowsAfterInvalidIDDrop": customers_rows_after_dropna_ids,
    "CustomersDroppedInvalidIDs": customers_dropped_invalid_ids,
    "QuotesRemoved": quotes_removed,
    "InvalidDatesFixed": invalid_dates_fixed
}])

# Create CleaningLog table using configured name
log_table_name = config.logging_config['sql_log_table']
with engine.connect() as conn:
    conn.execute(text(f"""
        IF NOT EXISTS (
            SELECT * FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{log_table_name}'
        )
        CREATE TABLE {log_table_name} (
            LogID INT IDENTITY(1,1) PRIMARY KEY,
            Timestamp DATETIME,
            BooksRowsInitial INT,
            BooksRowsAfterEmptyDrop INT,
            BooksRowsAfterDupesDrop INT,
            BooksRowsAfterInvalidIDDrop INT,
            BooksDroppedInvalidIDs INT,
            CustomersRowsInitial INT,
            CustomersRowsAfterEmptyDrop INT,
            CustomersRowsAfterDupesDrop INT,
            CustomersRowsAfterInvalidIDDrop INT,
            CustomersDroppedInvalidIDs INT,
            QuotesRemoved INT,
            InvalidDatesFixed INT
        )
    """))

log_df.to_sql(log_table_name, con=engine, if_exists='append', index=False)
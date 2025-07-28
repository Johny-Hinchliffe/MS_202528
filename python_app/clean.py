import pandas as pd
import numpy as np
import os
from datetime import datetime

books_file = "data/03_Library Systembook.csv"
customers_file = "data/03_Library SystemCustomers.csv"

# Load with fallback encoding
try:
    books_df = pd.read_csv(books_file, encoding='utf-8')
    customers_df = pd.read_csv(customers_file, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 failed, trying latin1 encoding...")
    books_df = pd.read_csv(books_file, encoding='latin1')
    customers_df = pd.read_csv(customers_file, encoding='latin1')
except Exception as e:
    print(f"🚨 Error loading CSVs: {e}")

# Initial row counts
books_rows_initial = len(books_df)
customers_rows_initial = len(customers_df)

# Replace empty strings/whitespace with NaN
books_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
customers_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Drop fully empty rows
books_df.dropna(how='all', inplace=True)
customers_df.dropna(how='all', inplace=True)
books_rows_after_dropna = len(books_df)
customers_rows_after_dropna = len(customers_df)

# Drop duplicates
books_df.drop_duplicates(inplace=True)
customers_df.drop_duplicates(inplace=True)
books_rows_after_dupes = len(books_df)
customers_rows_after_dupes = len(customers_df)

# Fix data types with conversion counts
books_df['Id'] = pd.to_numeric(books_df['Id'], errors='coerce')
books_df['Customer ID'] = pd.to_numeric(books_df['Customer ID'], errors='coerce')
customers_df['Customer ID'] = pd.to_numeric(customers_df['Customer ID'], errors='coerce')

books_rows_before_dropna_ids = len(books_df)
customers_rows_before_dropna_ids = len(customers_df)

books_df.dropna(subset=['Id', 'Customer ID'], inplace=True)
customers_df.dropna(subset=['Customer ID'], inplace=True)

books_rows_after_dropna_ids = len(books_df)
customers_rows_after_dropna_ids = len(customers_df)

books_dropped_invalid_ids = books_rows_before_dropna_ids - books_rows_after_dropna_ids
customers_dropped_invalid_ids = customers_rows_before_dropna_ids - customers_rows_after_dropna_ids

books_df['Id'] = books_df['Id'].astype(int)
books_df['Customer ID'] = books_df['Customer ID'].astype(int)
customers_df['Customer ID'] = customers_df['Customer ID'].astype(int)

# Clean date columns: remove quotes and unify format
date_cols = ["Book checkout", "Book Returned"]
quotes_removed = 0
invalid_dates_fixed = 0

for col in date_cols:
    if col in books_df.columns:
        # Count quotes before
        quotes_before = books_df[col].astype(str).str.count('"').sum()
        books_df[col] = books_df[col].astype(str).str.replace('"', '')
        quotes_after = books_df[col].astype(str).str.count('"').sum()
        quotes_removed += (quotes_before - quotes_after)

        # Convert to datetime to identify invalid/malformed dates
        parsed_dates = pd.to_datetime(books_df[col], dayfirst=True, errors='coerce')
        invalid_dates = parsed_dates.isna().sum()
        invalid_dates_fixed += invalid_dates
        books_df[col] = parsed_dates.dt.strftime('%d/%m/%Y')

# Create cleaned folder (single folder per day)
base_cleaned_dir = "../data/cleaned"
today_str = datetime.today().strftime("%Y%m%d")
dated_folder = os.path.join(base_cleaned_dir, today_str)
os.makedirs(dated_folder, exist_ok=True)

# Save cleaned files (overwrite if existing)
clean_books_file = os.path.join(dated_folder, "03_Library Systembook_cleaned.csv")
clean_customers_file = os.path.join(dated_folder, "03_Library SystemCustomers_cleaned.csv")

books_df.to_csv(clean_books_file, index=False)
customers_df.to_csv(clean_customers_file, index=False)

print(f"[INFO] Cleaned books data saved to {clean_books_file}")
print(f"[INFO] Cleaned customers data saved to {clean_customers_file}")

# --- DETAILED LOGGING ---
log_file = os.path.join(base_cleaned_dir, "cleaning_log.txt")
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

print(f"[INFO] Cleaning log updated at {log_file}")

import pandas as pd
import numpy as np

# Load CSV files with fallback encoding
books_file = "../data/03_Library Systembook.csv"
customers_file = "../data/03_Library SystemCustomers.csv"

try:
    books_df = pd.read_csv(books_file, encoding='utf-8')
    customers_df = pd.read_csv(customers_file, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 failed, trying latin1 encoding...")
    books_df = pd.read_csv(books_file, encoding='latin1')
    customers_df = pd.read_csv(customers_file, encoding='latin1')
except Exception as e:
    print(f"🚨 Error loading CSVs: {e}")

# Replace empty strings or whitespace-only strings with NaN
books_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
customers_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Count fully empty rows (all columns NaN)
empty_books_rows = books_df.isnull().all(axis=1).sum()
empty_customers_rows = customers_df.isnull().all(axis=1).sum()

print(f"[INFO] Fully blank or empty rows in Books: {empty_books_rows}")
print(f"[INFO] Fully blank or empty rows in Customers: {empty_customers_rows}")

# Drop fully empty rows before validations
books_df.dropna(how='all', inplace=True)
customers_df.dropna(how='all', inplace=True)

# 1. Check schema (column names)
def check_schema(df, expected_cols, name):
    if list(df.columns) != expected_cols:
        print(f"[ERROR] {name} file schema mismatch.")
        print(f"Expected: {expected_cols}")
        print(f"Found: {list(df.columns)}")
        return False
    return True

# 2. Check for missing values in important fields
def check_missing_values(df, required_cols, name):
    issues = df[required_cols].isnull().sum()
    if issues.any():
        print(f"[ERROR] Missing values found in {name}:")
        print(issues[issues > 0])
        return False
    return True

# 3. Check data types
def check_data_types(df, expected_types, name):
    valid = True
    for col, expected_type in expected_types.items():
        if not df[col].map(type).eq(expected_type).all():
            print(f"[ERROR] Column '{col}' in {name} has incorrect data types.")
            print(df[~df[col].map(type).eq(expected_type)])
            valid = False
    return valid

# 4. Foreign key check: Customer ID in books must exist in customers
def check_foreign_keys(books_df, customers_df):
    invalid_customers = books_df[~books_df["Customer ID"].isin(customers_df["Customer ID"])]
    if not invalid_customers.empty:
        print("[ERROR] Foreign key mismatch: Customer ID not found in customers file.")
        print(invalid_customers)
        return False
    return True

# Perform validations
schema_ok = check_schema(books_df, expected_books_cols, "Books") and \
            check_schema(customers_df, expected_customers_cols, "Customers")

missing_ok = check_missing_values(books_df, expected_books_cols, "Books") and \
             check_missing_values(customers_df, expected_customers_cols, "Customers")

types_ok = check_data_types(
    books_df,
    {
        "Id": int,
        "Books": str,
        "Book checkout": str,
        "Book Returned": str,
        "Days allowed to borrow": int,
        "Customer ID": int
    },
    "Books"
) and check_data_types(
    customers_df,
    {
        "Customer ID": int,
        "Customer Name": str
    },
    "Customers"
)

fk_ok = check_foreign_keys(books_df, customers_df)

# Final result
if schema_ok and missing_ok and types_ok and fk_ok:
    print("[SUCCESS] Data validation passed.")
else:
    print("[FAILED] Data validation failed. Please review the errors above.")

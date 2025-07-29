import pandas as pd
import numpy as np
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import urllib
from dotenv import load_dotenv


class EnvLoader:
    def __init__(self, base_path):
        self.dotenv_path = os.path.abspath(os.path.join(base_path, ".env"))
        load_dotenv(self.dotenv_path)


    def get_db_credentials(self):
        creds = {
            "server": os.getenv("DB_SERVER"),
            "database": os.getenv("DB_NAME"),
            "driver": os.getenv("DB_DRIVER"),
            "trusted_connection": os.getenv("DB_TRUSTED_CONNECTION", "yes")
        }

        # print("[DEBUG] Loaded DB credentials:")
        # for key, val in creds.items():
        #     print(f"  {key}: {val}")

        return creds



class FileManager:
    def __init__(self, script_dir):
        self.script_dir = script_dir

    def read_csvs(self):
        books_file = os.path.join(self.script_dir, "data", "03_Library Systembook.csv")
        customers_file = os.path.join(self.script_dir, "data", "03_Library SystemCustomers.csv")
        try:
            books_df = pd.read_csv(books_file, encoding='utf-8')
            customers_df = pd.read_csv(customers_file, encoding='utf-8')
        except UnicodeDecodeError:
            books_df = pd.read_csv(books_file, encoding='latin1')
            customers_df = pd.read_csv(customers_file, encoding='latin1')
        return books_df, customers_df

    def save_cleaned(self, books_df, customers_df):
        base_cleaned_dir = os.path.join(self.script_dir, "data", "cleaned")
        today_str = datetime.today().strftime("%Y%m%d")
        dated_folder = os.path.join(base_cleaned_dir, today_str)
        os.makedirs(dated_folder, exist_ok=True)

        clean_books_file = os.path.join(dated_folder, "03_Library Systembook_cleaned.csv")
        clean_customers_file = os.path.join(dated_folder, "03_Library SystemCustomers_cleaned.csv")

        books_df.to_csv(clean_books_file, index=False)
        customers_df.to_csv(clean_customers_file, index=False)

        return clean_books_file, clean_customers_file

    def write_log_txt(self, log_text):
        log_file = os.path.join(self.script_dir, "data", "cleaned", "cleaning_log.txt")
        with open(log_file, "a") as log:
            log.write(log_text)
        return log_file


class DataCleaner:
    def __init__(self, books_df, customers_df):
        self.books_df = books_df
        self.customers_df = customers_df
        self.quotes_removed = 0
        self.invalid_dates_fixed = 0
        self.rows_removed_invalid_dates = 0  # For NULL/invalid dates
        self.rows_removed_negative_loan = 0  # For checkout after return


    def clean(self):
        self._replace_blanks()
        self._drop_empty_rows()
        self._drop_duplicates()
        self._convert_types()
        self._clean_dates()
        self._calculate_loan_duration()
        self._remove_invalid_loans()
        return (
            self.books_df,
            self.customers_df,
            self.quotes_removed,
            self.invalid_dates_fixed,
            self.rows_removed_invalid_dates,
            self.rows_removed_negative_loan
        )


    def _replace_blanks(self):
        self.books_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.customers_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    def _drop_empty_rows(self):
        self.books_df.dropna(how='all', inplace=True)
        self.customers_df.dropna(how='all', inplace=True)

    def _drop_duplicates(self):
        self.books_df.drop_duplicates(inplace=True)
        self.customers_df.drop_duplicates(inplace=True)

    def _convert_types(self):
        self.books_df['Id'] = pd.to_numeric(self.books_df['Id'], errors='coerce')
        self.books_df['Customer ID'] = pd.to_numeric(self.books_df['Customer ID'], errors='coerce')
        self.customers_df['Customer ID'] = pd.to_numeric(self.customers_df['Customer ID'], errors='coerce')

        self.books_df.dropna(subset=['Id', 'Customer ID'], inplace=True)
        self.customers_df.dropna(subset=['Customer ID'], inplace=True)

        self.books_df['Id'] = self.books_df['Id'].astype(int)
        self.books_df['Customer ID'] = self.books_df['Customer ID'].astype(int)
        self.customers_df['Customer ID'] = self.customers_df['Customer ID'].astype(int)

    def _clean_dates(self):
        for col in ["Book checkout", "Book Returned"]:
            if col in self.books_df.columns:
                quotes_before = self.books_df[col].astype(str).str.count('"').sum()
                self.books_df[col] = self.books_df[col].astype(str).str.replace('"', '')
                quotes_after = self.books_df[col].astype(str).str.count('"').sum()
                self.quotes_removed += (quotes_before - quotes_after)

                parsed_dates = pd.to_datetime(self.books_df[col], dayfirst=True, errors='coerce')
                self.invalid_dates_fixed += parsed_dates.isna().sum()
                self.books_df[col] = parsed_dates

        # Count rows with NULL dates before dropping them
        null_dates_rows = self.books_df[
            self.books_df["Book checkout"].isna() | self.books_df["Book Returned"].isna()
        ].shape[0]

        self.rows_removed_invalid_dates = null_dates_rows

        self.books_df.dropna(subset=["Book checkout", "Book Returned"], inplace=True)


    def _calculate_loan_duration(self):
        self.books_df["LoanDurationDays"] = (self.books_df["Book Returned"] - self.books_df["Book checkout"]).dt.days

    def _remove_invalid_loans(self):
        invalid_loans = self.books_df[self.books_df["LoanDurationDays"] < 0].shape[0]
        self.rows_removed_negative_loan = invalid_loans
        self.books_df = self.books_df[self.books_df["LoanDurationDays"] >= 0]



class DatabaseWriter:
    def __init__(self, credentials):
        raw_conn_str = (
            f"DRIVER={credentials['driver']};"
            f"SERVER={credentials['server']};"
            f"DATABASE={credentials['database']};"
            f"Trusted_Connection={credentials['trusted_connection']};"
        )

        print(f"[DEBUG] Connection string before encoding:\n{raw_conn_str}\n")

        params = urllib.parse.quote_plus(raw_conn_str)
        final_conn_str = f"mssql+pyodbc:///?odbc_connect={params}"

        print(f"[DEBUG] SQLAlchemy connection string:\n{final_conn_str}\n")

        self.engine = create_engine(final_conn_str)


    def write_tables(self, books_df, customers_df):
        books_df.to_sql('SystemBook', con=self.engine, if_exists='replace', index=False)
        customers_df.to_sql('SystemCustomer', con=self.engine, if_exists='replace', index=False)

    def create_log_table(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                IF NOT EXISTS (
                    SELECT * FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_NAME = 'CleaningLog'
                )
                CREATE TABLE CleaningLog (
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
                    InvalidDatesFixed INT,
                    BooksRowsRemovedInvalidDates INT,
                    BooksRowsRemovedNegativeLoan INT
                )
            """))

    def write_log(self, log_df):
        log_df.to_sql("CleaningLog", con=self.engine, if_exists='append', index=False)

# Orchestration
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env = EnvLoader(script_dir)
    creds = env.get_db_credentials()

    file_mgr = FileManager(script_dir)
    books_df, customers_df = file_mgr.read_csvs()

    books_rows_initial = len(books_df)
    customers_rows_initial = len(customers_df)

    cleaner = DataCleaner(books_df, customers_df)
    (
        books_df,
        customers_df,
        quotes_removed,
        invalid_dates_fixed,
        rows_removed_invalid_dates,
        rows_removed_negative_loan
    ) = cleaner.clean()

    db_writer = DatabaseWriter(creds)
    db_writer.write_tables(books_df, customers_df)
    db_writer.create_log_table()

    clean_books_file, clean_customers_file = file_mgr.save_cleaned(books_df, customers_df)

    # Log details
    log_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Quotes removed: {quotes_removed} | Invalid dates fixed: {invalid_dates_fixed}\n"
    file_mgr.write_log_txt(log_text)

    log_df = pd.DataFrame([{
        "Timestamp": datetime.now(),
        "BooksRowsInitial": len(books_df),
        "BooksRowsAfterEmptyDrop": len(books_df),
        "BooksRowsAfterDupesDrop": len(books_df),
        "BooksRowsAfterInvalidIDDrop": len(books_df),
        "BooksDroppedInvalidIDs": 0,  # Placeholder
        "CustomersRowsInitial": len(customers_df),
        "CustomersRowsAfterEmptyDrop": len(customers_df),
        "CustomersRowsAfterDupesDrop": len(customers_df),
        "CustomersRowsAfterInvalidIDDrop": len(customers_df),
        "CustomersDroppedInvalidIDs": 0,  # Placeholder
        "QuotesRemoved": quotes_removed,
        "InvalidDatesFixed": invalid_dates_fixed
    }])

    db_writer = DatabaseWriter(creds)
    db_writer.write_tables(books_df, customers_df)
    db_writer.create_log_table()

    clean_books_file, clean_customers_file = file_mgr.save_cleaned(books_df, customers_df)

    # Log details
    log_text = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Quotes removed: {quotes_removed} | "
        f"Invalid dates fixed: {invalid_dates_fixed} | "
        f"Rows removed (NULL/invalid dates): {rows_removed_invalid_dates} | "
        f"Rows removed (checkout after return): {rows_removed_negative_loan}\n"
    )
    file_mgr.write_log_txt(log_text)

    log_df = pd.DataFrame([{
        "Timestamp": datetime.now(),
        "BooksRowsInitial": books_rows_initial,
        "BooksRowsAfterEmptyDrop": len(books_df),
        "BooksRowsAfterDupesDrop": len(books_df),
        "BooksRowsAfterInvalidIDDrop": len(books_df),
        "BooksDroppedInvalidIDs": 0,  # Placeholder
        "CustomersRowsInitial": customers_rows_initial,
        "CustomersRowsAfterEmptyDrop": len(customers_df),
        "CustomersRowsAfterDupesDrop": len(customers_df),
        "CustomersRowsAfterInvalidIDDrop": len(customers_df),
        "CustomersDroppedInvalidIDs": 0,  # Placeholder
        "QuotesRemoved": quotes_removed,
        "InvalidDatesFixed": invalid_dates_fixed,
        "BooksRowsRemovedInvalidDates": rows_removed_invalid_dates,
        "BooksRowsRemovedNegativeLoan": rows_removed_negative_loan
    }])


    db_writer.write_log(log_df)

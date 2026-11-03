
import csv
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "olist"


# --------------------------------------------------
# 2. DATASET FILES
# --------------------------------------------------

DATASET_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# --------------------------------------------------
# 3. GENERIC CSV LOADER
# --------------------------------------------------

def load_csv(filename):
    """
    Load a CSV file and return its records as dictionaries.
    """

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


# --------------------------------------------------
# 4. LOAD ALL NINE DATASETS
# --------------------------------------------------

def load_olist_data():
    """
    Load all nine Olist datasets.
    """

    data = {}

    for dataset_name, filename in DATASET_FILES.items():

        print(f"Loading {filename}...")

        records = load_csv(filename)

        data[dataset_name] = records

        print(
            f"Successfully loaded {len(records)} records."
        )

    return data


# --------------------------------------------------
# 5. DATASET SUMMARY
# --------------------------------------------------

def print_dataset_summary(data):

    print("\n" + "=" * 45)
    print("       OLIST DATASET SUMMARY")
    print("=" * 45)

    for dataset_name, records in data.items():

        print(
            f"{dataset_name:<25} : {len(records):>10} records"
        )

    print("=" * 45)


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    olist_data = load_olist_data()

    print_dataset_summary(olist_data)

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Add backend directory to Python path
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BACKEND_DIR))


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
from app.utils.config import (
    ACCOUNTS_FILE,
    CUSTOMERS_FILE,
    create_directories,
)


# ---------------------------------------------------------
# Generate Accounts
# ---------------------------------------------------------
def generate_accounts(
    min_accounts_per_customer: int = 1,
    max_accounts_per_customer: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic bank accounts linked to customers.

    Each customer receives between
    min_accounts_per_customer and
    max_accounts_per_customer accounts.
    """

    # Reproducibility
    np.random.seed(seed)

    # -----------------------------------------------------
    # Load customers
    # -----------------------------------------------------
    customers = pd.read_csv(CUSTOMERS_FILE)

    records = []

    account_counter = 1

    # -----------------------------------------------------
    # Account configuration
    # -----------------------------------------------------
    account_types = [
        "SAVINGS",
        "CURRENT",
        "BUSINESS",
    ]

    account_statuses = [
        "ACTIVE",
        "ACTIVE",
        "ACTIVE",
        "DORMANT",
    ]

    currencies = [
        "INR",
        "USD",
        "GBP",
        "SGD",
        "AED",
    ]

    branches = [
        "BR001",
        "BR002",
        "BR003",
        "BR004",
        "BR005",
    ]

    # -----------------------------------------------------
    # Generate accounts for each customer
    # -----------------------------------------------------
    for _, customer in customers.iterrows():

        num_accounts = np.random.randint(
            min_accounts_per_customer,
            max_accounts_per_customer + 1,
        )

        for _ in range(num_accounts):

            # Unique account ID
            account_id = f"A{account_counter:06d}"

            # Initial balance
            initial_balance = round(
                np.random.uniform(
                    5000,
                    500000,
                ),
                2,
            )

            # Random opening date
            opening_date = (
                pd.Timestamp("2015-01-01")
                + pd.Timedelta(
                    days=np.random.randint(
                        0,
                        4000,
                    )
                )
            )

            # -------------------------------------------------
            # Create account record
            # -------------------------------------------------
            records.append(
                {
                    "account_id": account_id,

                    "customer_id":
                        customer["customer_id"],

                    "account_type":
                        np.random.choice(
                            account_types
                        ),

                    "account_status":
                        np.random.choice(
                            account_statuses
                        ),

                    "currency":
                        np.random.choice(
                            currencies
                        ),

                    "country":
                        customer["residency_country"],

                    "branch_id":
                        np.random.choice(
                            branches
                        ),

                    "opening_date":
                        opening_date,

                    "initial_balance":
                        initial_balance,

                    "current_balance":
                        initial_balance,

                    "risk_rating":
                        customer["risk_rating"],
                }
            )

            account_counter += 1

    # ---------------------------------------------------------
    # Convert records to DataFrame
    # ---------------------------------------------------------
    df = pd.DataFrame(records)

    # ---------------------------------------------------------
    # Create required directories
    # ---------------------------------------------------------
    create_directories()

    # ---------------------------------------------------------
    # Save accounts using configured path
    # ---------------------------------------------------------
    df.to_csv(
        ACCOUNTS_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Output information
    # ---------------------------------------------------------
    print(f"Generated {len(df)} accounts.")
    print(f"Saved to: {ACCOUNTS_FILE}")

    # Verify using configured path
    print(
        f"Final account count: "
        f"{len(pd.read_csv(ACCOUNTS_FILE))}"
    )

    return df


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    generate_accounts()
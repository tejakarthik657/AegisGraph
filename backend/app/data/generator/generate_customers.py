import sys
from pathlib import Path

import pandas as pd
from faker import Faker

# Add backend to Python path
BACKEND_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BACKEND_DIR))

from app.utils.config import (
    CUSTOMERS_FILE,
    create_directories,
)


fake = Faker("en_IN")


def generate_customers(
    num_customers: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic AML-oriented customer data.

    The schema is designed to resemble fields that could
    reasonably exist in a banking/AML analytics dataset.
    """

    Faker.seed(seed)

    records = []

    countries = [
        "IN",
        "US",
        "GB",
        "SG",
        "AE",
    ]

    occupations = [
        "ENGINEER",
        "TEACHER",
        "BUSINESS_OWNER",
        "STUDENT",
        "DOCTOR",
        "ACCOUNTANT",
        "MANAGER",
        "CONSULTANT",
    ]

    risk_ratings = [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    kyc_statuses = [
        "VERIFIED",
        "PENDING",
    ]

    for i in range(1, num_customers + 1):

        customer_id = f"C{i:05d}"

        country = fake.random_element(
            countries
        )

        date_of_birth = fake.date_of_birth(
            minimum_age=18,
            maximum_age=75,
        )

        customer_since = fake.date_between(
            start_date="-10y",
            end_date="today",
        )

        records.append({
            "customer_id": customer_id,

            "customer_type": "INDIVIDUAL",

            "date_of_birth": date_of_birth,

            "country": country,

            "residency_country": country,

            "occupation": fake.random_element(
                occupations
            ),

            "customer_since": customer_since,

            "risk_rating": fake.random_element(
                risk_ratings
            ),

            "kyc_status": fake.random_element(
                kyc_statuses
            ),
        })

    df = pd.DataFrame(records)

    create_directories()

    df.to_csv(
        CUSTOMERS_FILE,
        index=False,
    )

    print(
        f"Generated {len(df)} customers."
    )

    print(
        f"Saved to: {CUSTOMERS_FILE}"
    )

    return df


if __name__ == "__main__":
    generate_customers()
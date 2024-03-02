import pandas as pd
import numpy as np
from faker import Faker
from enum import Enum


class BinaryChoice(Enum):
    YES = 1
    NO = 0

def generate_rating_with_random_mode():
    random_mode = np.random.uniform(0, 5)
    raw_rating = round(np.random.triangular(0, random_mode, 5), 1)
    return 0 if 0.1 <= raw_rating < 1.0 else raw_rating

def generate_single_data_point(fake):
    support_totp = BinaryChoice.YES if np.random.choice([True, False], p=[0.5, 0.5]) else BinaryChoice.NO
    support_facial_recognition = BinaryChoice.YES if np.random.choice([True, False], p=[0.5, 0.5]) else BinaryChoice.NO
    multiple_cryptocurrencies = BinaryChoice.YES if np.random.choice([True, False], p=[0.5, 0.5]) else BinaryChoice.NO
    non_custodial = BinaryChoice.YES if np.random.choice([True, False], p=[0.5, 0.5]) else BinaryChoice.NO
    custodial = BinaryChoice.YES if np.random.choice([True, False], p=[0.5, 0.5]) else BinaryChoice.NO

    # Ensure that 'Non-Custodial' and 'Custodial' are not both NO (0)
    if non_custodial == custodial == BinaryChoice.NO:
        # Randomly set one to YES
        if np.random.choice([True, False]):
            non_custodial = BinaryChoice.YES
        else:
            custodial = BinaryChoice.YES

    return {
        "Wallet Name": fake.company(),
        "Support TOTP": support_totp.value,
        "Support Facial Recognition": support_facial_recognition.value,
        "Multiple Cryptocurrencies": multiple_cryptocurrencies.value,
        "Wallet Age": fake.random_int(min=0, max=15),
        "Non-Custodial": non_custodial.value,
        "Custodial": custodial.value,
        "Rating": generate_rating_with_random_mode()
    }


def generate_synthetic_data(num_data_points=10000):
    fake = Faker()
    np.random.seed(42)
    return pd.DataFrame([generate_single_data_point(fake) for _ in range(num_data_points)])

def normalize_data(df):
    numerical_columns = ['Wallet Age', 'Rating']
    for column in numerical_columns:
        df[column] = df[column].apply(lambda x: round((x - df[column].min()) / (df[column].max() - df[column].min()), 1))
    return df

# Function to add Security Level
def add_security_level(row):
    if row['Support TOTP'] == 1 and row['Support Facial Recognition'] == 1:
        return 0.9
    elif row['Support TOTP'] == 1:
        return 0.6
    elif row['Support Facial Recognition'] == 1:
        return 0.7
    else:
        return 0.4

df = generate_synthetic_data(10000)
df_normalized = normalize_data(df)

# Adding the Security Level column
df['Security Level1'] = df.apply(add_security_level, axis=1)

# Normalizing the Security Level
df['Security Level'] = df['Security Level1'].apply(
    lambda x: round((x - df['Security Level1'].min()) / (df['Security Level1'].max() - df['Security Level1'].min()), 1)
)
df_normalized.to_excel("Dataset", index=False)

import pandas as pd
import random

# Define synthetic thyroid test data with additional diseases
data_thyroid = {
    "Age": [random.randint(18, 80) for _ in range(10000)],
    "Gender": [random.choice(["M", "F"]) for _ in range(10000)],
    "TSH": [round(random.uniform(0.1, 10.0), 2) for _ in range(10000)],
    "T3": [round(random.uniform(0.5, 3.0), 2) for _ in range(10000)],
    "T4": [round(random.uniform(4.5, 13.0), 2) for _ in range(10000)],
    "FT3": [round(random.uniform(2.0, 7.0), 2) for _ in range(10000)],
    "FT4": [round(random.uniform(0.7, 2.0), 2) for _ in range(10000)],
    "Thyroid_Peroxidase_Antibodies": [random.randint(0, 200) for _ in range(10000)]
}

# Define disease conditions based on thyroid test values
def assign_thyroid_disease(row):
    if row["TSH"] > 4.5 and row["T3"] < 0.8:
        return "Hypothyroidism"
    elif row["TSH"] < 0.3 and row["T3"] > 2.0:
        return "Hyperthyroidism"
    elif row["Thyroid_Peroxidase_Antibodies"] > 100:
        return "Hashimoto's Thyroiditis"
    elif row["TSH"] > 6.0 and row["FT4"] < 0.8:
        return "Severe Hypothyroidism"
    else:
        return "Normal"

# Create DataFrame
df_thyroid = pd.DataFrame(data_thyroid)
df_thyroid["Disease"] = df_thyroid.apply(assign_thyroid_disease, axis=1)

# Save CSV file locally
df_thyroid.to_csv("thyroid_test.csv", index=False)

print(" Dataset saved as 'thyroid_test.csv'.")

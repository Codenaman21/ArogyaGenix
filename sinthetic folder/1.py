import pandas as pd
import random

# Define synthetic blood test data with additional diseases
data = {
    "Age": [random.randint(18, 80) for _ in range(10000)],
    "Gender": [random.choice(["M", "F"]) for _ in range(10000)],
    "Hemoglobin": [round(random.uniform(7, 18), 1) for _ in range(10000)],
    "Platelet_Count": [random.randint(100000, 450000) for _ in range(10000)],
    "White_Blood_Cells": [random.randint(2500, 15000) for _ in range(10000)],
    "Red_Blood_Cells": [round(random.uniform(3.0, 6.5), 1) for _ in range(10000)],
    "MCV": [random.randint(70, 110) for _ in range(10000)],
    "MCH": [random.randint(20, 40) for _ in range(10000)],
    "MCHC": [random.randint(28, 38) for _ in range(10000)],
    "Glucose": [random.randint(60, 250) for _ in range(10000)],
    "Creatinine": [round(random.uniform(0.4, 2.5), 2) for _ in range(10000)],
    "TSH": [round(random.uniform(0.1, 10.0), 2) for _ in range(10000)],
}

# Define disease conditions based on multiple factors
def assign_disease(row):
    if row["Hemoglobin"] < 10:
        return "Anemia"
    elif row["White_Blood_Cells"] > 11000:
        return "Infection"
    elif row["Red_Blood_Cells"] < 4.0:
        return "Leukemia"
    elif row["Glucose"] > 180:
        return "Diabetes"
    elif row["Creatinine"] > 1.5:
        return "Kidney Disease"
    elif row["TSH"] > 4.5:
        return "Thyroid Disorder"
    else:
        return "Normal"

# Create DataFrame
df = pd.DataFrame(data)
df["Disease"] = df.apply(assign_disease, axis=1)

# Save CSV file locally
df.to_csv("blood_test_extended.csv", index=False)

print("Dataset saved as 'blood_test_extended.csv'.")

import pandas as pd
import random

# Define synthetic urine test data with additional diseases
data_urine = {
    "Age": [random.randint(18, 80) for _ in range(10000)],
    "Gender": [random.choice(["M", "F"]) for _ in range(10000)],
    "pH": [round(random.uniform(4.5, 8.5), 1) for _ in range(10000)],
    "Protein": [random.choice(["Negative", "Trace", "+1", "+2", "+3"]) for _ in range(10000)],
    "Glucose": [random.choice(["Negative", "+1", "+2", "+3"]) for _ in range(10000)],
    "Ketones": [random.choice(["Negative", "+1", "+2", "+3"]) for _ in range(10000)],
    "Blood": [random.choice(["Negative", "Trace", "+1", "+2", "+3"]) for _ in range(10000)],
    "Leukocytes": [random.choice(["Negative", "Trace", "+1", "+2", "+3"]) for _ in range(10000)],
    "Nitrite": [random.choice(["Negative", "Positive"]) for _ in range(10000)],
    "Specific_Gravity": [round(random.uniform(1.000, 1.030), 3) for _ in range(10000)],
}

# Define disease conditions based on urine test values
def assign_urine_disease(row):
    if row["Protein"] in ["+2", "+3"]:
        return "Kidney Disease"
    elif row["Glucose"] in ["+2", "+3"]:
        return "Diabetes"
    elif row["Ketones"] in ["+2", "+3"]:
        return "Diabetic Ketoacidosis"
    elif row["Blood"] in ["+2", "+3"]:
        return "Urinary Tract Infection (UTI)"
    elif row["Leukocytes"] in ["+2", "+3"] or row["Nitrite"] == "Positive":
        return "Bacterial Infection"
    elif row["pH"] < 5.0:
        return "Acidosis"
    elif row["pH"] > 8.0:
        return "Alkalosis"
    else:
        return "Normal"

# Create DataFrame
df_urine = pd.DataFrame(data_urine)
df_urine["Disease"] = df_urine.apply(assign_urine_disease, axis=1)

# Save CSV file locally
df_urine.to_csv("urine_test_extended.csv", index=False)

print(" Dataset saved as 'urine_test_extended.csv'.")

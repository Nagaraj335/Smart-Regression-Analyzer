import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample housing dataset
n_samples = 1000

# Create features
size = np.random.normal(2000, 500, n_samples)  # House size in sq ft
bedrooms = np.random.poisson(3, n_samples)  # Number of bedrooms
bathrooms = bedrooms * 0.7 + np.random.normal(0, 0.3, n_samples)  # Bathrooms correlated with bedrooms
age = np.random.uniform(1, 50, n_samples)  # Age of house in years
location_score = np.random.uniform(1, 10, n_samples)  # Location desirability score
garage = np.random.binomial(1, 0.7, n_samples)  # Has garage (0 or 1)

# Create target variable (house price) with realistic relationships
price = (
    size * 150 +  # $150 per sq ft
    bedrooms * 10000 +  # $10k per bedroom
    bathrooms * 15000 +  # $15k per bathroom
    (50 - age) * 1000 +  # Newer houses worth more
    location_score * 8000 +  # Location premium
    garage * 25000 +  # Garage adds $25k
    np.random.normal(0, 30000, n_samples)  # Random noise
)

# Ensure realistic values
size = np.clip(size, 800, 4000)
bedrooms = np.clip(bedrooms, 1, 6)
bathrooms = np.clip(bathrooms, 1, 4)
price = np.clip(price, 100000, 800000)

# Create DataFrame
df = pd.DataFrame({
    'Size_SqFt': size.round(0).astype(int),
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms.round(1),
    'Age_Years': age.round(1),
    'Location_Score': location_score.round(1),
    'Has_Garage': garage,
    'Price': price.round(0).astype(int)
})

# Save to CSV
df.to_csv('d:/Data_Science/sample_housing_data.csv', index=False)

print("Sample housing dataset created!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset statistics:")
print(df.describe())
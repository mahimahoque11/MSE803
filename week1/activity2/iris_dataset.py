from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch the Iris dataset
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

# Combine features and target for duplicate checking
dataset = pd.concat([X, y], axis=1)

number_of_features = X.shape[1]
number_of_classes = y["class"].nunique()
class_names = y["class"].unique()
duplicate_count = dataset.duplicated().sum()

print("First five records:")
print(dataset.head())

print(f"\nNumber of records: {dataset.shape[0]}")
print(f"Number of features: {number_of_features}")
print(f"Number of classes: {number_of_classes}")
print(f"Class names: {list(class_names)}")
print(f"Number of duplicate records: {duplicate_count}")
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

labels = ["sit", "walk", "run", "squat"]


def load_training_data(csv_files):
    # Load accelerometer data from CSV files
    X = []
    y = []

    for csv_file in csv_files:
        print(f"Loading {csv_file}...")
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    features = [
                        float(row["Ax"]),
                        float(row["Ay"]),
                        float(row["Az"]),
                        float(row["A_mag"]),
                    ]
                except (TypeError, ValueError):
                    print(f"Skipping invalid row in {csv_file}: {row}")
                    continue
                X.append(features)
                y.append(row["label_name"])

    return np.array(X), np.array(y)


def train_svm(csv_files, model_path="svm_model.pkl"):
    # Train multiclass SVM and save to file
    print("Loading training data...")
    X, y = load_training_data(csv_files)

    total_samples = len(X)
    print(f"Loaded {total_samples} samples")

    label_counts = Counter(y)
    print("Samples per label:")
    for label in labels:
        print(f"  {label}: {label_counts.get(label, 0)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training SVM classifier...")
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        decision_function_shape="ovr",
        random_state=42,
    )
    svm.fit(X_scaled, y)

    print(f"Training complete. Accuracy: {svm.score(X_scaled, y):.2%}")

    with open(model_path, "wb") as f:
        pickle.dump({"model": svm, "scaler": scaler}, f)
    print(f"Model saved to {model_path}")

    return svm, scaler


def main():
    if len(sys.argv) < 2:
        csv_files = list(Path(".").glob("log_*.csv"))
        if not csv_files:
            print("Usage: python train.py <csv_file1> <csv_file2> ...")
            print(
                "\nOr place log_*.csv files in the current directory for auto-discovery"
            )
            print("Error: No CSV files found.")
            sys.exit(1)
        csv_files = [str(f) for f in csv_files]
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
    else:
        csv_files = sys.argv[1:]

    train_svm(csv_files)


if __name__ == "__main__":
    main()

import csv
import pickle
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

labels = ["sit", "walk", "run", "squat"]
DEFAULT_WINDOW_SIZE = 10


def summarize_window(window):
    array = np.array(window)
    features = []
    for axis in range(array.shape[1]):
        axis_data = array[:, axis]
        features.extend(
            [
                np.mean(axis_data),
                np.std(axis_data),
                np.max(axis_data),
                np.min(axis_data),
            ]
        )
    return np.array(features)


def load_training_data(csv_files, window_size=10):
    # Load accelerometer data from CSV files and emit 16-dim window summaries
    X = []
    y = []
    window = deque(maxlen=window_size)

    for csv_file in csv_files:
        print(f"Loading {csv_file}...")
        window.clear()
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
                window.append((features, row["label_name"]))
                if len(window) == window_size:
                    X.append(summarize_window([item[0] for item in window]))
                    label_counts = Counter(item[1] for item in window)
                    y.append(label_counts.most_common(1)[0][0])

    return np.array(X), np.array(y)


def _build_svm():
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        decision_function_shape="ovr",
        random_state=42,
    )


def train_svm(csv_files, model_path="svm_model.pkl", window_size=DEFAULT_WINDOW_SIZE):
    # Train multiclass SVM and save to file
    print("Loading training data...")
    X, y = load_training_data(csv_files, window_size)

    total_samples = len(X)
    print(f"Loaded {total_samples} windowed samples (window_size={window_size})")

    label_counts = Counter(y)
    print("Samples per label:")
    for label in labels:
        print(f"  {label}: {label_counts.get(label, 0)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    validation_scaler = StandardScaler()
    X_train_scaled = validation_scaler.fit_transform(X_train)
    svm = _build_svm()
    svm.fit(X_train_scaled, y_train)

    print(f"Training accuracy (train split): {svm.score(X_train_scaled, y_train):.2%}")

    X_val_scaled = validation_scaler.transform(X_val)
    y_val_pred = svm.predict(X_val_scaled)

    print("\nValidation metrics (hold-out split):")
    print(classification_report(y_val, y_val_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred, labels=labels))

    print("\nRetraining on entire dataset for final model...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_svm = _build_svm()
    final_svm.fit(X_scaled, y)

    print(
        f"Training complete. Accuracy (full data): {final_svm.score(X_scaled, y):.2%}"
    )

    with open(model_path, "wb") as f:
        pickle.dump({"model": final_svm, "scaler": final_scaler}, f)
    print(f"Model saved to {model_path}")

    return final_svm, final_scaler


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

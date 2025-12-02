import math
import pickle
import sys
import time
from collections import deque

import numpy as np
from sense_hat import SenseHat

sense = SenseHat()

labels = ["sit", "walk", "run", "squat"]

COLORS = {
    "sit": (0, 0, 255),
    "walk": (0, 255, 0),
    "run": (255, 0, 0),
    "squat": (255, 255, 0),
}

DEFAULT_WINDOW_SIZE = 10


def extract_window_features(window):
    if len(window) == 0:
        return np.zeros(16)

    window = np.array(window)

    features = []
    for i in range(window.shape[1]):
        axis_data = window[:, i]
        features.extend(
            [
                np.mean(axis_data),
                np.std(axis_data),
                np.max(axis_data),
                np.min(axis_data),
            ]
        )

    return np.array(features)


def load_model(model_path="svm_model.pkl"):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


def show_classification(label_name):
    color = COLORS.get(label_name, (255, 255, 255))
    sense.clear(color)


def live_classify(
    model_path="svm_model.pkl", window_size=DEFAULT_WINDOW_SIZE, sample_rate=0.5
):
    print("Loading model...")
    svm, scaler = load_model(model_path)
    print("Model loaded successfully!")

    window = deque(maxlen=window_size)

    print(f"\nStarting live classification...")
    print(f"Window size: {window_size} samples ({window_size * sample_rate}s)")
    print(f"Sample rate: {sample_rate}s")
    print("\nPress Ctrl+C to stop.\n")

    last_label = None

    try:
        while True:
            a = sense.get_accelerometer_raw()
            Ax, Ay, Az = a["x"], a["y"], a["z"]
            A_mag = math.sqrt(Ax**2 + Ay**2 + Az**2)

            window.append([Ax, Ay, Az, A_mag])

            if len(window) == window_size:
                features = extract_window_features(window)
                features_scaled = scaler.transform([features])

                prediction = svm.predict(features_scaled)[0]
                confidence = np.max(svm.decision_function(features_scaled))

                if prediction != last_label:
                    show_classification(prediction)
                    print(
                        f"Classified: {prediction.upper()} (confidence: {confidence:.2f})"
                    )
                    last_label = prediction

            time.sleep(sample_rate)

    except KeyboardInterrupt:
        print("\n\nStopped classification.")
        sense.clear()


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "svm_model.pkl"
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WINDOW_SIZE
    sample_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    live_classify(model_path, window_size, sample_rate)


if __name__ == "__main__":
    main()

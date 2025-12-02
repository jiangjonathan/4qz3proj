import csv
import math
import sys
import time
from datetime import datetime

from sense_hat import SenseHat

sense = SenseHat()

labels = ["sit", "walk", "run", "squat"]
current_label = 0

filename = "log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"


def show_label(idx):
    sense.show_message(labels[idx], text_colour=(0, 255, 0), scroll_speed=0.05)


with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "Ax", "Ay", "Az", "A_mag", "label_name"])

    while True:
        events = sense.stick.get_events()
        for e in events:
            if e.action == "pressed":
                if e.direction == "up":
                    current_label = 1
                    print("Selected: WALK")
                    show_label(current_label)
                elif e.direction == "down":
                    current_label = 0
                    print("Selected: SIT")
                    show_label(current_label)
                elif e.direction == "right":
                    current_label = 2
                    print("Selected: RUN")
                    show_label(current_label)
                elif e.direction == "left":
                    current_label = 3
                    print("Selected: SQUAT")
                    show_label(current_label)
                elif e.direction == "middle":
                    continue

                print(f"Starting 15s recording for '{labels[current_label]}'...")
                samples = int(15 / 0.5)
                aborted = False

                for i in range(samples):
                    rec_events = sense.stick.get_events()
                    for re in rec_events:
                        if re.action == "pressed" and re.direction == "middle":
                            print("\n⚠ Segment aborted by user.")
                            aborted = True
                            break
                    if aborted:
                        break

                    a = sense.get_accelerometer_raw()
                    Ax, Ay, Az = a["x"], a["y"], a["z"]
                    A_mag = math.sqrt(Ax**2 + Ay**2 + Az**2)

                    writer.writerow(
                        [
                            datetime.now().isoformat(),
                            Ax,
                            Ay,
                            Az,
                            A_mag,
                            labels[current_label],
                        ]
                    )
                    f.flush()

                    # overwrite same line with progress
                    print(
                        f"\r[{i + 1}/{samples}] Recording {labels[current_label]}...",
                        end="",
                    )
                    sys.stdout.flush()
                    time.sleep(0.5)

                if not aborted:
                    print(f"\n Finished recording '{labels[current_label]}'.\n")

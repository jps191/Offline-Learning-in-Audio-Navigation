import re
import csv

LOG_FILE = "training_log_SAC_Run4.txt"
CSV_FILE = "training_data_SAC.csv"

pattern = re.compile(
    r"Step:\s*(\d+)\.\s*Time Elapsed:\s*([\d.]+)\s*s\.\s*Mean Reward:\s*([-.\d]+)\.\s*Std of Reward:\s*([-.\d]+)\.",
    re.MULTILINE
)

def parse_training_log(path):
    rows = []
    seen_steps = set()
    duplicates = []

    with open(path, "r", encoding="utf-16") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                time_elapsed = float(match.group(2))
                mean_reward = float(match.group(3))
                std_reward = float(match.group(4))

                # Check for duplicate step values
                if step in seen_steps:
                    duplicates.append(step)
                    # OPTIONAL: continue   # uncomment to skip writing duplicates
                else:
                    seen_steps.add(step)

                rows.append([step, time_elapsed, mean_reward, std_reward])

    return rows, duplicates


def save_to_csv(rows, out_path):
    header = ["step", "time_elapsed", "mean_reward", "std_reward"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    rows, duplicates = parse_training_log(LOG_FILE)
    save_to_csv(rows, CSV_FILE)

    print(f"Extracted {len(rows)} total rows → {CSV_FILE}")
    print(f"Found {len(duplicates)} duplicate steps")

    if duplicates:
        print("Duplicate step values:")
        print(sorted(duplicates))

import os
import subprocess


def process_all_files():
    raw_data_dir = "data/impact_data"

    if not os.path.exists(raw_data_dir):
        print(f"Directory not found: {raw_data_dir}")
        return

    for root, dirs, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.endswith(".csv"):
                filepath = os.path.join(root, filename)
                command = [
                    "python",
                    "preprocessing/preprocess.py",
                    filepath,
                ]
                print(f"Executing: {' '.join(command)}")
                subprocess.run(command, check=True)


if __name__ == "__main__":
    process_all_files()

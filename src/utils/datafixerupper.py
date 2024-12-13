#
# Takes gregg-1916 dataset with labels in filenames and converts to CSV-labeled data
#

import os
import csv
from pathlib import Path

data_folder = '../data'  
output_folder = '../data-labeled' 

def generate_csv_from_data_folder(data_folder, output_folder, csv_filename='labels.csv'):
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_path = output_folder / csv_filename

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'label'])

        for file in data_folder.iterdir():
            if file.is_file():
                label = file.stem  # Get the filename without the extension
                writer.writerow([file.name, label])

    print(f"CSV file generated at: {csv_path}")

generate_csv_from_data_folder(data_folder, output_folder)
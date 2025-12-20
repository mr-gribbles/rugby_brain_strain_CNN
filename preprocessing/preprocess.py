"""
Takes a cleaned impact time series CSV file as input, and applies teh preproccessing steps from the 
Convolutional neural network for efficient estimation of regional brain strains paper. This includes:
1) Generating all permutations of the XYZ axes
2) Applying the conjugate rotational vector transform
3) Centering and padding the time series to a fixed length

Saves all permutations to a single HDF5 file under a group named after the input file.

Original MATLAB implementation can be found here: https://github.com/Jilab-biomechanics/CNN-brain-strains
"""

import itertools
import os

import h5py
import numpy as np
import pandas as pd
from conjugate import conjugate_vrot_transform
from shift_and_pad import shift_and_pad
from calculate_ubric import calculate_ubric_from_profile
from link_metadata import get_metadata

def process_file(filepath, output_h5_path):
    """
    Processes a single input CSV file and saves all its augmented
    permutations to a single HDF5 file.

    Args:
        filepath (str): Path to the input CSV file.
        output_h5_path (str): Path to the output HDF5 file.
    """
    df = pd.read_csv(filepath)
    
    # Calculate sampling frequency
    time = df.iloc[:, 0].astype(float).to_numpy()
    fs = 1 / (time[1] - time[0])
    
    profile = df.iloc[:, [4, 5, 6]].to_numpy()
    cnn_length = 2000
    axes_permutations = list(itertools.permutations([0, 1, 2]))
    axes_labels = ["x", "y", "z"]
    target_idx = cnn_length // 2
    base_name = os.path.basename(filepath)
    group_name, _ = os.path.splitext(base_name)

    # Get metadata prediction and ubric score
    pred, impact_location = get_metadata(filepath)
    ubric_score = calculate_ubric_from_profile(profile, time)
    with h5py.File(output_h5_path, "a") as hf:
        if group_name in hf:
            del hf[group_name]
        group = hf.create_group(group_name)
        group.attrs["pred"] = pred
        group.attrs["impact_location"] = impact_location
        group.attrs["ubric_score"] = ubric_score
        print(f"Processing {filepath}")

        for i, perm in enumerate(axes_permutations):
            permuted = profile[:, perm]
            conj_profile = conjugate_vrot_transform(permuted)
            padded_profile = shift_and_pad(conj_profile, target_idx, cnn_length)
            cnn_input = padded_profile.T[np.newaxis, :, :]
            perm_name = "".join([axes_labels[p] for p in perm])
            dataset_name = f"perm_{perm_name}"
            group.create_dataset(dataset_name, data=cnn_input)
            print(f"Saved dataset '{dataset_name}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process impact data for CNN input")
    parser.add_argument("filepath", type=str)
    parser.add_argument(
        "--output_h5", type=str, default=None
    )

    args = parser.parse_args()
    
    output_h5_path = args.output_h5
    if output_h5_path is None:
        base_name = os.path.basename(args.filepath)
        if "_g" in base_name:
            output_h5_path = "data/impact_data_game.h5"
        elif "_tw" in base_name:
            output_h5_path = "data/impact_data_training.h5"
    process_file(args.filepath, output_h5_path)

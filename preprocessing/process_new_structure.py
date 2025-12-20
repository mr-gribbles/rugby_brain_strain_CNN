import os
import itertools
import h5py
import numpy as np
import pandas as pd
import re
from conjugate import conjugate_vrot_transform
from shift_and_pad import shift_and_pad
from calculate_ubric import calculate_ubric_from_profile

# List of all possible impact locations (from link_metadata.py)
IMPACT_LOCATIONS = [
    'Back', 'Back Left', 'Back Neck', 'Back Right', 'Back Top Left', 'Back Top Right',
    'Bottom Back', 'Bottom Back Left', 'Bottom Back Right', 'Bottom Front', 'Bottom Left',
    'Bottom Right', 'Front', 'Front Bottom Left', 'Front Bottom Right', 'Front Left',
    'Front Neck', 'Front Right', 'Front Top Left', 'Front Top Right', 'Left',
    'Left Neck', 'Right', 'Right Neck', 'Top Back', 'Top Front', 'Top Left',
    'Top Right', 'Unknown'
]

def one_hot_encode(location, locations_list):
    """
    One-hot encodes a single location string into a vector.
    """
    encoding = np.zeros(len(locations_list), dtype=int)
    try:
        # Handle potential nan/float type for location
        if pd.isna(location):
            # print("Warning: Location is NaN. Assigning to Unknown if possible or skipping.")
            # Depending on desired behavior, we might want to map to 'Unknown' or just leave as zeros
            if 'Unknown' in locations_list:
                index = locations_list.index('Unknown')
                encoding[index] = 1
            return encoding
            
        index = locations_list.index(location)
        encoding[index] = 1
    except ValueError:
        # print(f"Warning: Location '{location}' not found in the predefined list.")
        if 'Unknown' in locations_list:
            index = locations_list.index('Unknown')
            encoding[index] = 1
    return encoding

def process_file(filepath, output_h5_path, pred, impact_location, ubric_hitiq):
    """
    Processes a single input CSV file and saves all its augmented
    permutations to a single HDF5 file.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Calculate sampling frequency
        time = df.iloc[:, 0].astype(float).to_numpy()
        # fs = 1 / (time[1] - time[0]) # Unused variable
        
        # Assuming columns 4, 5, 6 correspond to angular acceleration X, Y, Z
        # This matches preprocess.py: profile = df.iloc[:, [4, 5, 6]].to_numpy()
        profile = df.iloc[:, [4, 5, 6]].to_numpy()
        
        cnn_length = 2000
        axes_permutations = list(itertools.permutations([0, 1, 2]))
        axes_labels = ["x", "y", "z"]
        target_idx = cnn_length // 2
        base_name = os.path.basename(filepath)
        group_name, _ = os.path.splitext(base_name)

        ubric_score = calculate_ubric_from_profile(profile, time)
        
        # Encode impact location
        encoded_location = one_hot_encode(impact_location, IMPACT_LOCATIONS)

        with h5py.File(output_h5_path, "a") as hf:
            if group_name in hf:
                del hf[group_name]
            group = hf.create_group(group_name)
            group.attrs["pred"] = pred
            group.attrs["impact_location"] = encoded_location
            group.attrs["ubric_score"] = ubric_score
            group.attrs["ubric_hitiq"] = ubric_hitiq
            
            for i, perm in enumerate(axes_permutations):
                permuted = profile[:, perm]
                conj_profile = conjugate_vrot_transform(permuted)
                padded_profile = shift_and_pad(conj_profile, target_idx, cnn_length)
                cnn_input = padded_profile.T[np.newaxis, :, :]
                perm_name = "".join([axes_labels[p] for p in perm])
                dataset_name = f"perm_{perm_name}"
                group.create_dataset(dataset_name, data=cnn_input)
        
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def get_h5_path(team_dir, team_name, session_type):
    if session_type == 'Training':
        return os.path.join(team_dir, f"{team_name}_training.h5")
    elif session_type == 'Game':
        return os.path.join(team_dir, f"{team_name}_game.h5")
    return None

def process_all_data():
    root_data_dir = "data"
    unknown_folders = []
    
    # Iterate over team directories
    for team_name in os.listdir(root_data_dir):
        team_path = os.path.join(root_data_dir, team_name)
        if not os.path.isdir(team_path) or team_name == 'metadata' or team_name.startswith('.'):
            continue
            
        print(f"Processing Team: {team_name}")
        
        team_metadata_rows = []
        
        # Iterate over session directories within team directory
        for session_name in os.listdir(team_path):
            session_path = os.path.join(team_path, session_name)
            if not os.path.isdir(session_path) or session_name.startswith('.'):
                continue
                
            # Determine if Training or Game
            session_type = None
            if any(x in session_name for x in ['_P', '_practice', '_T', '_Training']):
                session_type = 'Training'
            elif any(x in session_name for x in ['_game', '_G']):
                session_type = 'Game'
            
            if not session_type:
                # Log unknown folders (skipping trajectories folder if checking session name)
                # But wait, session names are directories like "2025-05-12_Game_1_vs_Merivale"
                # If checking subdirs of team_path, they should be sessions.
                # However, ensure we don't log 'training.h5' or other files, but we check isdir above.
                print(f"Skipping unknown session type: {session_name}")
                unknown_folders.append(os.path.join(team_name, session_name))
                continue
                
            h5_path = get_h5_path(team_path, team_name, session_type)
            if not h5_path:
                continue
                
            print(f"  Session: {session_name} ({session_type}) -> {h5_path}")
            
            # Find metadata CSV
            metadata_file = None
            for f in os.listdir(session_path):
                if f.endswith('.csv') and f != 'trajectories':
                    metadata_file = os.path.join(session_path, f)
                    break
            
            if not metadata_file:
                print(f"    No metadata CSV found in {session_path}")
                continue
                
            try:
                metadata_df = pd.read_csv(metadata_file)
            except Exception as e:
                print(f"    Error reading metadata {metadata_file}: {e}")
                continue
            
            # Look for trajectories folder
            trajectories_dir = os.path.join(session_path, "trajectories")
            if not os.path.exists(trajectories_dir):
                print(f"    No trajectories folder in {session_path}")
                continue
                
            # Normalize column names for easier access
            metadata_df.columns = [c.strip() for c in metadata_df.columns]
            
            # Map columns to standard names if possible
            # Prefer 'Id' > '_id'
            id_col = 'Id' if 'Id' in metadata_df.columns else ('_id' if '_id' in metadata_df.columns else None)
            
            # Prefer 'Pred' > 'prediction'
            pred_col = 'Pred' if 'Pred' in metadata_df.columns else ('prediction' if 'prediction' in metadata_df.columns else None)
            
            # Prefer 'Impact Location' > 'impact_location'
            loc_col = 'Impact Location' if 'Impact Location' in metadata_df.columns else ('impact_location' if 'impact_location' in metadata_df.columns else None)
            
            # Prefer 'UBrIC' > 'ubric'
            ubric_col = 'UBrIC' if 'UBrIC' in metadata_df.columns else ('ubric' if 'ubric' in metadata_df.columns else None)

            if not id_col:
                print(f"    No ID column found in metadata {metadata_file}")
                continue

            # Process each impact in metadata
            count = 0
            for idx, row in metadata_df.iterrows():
                impact_id = row.get(id_col)
                if pd.isna(impact_id):
                    continue
                
                impact_id = str(impact_id).strip()
                trajectory_file = os.path.join(trajectories_dir, f"{impact_id}.csv")
                
                if os.path.exists(trajectory_file):
                    pred = row.get(pred_col) if pred_col else np.nan
                    impact_loc = row.get(loc_col) if loc_col else 'Unknown'
                    ubric_val = row.get(ubric_col) if ubric_col else np.nan
                    
                    if process_file(trajectory_file, h5_path, pred, impact_loc, ubric_val):
                        count += 1
                        team_metadata_rows.append(row)
                else:
                    # Optional: print missing files
                    # print(f"    Trajectory file not found: {trajectory_file}")
                    pass
            print(f"    Processed {count} impacts")
        
        # Save aggregated metadata for the team
        if team_metadata_rows:
            team_agg_df = pd.DataFrame(team_metadata_rows)
            agg_csv_path = os.path.join(team_path, f"{team_name}_all_impacts.csv")
            team_agg_df.to_csv(agg_csv_path, index=False)
            print(f"  Saved aggregated metadata to {agg_csv_path}")

    # Save unknown folders log
    if unknown_folders:
        with open(os.path.join(root_data_dir, "unknown_folders.txt"), "w") as f:
            for folder in unknown_folders:
                f.write(f"{folder}\n")
        print(f"Logged {len(unknown_folders)} unknown folders to data/unknown_folders.txt")

if __name__ == "__main__":
    process_all_data()

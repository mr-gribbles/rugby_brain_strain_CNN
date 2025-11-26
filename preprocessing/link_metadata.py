import pandas as pd
import re
import os

def get_metadata_prediction(impact_filepath):
    """
    Parses an impact file path to find the corresponding metadata entry and return the 'pred' value.

    Args:
        impact_filepath (str): The path to the impact data file.

    Returns:
        bool: The 'pred' value from the metadata file.
    """
    base_name = os.path.basename(impact_filepath)
    
    # Regex to extract identifiers from the filename
    match = re.match(r'(\d+)_(\d+)_(g\d+|tw\d+)(?:_(\d+))?\.csv', base_name)
    if not match:
        raise ValueError(f"Filename {base_name} does not match expected pattern.")

    team_code_str, id_str, suffix, instance_str = match.groups()
    team_code = int(team_code_str)
    player_id = int(id_str)
    instance = int(instance_str) if instance_str else 0

    metadata_filename = f"metadata_{suffix}.csv"
    metadata_filepath = os.path.join('data', 'metadata', metadata_filename)

    if not os.path.exists(metadata_filepath):
        raise FileNotFoundError(f"Metadata file not found at {metadata_filepath}")

    metadata_df = pd.read_csv(metadata_filepath)

    # Find all matching rows
    matching_rows = metadata_df[(metadata_df['team_code'] == team_code) & (metadata_df['id'] == player_id)]

    if instance < len(matching_rows):
        return matching_rows.iloc[instance]['pred']
    else:
        raise IndexError(f"Instance {instance} not found for team_code {team_code} and id {player_id} in {metadata_filename}")

if __name__ == '__main__':
    # Example usage:
    test_file_1 = 'data/impact_data/005_g2/005_002_g2.csv'
    test_file_2 = 'data/impact_data/005_g2/005_004_g2_00.csv'
    
    try:
        pred_value_1 = get_metadata_prediction(test_file_1)
        print(f"Prediction for {os.path.basename(test_file_1)}: {pred_value_1}")

        pred_value_2 = get_metadata_prediction(test_file_2)
        print(f"Prediction for {os.path.basename(test_file_2)}: {pred_value_2}")
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(e)

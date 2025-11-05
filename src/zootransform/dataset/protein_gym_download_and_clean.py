import subprocess
import os
import glob
import pandas as pd

def download_gym_data(download_path: str = "src/zootransform/dataset/ProteinGym_DMS_data"):
    """
    Download and extract ProteinGym DMS substitution dataset.
    Please uncomment and run the download commands if data is not present.
    """
    print(os.getcwd())
    os.makedirs(download_path, exist_ok=True)
    url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
    zip_path = os.path.join(download_path, "DMS_ProteinGym_substitutions.zip")
    extract_dir = os.path.join(download_path)

    # Download the dataset zip file if not already downloaded
    if not os.path.exists(zip_path):
        subprocess.run(["wget", "-q", url, "-O", zip_path], check=True)

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

    # Unzip the file
    subprocess.run(["unzip", "-q", zip_path, "-d", extract_dir], check=True)

    # Remove the zip file after extraction
    if os.path.exists(zip_path):
        os.remove(zip_path)

def get_gym_data(data_dir='src/zootransform/dataset/ProteinGym_DMS_data/DMS_ProteinGym_substitutions', approved_ids: list = []):

    data_path = os.path.join(data_dir, "*.csv")
    if not approved_ids:
        with open('src/zootransform/dataset/GYM_Ids.txt', 'r') as file:
            approved_ids = file.read().splitlines()
            approved_ids = [id.strip() + '.csv' for id in approved_ids]
    # Load all CSVs into a list of dataframes
    print("Loading CSV files...")
    dfs = []
    csv_files = glob.glob(data_path)
    if not csv_files:
        print(f"⚠ No CSV files found at path: {data_path}")
        print("  Please make sure the data has been downloaded first (run Step 1 cell above).")
    else:
        count = 0
        for f in csv_files:
            filename = os.path.basename(f)
            if filename not in approved_ids:
                continue
            count += 1
            df = pd.read_csv(f)
            df["source_file"] = f.split("/")[-1].replace(".csv", "")
                        # Split by underscore
            split_cols = df['source_file'].str.split('_', expand=True)

            # Assign new columns:
            df['protein'] = split_cols[0]
            df['species'] = split_cols[1] + '_' + split_cols[2]
            dfs.append(df)
        
        # Combine into a single dataframe
        data = pd.concat(dfs, ignore_index=True)
        
        print(f"✓ Loaded {len(dfs)} files with {len(data):,} total rows")
    return data[['DMS_score', 'mutated_sequence', 'protein', 'species']]

# def clean_gym_data(data: pd.DataFrame):
#     # Make column names consistent and easier to work with
#     df = data.rename(columns={
#         'DMS_score': 'score',
#         'mutant': 'mutation',
#         'mutated_sequence': 'mut_seq',
#         'DMS_score_bin': 'score_bin',
#         'DMS_bin_score': 'score_bin_float'
#     })

#     print("✓ Column names standardized")
#     print(f"New columns: {list(df.columns)}")


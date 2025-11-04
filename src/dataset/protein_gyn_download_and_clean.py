import subprocess
import os
import glob
import pandas as pd

def download_gym_data(download_path: str = "../ProteinGym_DMS_data", download: bool = True, approved_ids: list = []):
    """
    Download and extract ProteinGym DMS substitution dataset.
    Please uncomment and run the download commands if data is not present.
    """    
    os.makedirs(download_path, exist_ok=True)
    if not approved_ids:
        with open('src/dataset/GYM_Ids.txt', 'r') as file:
            approved_ids = file.read().splitlines()
            approved_ids = [id.strip() + '.csv' for id in approved_ids]
    url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
    zip_path = os.path.join(download_path, "DMS_ProteinGym_substitutions.zip")
    extract_dir = os.path.join(download_path, "DMS_ProteinGym_substitutions")

    # Download the dataset zip file if not already downloaded
    if download and not os.path.exists(zip_path):
        subprocess.run(["wget", "-q", url, "-O", zip_path], check=True)

    # Create the extraction directory if it doesn't exist
    if download and not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

    # Unzip the file
    subprocess.run(["unzip", "-q", zip_path, "-d", extract_dir], check=True)

    # Remove the zip file after extraction
    if download and os.path.exists(zip_path):
        os.remove(zip_path)

    # Check if data was downloaded successfully (should be in parent directory of notebook)
    data_dir = os.path.join(download_path, "DMS_ProteinGym_substitutions")
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            print("✓ Data download complete!")
            print(f"  Found {len(csv_files)} CSV files in {data_dir}/")
        else:
            print(f"⚠ Directory {data_dir}/ exists but contains no CSV files.")
    else:
        print(f"⚠ Data directory '{data_dir}/' not found.")
        print("  Please uncomment the download commands above and run the cell again.")

    data_path = os.path.join(download_path, "DMS_ProteinGym_substitutions/*.csv")

    # Load all CSVs into a list of dataframes
    print("Loading CSV files...")
    dfs = []
    csv_files = glob.glob(data_path)

    if not csv_files:
        print(f"⚠ No CSV files found at path: {data_path}")
        print("  Please make sure the data has been downloaded first (run Step 1 cell above).")
    else:
        for f in csv_files:
            if f not in approved_ids:
                continue
            df = pd.read_csv(f)
            df["source_file"] = f.split("/")[-1].replace(".csv", "")
            dfs.append(df)
        
        # Combine into a single dataframe
        data = pd.concat(dfs, ignore_index=True)
        
        print(f"✓ Loaded {len(dfs)} files with {len(data):,} total rows")
    return data[['DMS_score', 'mutated_sequence']]

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


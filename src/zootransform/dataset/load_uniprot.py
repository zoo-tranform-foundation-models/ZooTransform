

import pandas as pd
import numpy as np
import os


NAMES_SPECIES_EXTRA = ['Rhodotorula toruloides',
                       'Staphylococcus aureus',
                       'Saccharolobus solfataricus']


def constrain_species_names(df, n_top=10):
    names_common = list(df['species'].value_counts().index[:n_top])
    names_funky_but_missing = ['Mesorhizobium opportunistum']

    names_selected = names_common + NAMES_SPECIES_EXTRA

    return df[df['species'].isin(names_selected)]


def process_dataset(fn_uniprot='uniprot_data/uniprot_sprot_cleaned.tsv'):

    if not os.path.exists(fn_uniprot):
        try:
            from src.zootransform.dataset.uniprot_download_and_clean import main as download_uniprot
            download_uniprot()
        except ImportError:
            raise FileNotFoundError(
                f"File {fn_uniprot} not found. Please download the UniProt dataset first." + 
                f"You can run the command: python3 src/zootransform/dataset/uniprot_download_and_clean.py")
    df = pd.read_csv(fn_uniprot, sep="\t", dtype=str, na_filter=False)
    df['species_raw'] = df['protein_name'].apply(lambda x: x.split(
        'OS=')[-1].split(' OX=')[0].strip() if 'OS=' in x else '')
    df['species'] = df['species_raw'].apply(
        lambda x: ' '.join(x.split(' ')[:2]))

    df_spec = constrain_species_names(df)
    df_spec.to_csv(
        'uniprot_data/uniprot_sprot_cleaned_selected_species.tsv', sep="\t", index=False)


def load_uniprot():
    fn_selected = 'uniprot_data/uniprot_sprot_cleaned_selected_species.tsv'
    if not os.path.exists(fn_selected):
        process_dataset()
    return pd.read_csv('uniprot_data/uniprot_sprot_cleaned_selected_species.tsv', sep="\t", dtype=str, na_filter=False)

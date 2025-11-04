

import pandas as pd


def constrain_species_names(df: pd.DataFrame, n_top=10):
    """
    Filters a DataFrame to include only rows with specific species names.
    This function selects the top `n_top` most common species names from the 
    DataFrame, adds a predefined list of additional species names, and filters 
    the DataFrame to include only rows with species names in this combined list.
    Args:
        df (pd.DataFrame): A pandas DataFrame containing a column named 'species'.
        n_top (int, optional): The number of most common species names to include. 
            Defaults to 10.
    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the 'species' 
        column matches one of the selected species names.
    """

    d = ['Haemophilus aegyptius',
    'Chlorobaculum tepidum',
    'Thermotoga maritima',
    'Influenza A virus',
    'MTH3_HAEAE_RockahShmuel_2015',
    'BCHB_CHLTE_Tsuboyama_2023_2KRU',
    'A0A2Z5U3Z0_9INFA_Doud_2016',
    'C6KNH7_9INFA_Lee_2018',
    'TRPC_THEMA_Chan_2017']
    
    names_common = list(df['species'].value_counts().index[:n_top])

    names_funky = ['Rhodotorula toruloides',
                   'Staphylococcus aureus',
                   'Saccharolobus solfataricus',
                   ]
    names_funky_but_missing = ['Mesorhizobium opportunistum']

    names_selected = names_common + names_funky
    
    return df[df['species'].isin(names_selected)]
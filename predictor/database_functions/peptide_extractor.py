import pandas as pd
import warnings

def extract_peptide_data(input_file_path, conditions_dictionary=None, iedb=True):
    """ Extracts peptides in IEDB under user defined conditions
        Returns a dictionary with keys: uniprot_id, values: peptide_list
    """
    if iedb:
        print("---> Extracting peptide data from IEDB...")
        df = generate_df(input_file_path, conditions_dictionary, iedb=True)
        print("---> Applying filtering conditions defined by the user...")
        df_filtered = apply_conditions(df, conditions_dictionary)
    else:
        print("---> Extracting peptide data from other databases ...")
        df_filtered = generate_df(input_file_path, iedb=False)

    print("---> Creating the dictionary...")
    data = create_dictionary(df_filtered)
    return data

def generate_df(input_file_path, conditions_dictionary=None, iedb=True):
    """ Only reads columns listed in dictionary keys (condition keys)
        Gets the first sequence separated by space in peptide list. Modificated peptides are weirdly anonated: peptide + modification
        Drops rows contaning NaN in any condition column
        Resets the index of the dataframe because of removing NaNs
        Returns the dataframe
    """
    if iedb:
        warnings.filterwarnings('ignore', message="^Columns.*")
        df = pd.read_csv(input_file_path, header=0, usecols=list(conditions_dictionary.keys()))
        df.rename(columns={'Description':'peptide_sequence','Parent.Protein.IRI':'uniprot_id'},inplace=True)
        df["peptide_sequence"] = df["peptide_sequence"].str.split().str[0]
    else:
        df = pd.read_csv(input_file_path)
        df['uniprot_id'].replace(to_replace="([\w]+)", value=r'http://www.uniprot.org/uniprot/\1', regex=True,inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def apply_conditions(df, conditions_dictionary):
    """ Apply conditions reported in the dictionary of conditions
        Keys represent the column names
        Values the conditions to use:
            None: obligatory arguments for the generation of the df only
            Tuple:
                First item: if condition must be found partially or must match entirely
                Second item: filtering value
        Returns filtered df by the conditions specified by the user
    """
    for condition_type, condition_values in conditions_dictionary.items():
        if condition_values is not None:
            condition_search, condition_value = condition_values[0], condition_values[1]
            if condition_search == "contains":
                df = df[df[condition_type].str.contains(condition_value, regex=False)]
            if condition_search == "not_contains":
                df = df[df[condition_type].str.contains(condition_value, regex=False)==False]
            if condition_search == "match":
                df = df[df[condition_type] == condition_value]
            if condition_search == "not_match":
                df = df[df[condition_type] != condition_value]
            if condition_search == "is_in":
                df = df[df[condition_type].isin(condition_value)]
    return df

def create_dictionary(df):
    """ Reduces the df to the minimum needed information: peptide sequence and uniprot entry
        Removes duplicates but keep one: same peptide can be repeated by another different MS study (different category)
        Convert the uniprot link https into uniprot code by getting last string after last "/"
        Generates a dictionary where keys are uniprot codes and values a list of non-repeated peptides for that uniprot code
    """
    exporting_df = df[["peptide_sequence", "uniprot_id"]]
    exporting_df = exporting_df.drop_duplicates(keep="first")
    exporting_df["uniprot_id"] = exporting_df["uniprot_id"].str.split("/").str[-1]
    data = exporting_df.groupby("uniprot_id")["peptide_sequence"].apply(list).to_dict()
    return data

def merge_peptide_data(dict1,dict2):

    print('---> Merging peptide data from IEDB and other databases...')
    d = dict(dict1, **dict2)
    return d

import argparse
import os
from predictor.database_functions import (
    peptide_extractor,
    uniprot_extractor,
    uniparc_extractor,
)
from predictor.core import (
    all_peptide_uniprot_locator,
    all_training_data_generator,
    cleavage_site_generator,
)
from predictor.ml_main import run_NN
from predictor.predictions import predict_csv
import time

HELP = " \
Command:\n \
----------\n \
Run: python3 NetCleave.py --ARG\
"
MAIN_PATH = __file__.split("NetCleave.py")[0]

# Append to path
import sys

sys.path.append(MAIN_PATH)


def parse_args():
    """
    Parse command-line arguments given by the user.
    """
    parser = argparse.ArgumentParser(
        description="Arguments necessary for NetCleave's execution."
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        help="Path to the training data.",
        action="store",
        default="None",
    )
    parser.add_argument(
        "--epitope_length",
        dest="epitope_length",
        help="Desired length of the epitopes generated from a FASTA file.",
        action="store",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--generate",
        dest="generate",
        help="Generate custom training data for later training the neural network.",
        action="store_true",
    )
    parser.add_argument(
        "--pred_input",
        dest="pred_input",
        help="""Type of data used as input for predicting C-terminal antigen processing:
                            1- FASTA file of a single protein, from which epitopes (8 to 11 residue-long) will be generated and scored.
                            2- CSV file with peptide sequences to score (column name: epitope) and the UniProt identifier of the protein where they come (column name: uniprot_id). NetCleave will retrieve the sequence of the whole protein, and create the 4+3 sequences that conform the cleavage site and are necessary for the scoring method.
                            3- CSV file with peptide sequences to score (column name: epitope), the identifier (column name: protein_id) and sequence of the protein where they come (column name: protein_seq). Same as type 2, those complete protein sequences will be used to create the 4+3 sequences.
                            """,
        action="store",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--mhc_allele",
        dest="mhc_allele",
        help="Major Histocompatibility Complex allele.",
        action="store",
        default="HLA",
    )
    parser.add_argument(
        "--mhc_class",
        dest="mhc_class",
        help="Major Histocompatibility Complex class. It can be either class I or II.",
        action="store",
        default="I",
    )
    parser.add_argument(
        "--mhc_options",
        dest="mhc_options",
        help="Prints characteristics of the available pre-trained models located at /data/models. It describes the mhc_allele, mhc_class and technique.",
        action="store_true",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        help="Path to the NetCleave retrained model.",
        action="store",
        default="None",
    )
    parser.add_argument(
        "--peptide_data",
        dest="peptide_data",
        help="Path to peptide data to use for generating data to later train the model.",
        action="store",
        default="./data/databases/iedb/mhc_ligand_full.csv",
    )
    parser.add_argument(
        "--peptide_data_additional",
        dest="peptide_data_additional",
        help="Path to additional peptide data to use (combined with IEDB data) for generating data to later train the model",
        action="store",
        default="./data/databases/other/HLA.csv",
    )
    parser.add_argument(
        "--predict",
        dest="predict",
        help="Predict the C-terminal cleavage of peptides. Indicate the input file using this flag. For example: python NetCleave.py --predict input/toy1.fasta --pred_input 1 --mhc_class II",
        action="store",
        default="None",
    )
    parser.add_argument(
        "--technique",
        dest="technique",
        help="Experimental technique used to obtain data.",
        action="store",
        default="mass spectrometry",
    )
    parser.add_argument(
        "--train",
        dest="train",
        help="""Train NetCleave's neural network model.
                            One of the main advantages of having a neural network model is that it can be retrained, so it is updated and adapt to your specific goal.
                            To do so, you can choose between (a) Generating your own custom data and then training the model with it; (b) Using pre-trained models during the training of the model.
                            """,
        action="store_true",
    )
    parser.add_argument(
        "--train_input",
        dest="train_input",
        help="""
                            Type of model training: (1) using a newer version of the Immune Epitope Database (IEDB),
                            (2) combining the IEDB with additional data from other sources and (3) using other data sources without taking into account the IEDB.
                            """,
        action="store",
        default=1,
        type=int,
    )

    return parser.parse_args()


def generating_data(
    uniprot_path,
    uniparc_path_headers,
    uniparc_path_sequence,
    train_input,
    iedb_path=None,
    conditions=None,
    other_path=None,
):
    """
    Generate training data that will be later used to retrain the neural network.
    Returns a dictionary with the selected peptides (key: C-terminal residue, value: peptide list)

    PARAMETERS
    ------------------------------------------------------------
    · uniprot_path: path to the UNIPROT data
    · uniparc_path_headers: path to the UNIPARC headers data
    · uniparc_path_sequence: path to the UNIPARC sequence data

    · conditions: dictionary with the conditions that the IEDB data must fulfill
    ------------------------------------------------------------
    """

    # Get data from files
    if train_input == 1:
        peptide_data = peptide_extractor.extract_peptide_data(
            iedb_path, conditions, iedb=True
        )
    elif train_input == 2:
        peptide_data1 = peptide_extractor.extract_peptide_data(
            iedb_path, conditions, iedb=True
        )
        peptide_data2 = peptide_extractor.extract_peptide_data(other_path, iedb=False)
        peptide_data = peptide_extractor.merge_peptide_data(
            peptide_data1, peptide_data2
        )
    elif train_input == 3:
        peptide_data = peptide_extractor.extract_peptide_data(other_path, iedb=False)

    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
    uniparc_data = uniparc_extractor.extract_uniparc_data(
        uniparc_path_headers, uniparc_path_sequence
    )
    sequence_data = all_peptide_uniprot_locator.join_data(uniprot_data, uniparc_data)

    # Generate dictionary
    selected_dictionary = all_peptide_uniprot_locator.locate_peptides(
        peptide_data, sequence_data
    )

    return selected_dictionary


def main(generate=False, train=False, predict=False):
    """
    This is the main function of the program.

    PARAMETERS
    ------------------------------------------------------------
    · generate: generate training data for the neural network
    · train: train the neural network with previously generated data
    · predict: predict a set of cleavage sites from a csv file
    ------------------------------------------------------------
    """

    if data_path != "None":
        training_data_path = data_path
    else:
        training_data_path = MAIN_PATH + "data/training_data/{}_{}_{}".format(
            mhc_class, technique.replace(" ", "-"), mhc_allele
        )
    if model_path != "None":
        models_export_path = model_path
    else:
        models_export_path = MAIN_PATH + "data/models/{}_{}_{}".format(
            mhc_class, technique.replace(" ", "-"), mhc_allele
        )

    if not any([generate, train, predict]):
        print(
            "Please, provide an argument. See python3 NetCleave.py -h for more information"
        )

    if generate:
        print("---> Generating training data, type {}...".format(train_input))
        uniprot_path = (
            MAIN_PATH + "data/databases/uniprot/uniprot_sprot.fasta"
        )  # download and decompress from https://www.uniprot.org/downloads REVIEWED fasta
        uniparc_path_headers = (
            MAIN_PATH
            + "data/databases/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.tab"
        )
        uniparc_path_sequence = (
            MAIN_PATH
            + "data/databases/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.fasta"
        )
        if train_input == 1:
            peptide_path = peptide_data
            iedb_conditions = {
                "Description": None,
                "Parent.Protein.IRI": None,
                "Method.Technique": ("contains", technique),
                "MHC.allele.class": ("match", mhc_class),
                "Allele.Name": ("contains", mhc_allele),
                #'Name': ('contains', 'Homo sapiens'),
                #'Parent Species': ('contains', 'Homo sapiens')
            }
            selected_dictionary = generating_data(
                uniprot_path,
                uniparc_path_headers,
                uniparc_path_sequence,
                train_input=1,
                iedb_path=peptide_path,
                conditions=iedb_conditions,
            )
        elif train_input == 2:
            peptide_path = peptide_data
            peptide_path2 = peptide_data_additional
            iedb_conditions = {
                "Description": None,
                "Parent.Protein.IRI": None,
                "Method.Technique": ("contains", technique),
                "MHC.allele.class": ("match", mhc_class),
                "Allele.Name": ("contains", mhc_allele),
                #'Name': ('contains', 'Homo sapiens'),
                #'Parent Species': ('contains', 'Homo sapiens')
            }
            selected_dictionary = generating_data(
                uniprot_path,
                uniparc_path_headers,
                uniparc_path_sequence,
                train_input=2,
                iedb_path=peptide_path,
                conditions=iedb_conditions,
                other_path=peptide_path2,
            )
        elif train_input == 3:
            peptide_path = peptide_data
            selected_dictionary = generating_data(
                uniprot_path,
                uniparc_path_headers,
                uniparc_path_sequence,
                train_input=3,
                other_path=peptide_path,
            )

        all_training_data_generator.prepare_cleavage_data(
            selected_dictionary, training_data_path
        )
        print("---> Training data available at: {}".format(training_data_path))
    if train:
        print("---> Training NetCleave with: {}".format(training_data_path))
        run_NN.create_models(training_data_path, models_export_path)
        print("---> Custom NetCleave model available at: {}".format(models_export_path))

    if predict != "None":
        if pred_input == 1:  # predict fasta file
            if epitope_length != 0:
                outfile = cleavage_site_generator.generateCleavageSites(
                    predict, custom_length=epitope_length
                )
                predict_csv.score_set(outfile, models_export_path, "ABC")
            else:
                outfile = cleavage_site_generator.generateCleavageSites(
                    predict, mhc=mhc_class
                )
                predict_csv.score_set(outfile, models_export_path, "ABC")

        if pred_input == 2:  # predict csv file with uniprot id
            uniprot_path = MAIN_PATH + "data/databases/uniprot/uniprot_sprot.fasta"
            uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
            outfile = cleavage_site_generator.generateCleavageSitesUniprot(
                predict, uniprot_data
            )
            predict_csv.score_set(outfile, models_export_path, "ABC", uniprot=True)

        if pred_input == 3:  # predict csv file with protein sequence
            outfile = cleavage_site_generator.generateCleavageSitesSequence(predict)
            predict_csv.score_set(outfile, models_export_path, "ABC", uniprot=True)


if __name__ == "__main__":
    time_initial = time.time()
    # Get arguments
    arguments = parse_args()
    epitope_length = arguments.epitope_length
    generate = arguments.generate
    pred_input = arguments.pred_input
    mhc_allele = arguments.mhc_allele
    mhc_class = arguments.mhc_class
    mhc_options = arguments.mhc_options
    model_path = arguments.model_path
    peptide_data = arguments.peptide_data
    peptide_data_additional = arguments.peptide_data_additional
    predict = arguments.predict
    technique = arguments.technique
    train = arguments.train
    data_path = arguments.data_path
    train_input = arguments.train_input

    if mhc_options:
        files = sorted(os.listdir(MAIN_PATH + "data/models"))
        print("\nAVAILABLE PRE-TRAINED MODELS:")
        for n, i in enumerate(files):
            elements = i.split("_")
            print(
                "\nModel {}\n ·mhc_allele:{}\n ·mhc_class:{}\n ·technique:{}\n".format(
                    n + 1, elements[2], elements[0], elements[1]
                )
            )
        exit()

    # Call main function
    main(generate, train, predict)
    time_final = time.time()
    time_dif = time_final - time_initial
    print("---> NetCleave's execution: {:.2f} s".format(time_dif))

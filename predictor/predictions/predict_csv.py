import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from tensorflow.keras.optimizers import SGD

MAIN_PATH = __file__.split('NetCleave.py')[0]

def score_set(data_path, model_path, name, uniprot=False):
    print('---> Prediction using model: {}'.format(model_path))
    peptide_length = 7
    model = load_model(model_path)
    df = read_data_table(data_path)
    descriptors_df = read_descriptors_table()
    encode_data = encode_sequence_data(df, descriptors_df)
    encoded_df = generate_encoded_df(encode_data, peptide_length, descriptors_df)
    prediction = model.predict(encoded_df)
    prediction_df = pd.DataFrame(prediction, columns=['prediction'])
    pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
    df['prediction'] = prediction_df['prediction']
    df.loc[df.cleavage_site.isna(), "prediction"] = np.nan
    #df['prediction'][df.cleavage_site.isna()] = np.nan

    if uniprot==False:
        df = df.set_index('epitope_id')

    if not os.path.exists(MAIN_PATH+'output/'):
        os.mkdir(MAIN_PATH+'output/')

    outfile = data_path.split('.')[0] + '_NetCleave.csv'
    df.to_csv(outfile, header=True)
    os.remove(data_path)
    print('---> Exporting predictions to: {}'.format(outfile))

    return df

def load_model(model_path):
    model_file_path = "{}/{}_model.h5".format(model_path, model_path.split("/")[-1])
    neurons = 336
    model = Sequential()
    model.add(Dense(int(neurons), input_dim=neurons, activation='tanh', kernel_initializer="glorot_normal"))
    model.add(Dense(int(neurons/3), activation='tanh', kernel_initializer="glorot_normal"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.01, momentum=0.00, nesterov=False, name='SGD')
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.load_weights(model_file_path)
    return model

def read_data_table(path):
    print("---> Reading cleavage sites ...")
    df = pd.read_csv(path)
    return df

def read_descriptors_table():
    print("---> Reading descriptors...")
    path = MAIN_PATH+"predictor/ml_main/QSAR_table.csv"
    df = pd.read_csv(path, sep=",", header=0, index_col=0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df

def encode_sequence_data(sequence_table, df):
    print("---> Encoding data using the descriptors...")
    encode_map, encode_data = {}, []
    for r in list("ACDEFGHIKLMNPQRSTVWY"):
        encode_map.setdefault(r, df.loc[r].tolist())

    for sequence in sequence_table['cleavage_site'].values:
        sequence_encode = []
        try:
            for r in sequence:
                sequence_encode.extend(encode_map[r])
            encode_data.append(sequence_encode)
        except:
            sequence = 'YYYYYYY'
            for r in sequence:
                sequence_encode.extend(encode_map[r])
            encode_data.append(sequence_encode)
    return encode_data

def generate_encoded_df(encode_data, peptide_length, df):
    print("---> Generating a descriptor dataframe...")
    descriptor_header = df.columns.tolist()
    encoded_df = pd.DataFrame(encode_data, columns=["{}_{}".format(i, j) for i in range(peptide_length) for j in descriptor_header])
    return encoded_df

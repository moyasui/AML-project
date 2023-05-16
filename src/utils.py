import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prep_data(dyn_sys, len_seq):
    """
    This function prepares the data for the RNNs
    """
    assert dyn_sys in ['lorenz', 'spiral'], "dyn_sys must be either 'lorenz' or 'spiral'"

    # Reading data
    dataset = pd.read_csv(f'csvs/{dyn_sys}.csv', header=0)
    # eliminate all white spaces from the column names
    dataset.columns = dataset.columns.str.replace(' ', '')
    # print(dataset.shape)

    coords = ['x', 'y', 'z'] if dyn_sys == 'lorenz' else ['x', 'y']
    spacial_dim = len(coords)
    skip = 15 if dyn_sys == 'spiral' else 1

    # Scale the entire DataFrame
    scaler = MinMaxScaler()

    dataset = dataset.iloc[::skip]
    # print(dataset.shape)
    scaled_data = scaler.fit_transform(dataset[coords])
    # print(scaled_data)

    # Convert the scaled data back to a DataFrame
    dataset[coords] = scaled_data
    # print(dataset.shape)
            
    # format
    input_len = len(dataset[dataset['particle']==0])-len_seq
    inputs = np.zeros((10,input_len, len_seq, spacial_dim)) # ILDC
    targets = np.zeros((10,input_len, spacial_dim))

    for i in range(10):
        particle_data = dataset[dataset['particle'] == i]
        
        particle_data = particle_data.drop(['t', 'particle'], axis=1)
        
        for j in range(len(particle_data)-len_seq):
            inputs[i][j] = particle_data.iloc[j:j+len_seq]
            targets[i][j] = particle_data.iloc[j+len_seq]            
   

    return dataset, inputs, targets


def train_test_split(inputs, targets, train_size, len_seq, spacial_dim):
    # combined all the points,,, again
    sequences_train_inputs, sequenced_test_inputs = inputs[:int(train_size*len(inputs))], inputs[int(train_size*len(inputs)):]
    sequenced_train_targets, sequenced_test_targets = targets[:int(train_size*len(inputs))], targets[int(train_size*len(inputs)):]

    train_inputs, test_inputs = sequences_train_inputs.reshape(-1, len_seq, spacial_dim), sequenced_test_inputs.reshape(-1, len_seq, spacial_dim)
    train_targets, test_targets = sequenced_train_targets.reshape(-1,spacial_dim), sequenced_test_targets.reshape(-1,spacial_dim)
    # train_inputs.shape, train_targets.shape
    # print(sequences_train_inputs), test_targets.shape
    train_test = [train_inputs, test_inputs, train_targets, test_targets]
    sequenced_train_test = [sequences_train_inputs, sequenced_test_inputs, sequenced_train_targets, sequenced_test_targets]

    return train_test, sequenced_train_test
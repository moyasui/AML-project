# build and run model

from model import *

# TODO: potential changes: rename X to t
def prep_data():

    # Reading data
    dataset = pd.read_csv(data_folder + 'lorenz.csv', header=0)
    # eliminate all white spaces from the column names
    dataset.columns = dataset.columns.str.replace(' ', '')

    groups = dataset.groupby('particle')
    print(groups)
    sequences = np.zeros(10, dtype=object) # a dictionary 

    # TODO: is this necessary
    # Iterate over the groups and store the split sequences in the dictionary
    for particle, group in groups:
        sequences[particle] = group.drop(["particle"], axis=1)
        # debug_print(sequences[particle].head())


    return sequences

def single_data(sequences, indx):
    X = sequences[indx]['t']
    Y = sequences[indx].drop(['t'], axis=1)  

    # TODO: does it matter if we do the training in order?

    train_size = int(len(sequences[0]) * 0.8)

    # TODO: Bootstrap?

    # scale the columns of the X and Y datasets
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.values.reshape(-1, 1))
    Y = scaler.fit_transform(Y)


    ## train test split  
    debug_print(train_size)
    y_train, y_test = Y[0:train_size,:], Y[train_size:len(Y),:]
    
    X,Y, y_train, y_test
    rnn_train_input, rnn_train_input_target = format_data(y_train, 2)
    
    # debug_print("Here")
    # debug_print(rnn_train_input.shape)
    # debug_print(rnn_train_input_target.shape)
    return Y, rnn_train_input, rnn_train_input_target


def run_single(rnn_train_input, rnn_train_input_target, batch_size=100, epochs=1000, n_hidden=8):

    '''
        get the data for a single sequence, then build and train a rnn with 1 layer LSTM, test using the same sequence, and visualisation
        input: batch_size (default = 100), epochs default = 1000, n_hidden default = 64 
        return: y_pred, the prediction from the rnn
            y, the actul value from the data
        '''
    
    
    # debug_print(rnn_input.shape)
    # debug_print(rnn_train.shape) 
    
    # TODO just roll by a time step?
    
    # build and train model
    model = build_rnn(rnn_train_input.shape[1:], n_hidden)
    hist, model = train_rnn(model, rnn_train_input, rnn_train_input_target, batch_size=batch_size, epochs=epochs, n_hidden=n_hidden)
    # TODO: what's hist

    return hist, model


# TODO: unfinished
def multi_data(sequences):

    train_size = 8 # sequences
    inputs =np.zeros(train_size, dtype=object)
    targets =np.zeros(train_size, dtype=object)
    for i in range(train_size):
        Y, inputs[i], targets[i] = single_data(sequences, i)

    # print(inputs.shape, targets.shape)
    return Y, inputs, targets

def run_multiple(rnn_train_input, rnn_train_input_target, batch_size=10, epochs=10, n_hidden=64):

    # print("this", rnn_train_input)
    
    model = build_rnn(rnn_train_input[0].shape[1:], n_hidden)
    for i in range(8):
        sth, model = train_rnn(model, rnn_train_input[i], rnn_train_input_target[i], batch_size=batch_size, epochs=epochs, n_hidden=n_hidden)
    
    return sth, model

sequences = prep_data()
Y, input, train = single_data(sequences, 0)
# hist, model = run_single(input, train, epochs=500, n_hidden=36)
# test_rnn(Y, model)

Y, input, train = multi_data(sequences)
sth, model = run_multiple(input, train, epochs=20, n_hidden=16)

test_rnn(Y, model)
# new_test(model, sequences)


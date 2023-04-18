import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import pickle
import time

'''
DEFINE GLOBAL FUNCTIONS
_______________________________________________________________________________________________________________________
'''

def save_params(params, directory):
    '''Used by server to save a params file which can be used by remote clients
    Args:
        params (dict): dictionary of parameters
        directory (str): name of directory where file will be saved
    '''
    id = params['id']
    signal_counts = [2, 3, 2, 1, 2, 2, 2, 1, 1, 4]  # number of signals used by each message ID
    num_signals = signal_counts[id-1]
    params['msg_id'] = 'id'+str(id)
    params['num_signals'] = num_signals
    params['input_dim'] = num_signals
    params['latent_dim'] = 16 * num_signals
    with open(directory + 'params.dict', 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
    return

def load_params(directory):
    '''Used by clients to load a params file as a dict
    Args:
        directory (str): name of directory where file will be loaded
    Returns a params dictionary
    '''
    with open(directory+'params.dict', 'rb') as f:
        params = pickle.load(f)
    return params

def import_data(csv_path, msg_id=None, start_time=0, end_time=None):  # imports SynCAN csv into dataframe
    '''Imports SynCAN data csv files into a time-indexed DataFrame
    Args:
        csv_path (str): filepath for the csv file
        msg_id (int): optional, used to specify a single CAN message ID
        start_time (int/float): specify starting time
        end_time (int/float): specify ending time
    Returns a pd.DataFrame containing SynCAN messages
    '''
    df = pd.read_csv(csv_path, header=None, skiprows=1, names=['Label',  'Time', 'ID',
                                                               'Signal1',  'Signal2',  'Signal3',  'Signal4'])
    df = pd.DataFrame(df.set_index(df.Time))
    df.index = df.index - df.index.min()      # set starting time to 0
    end_time = df.index.max() if not end_time else end_time
    df = df[((df.index >= start_time) & (df.index < end_time))]
    print(f'{len(df):,} total messages (id1,id2,...,id10)')
    df_labels = df.iloc[:,0:1].astype(int)
    if msg_id:
        df = df[df.ID==msg_id]
        df_labels = df.iloc[:,0:1].astype(int)
        df = df.dropna(axis=1, how='all')
        print(f'{len(df):,} messages used ({msg_id})')
        df = df.iloc[:,3:]
        df = df_labels.join(df)
    num_anomalous = len(df[df['Label']==1])
    print(f'{num_anomalous:,} anomalous messages out of {len(df):,}\n')
    return df

def clean_labels(df, msg_id, real_ranges):
    '''Removes labels from a test/evaluation DataFrame which are not anomalous for msg_id
    Args:
        df (pd.DataFrame): a DataFrame containing evaluation SynCAN data
        msg_id (int): CAN message id which labels should be cleaned
        real_ranges: a list of time ranges which should have labels kept
    Returns the input DataFrame with fewer '1' labels
    '''
    clean_df = df.copy()
    clean_df.Label = 0
    for start_time, end_time in real_ranges:
        clean_df.loc[start_time:end_time, 'Label'] = 1
    return clean_df

def find_ranges(predictions, index):
    '''Returns a list of ranges which contain a label of 1 (anomalous)
    Args:
        predictions (list-like): a continuous list of 0/1 values
        index (list-like): a continuous set of indices which should be used for the ranges
    Returns a list of (start, end) range tuples
    '''
    ranges = []
    i = 0
    while i < len(predictions):
        if predictions[i]:
            start = index[i]
            while True:
                i += 1
                if i >= len(predictions):
                    break
                if not predictions[i]:
                    break
            end = index[i-1]
            # if end - start > 100:
            ranges.append((start, end))
        i += 1
    return ranges

def visualize_data(df, start_time=0, end_time=None):
    '''Produces a plot to visualize SyCAN message signals over time
    Args:
        df (pd.DataFrame): DataFrame containing SynCAN values for a single message ID
        start_time (int/float): time which will be at the beginning of the plot
        end_time (int/float): time which will be at the end of the plot
    '''
    num_signals = df.shape[1]-1
    end_time = df.index.max() if not end_time else end_time
    data = df[((df.index >= start_time) & (df.index < end_time))]
    anomaly_ranges = find_ranges(data['Label'].to_numpy(), data.index)
    colors = ['blue', 'orange', 'green', 'red']
    fig, axes = plt.subplots(nrows=num_signals, ncols=1, figsize=(13, 2*num_signals), sharex=True)
    for i in range(num_signals):
        key = 'Signal'+str(i+1)
        c = colors[i % (len(colors))]
        t_data = data[key]
        ax = t_data.plot(
            ax = list(axes)[i],
            xlabel = 'Time (ms)',
            color = c,
            rot = 25)
        ax.legend([f'Signal {i}'], loc='upper left')
        for start, end in anomaly_ranges:
            ax.axvspan(start, end, color='grey', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return

def sequences_from_indices(data, indices_ds, start_index, end_index):
    '''A helper function used by the timeseries_dataset function
    Args:
        data (list-like): list of values for the sequences
        indicies_ds (tf.Dataset): dataset containing indices for the sequences
        start_index (int): starting index for the dataset
        end_index (int): ending index for the dataset
    Returns a tf.Dataset containing subsequences from the data
    '''
    dataset = tf.data.Dataset.from_tensors(data[start_index : end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        tf.autograph.experimental.do_not_convert(lambda steps, inds: tf.gather(steps, inds)),  # pylint: disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def timeseries_dataset(data, targets,
                        sequence_length,
                        sequence_stride=1,
                        data_is_target=False,
                        warm_up=0,
                        batch_size=1):
    '''Used to create a 'rolling-window' timeseries dataset
    Args:
        data (list-like): data which contains the values for the sequences
        sequence_length (int): number of values used in each subsequence
        data_is_target (bool): used to calculate reconstruction loss
        warm_up (int): number of values used before the output begins
        batch_size (int): batch size used by the tf.Dataset.batch() method
    Returns a tf.Dataset containing subsequence samples and any given targets
    '''
    index_dtype = 'int32'
    start_index = 0
    end_index = len(data)
    stop_idx = (end_index - start_index) - (sequence_length + warm_up)
    
    # Generate start positions
    start_positions = np.arange(0, stop_idx, sequence_stride, dtype=index_dtype)
    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    warm_up = tf.cast(warm_up, dtype=index_dtype)
    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat() # infinite times

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
            tf.autograph.experimental.do_not_convert(lambda i, positions: tf.range(positions[i], positions[i] + (sequence_length + warm_up))),
            num_parallel_calls=tf.data.AUTOTUNE)

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
                tf.autograph.experimental.do_not_convert(lambda i, positions: positions[i]),
                num_parallel_calls=tf.data.AUTOTUNE)
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index)
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    else:
        if data_is_target:
            if warm_up is None:
                dataset = tf.data.Dataset.zip((dataset, dataset))
            else:
                start_positions = np.arange(warm_up, stop_idx + warm_up, sequence_stride, dtype=index_dtype)
                positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()
                indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
                    tf.autograph.experimental.do_not_convert(lambda i, positions: tf.range(positions[i], positions[i] + sequence_length)),
                    num_parallel_calls=tf.data.AUTOTUNE)
                target_ds = sequences_from_indices(data, indices, start_index, end_index)
                dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset

def create_dataset(df, params, batch_size=None, verbose=False):
    '''Used to create a 'rolling-window' tf.Dataset for the SynCAN dataset
    Args:
        df (pd.DataFrame): DataFrame containing SynCAN messages
        params (dict): dictionary containing dataset parameters
        batch_size (int): batch size used for the dataset
        verbose (bool): verbose output
    Returns a (tf.Dataset, pd.DataFrame) tuple. The DataFrame is trimmed to the reconstruction size
    '''
    time_steps = params['time_steps']
    seq_stride = params['seq_stride']
    warm_up = params['warm_up']
    batch_size = batch_size if batch_size else params['batch_size']
    ds_df = df.drop(['Label'], axis=1, errors='ignore')    # remove labels if they exist
    ds = timeseries_dataset(ds_df.to_numpy(),
                            None,
                            time_steps,
                            sequence_stride=seq_stride,
                            data_is_target=True,
                            batch_size=params['batch_size'],
                            warm_up=warm_up)
    starting_indices = np.arange(warm_up, len(df) - time_steps, seq_stride)
    df = df.iloc[starting_indices[0]:starting_indices[-1]+time_steps]
    if verbose:
        num_sequences = len(starting_indices)
        print(f"{num_sequences:,} subsequences of length {time_steps}")
        print(f"{ds.__len__().numpy():,} batches (batch size {batch_size})")
    return ds, df

def get_train_val_test(df, params, verbose=False):
    '''Used split SynCAN training data into train, validation and test sets
    Args:
        df (pd.DataFrame): DataFrame containing SynCAN messages
        params (dict): dictionary containing dataset parameters
        verbose (bool): verbose output
    Returns a tf.Dataset dict and pd.Dataframe dict tuple
    '''
    data_length = len(df)
    train_size = int(data_length*params['train_split'])
    val_size = int(data_length*params['val_split'])
    test_size = data_length - train_size - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    if test_size > 0:
        test_df = df.iloc[-test_size:]
    time_steps = params['time_steps']
    seq_stride = params['seq_stride']
    warm_up = params['warm_up']
    if verbose:
        print('Train:')
    train_ds, train_df = create_dataset(train_df, params, verbose=verbose)
    if verbose:
        print(f'\nVal:')
    val_ds, val_df = create_dataset(val_df, params, verbose=verbose)
    if test_size > time_steps:
        if verbose:
            print(f'\nTest:')
        test_ds, test_df = create_dataset(test_df, params, verbose=verbose)
    if verbose:
        print()

    if test_size > time_steps:
        dataset_dict = {'train': train_ds, 'val': val_ds, 'test': test_ds}
        dataframe_dict = {'train': train_df, 'val': val_df, 'test': test_df}
    else:
        dataset_dict = {'train': train_ds, 'val': val_ds}
        dataframe_dict = {'train': train_df, 'val': val_df}
    return dataset_dict, dataframe_dict

def autoencoder(time_steps, warm_up, input_dim, latent_dim, drop_out=False, attention=False):
    '''Function used to create an INDRA-like recurrent autoencoder Keras model
    Args:
        time_steps (int): number of time steps used in each sample (subsequence length)
        warm_up (int): number of beginning time steps used in the input before any output is given
        input_dim (int): number of variables (signals) in each time step
        latent_dim (int): latent vector dimension (size of output from the encoder)
        drop_out (bool): use dropout layers in the encoder and decoder (drop_out=0.2)
        attention (bool): use attention layer in the decoder
    Returns an uncompiled tf.keras.Model object
    '''
    inputs = layers.Input(shape=(time_steps+warm_up, input_dim)) # shape = (time_steps, data_dimension/num_features)
    # encoder
    x = layers.Dense(latent_dim*2, activation='tanh')(inputs)
    if drop_out: x = layers.Dropout(0.2)(x)
    enc_out, enc_hidden = layers.GRU(latent_dim, return_sequences=True, return_state=True, activation="tanh")(x)
    if drop_out:
        enc_out = layers.Dropout(0.2)(enc_out)
        enc_hidden = layers.Dropout(0.2)(enc_hidden)
    # decoder
    dec_out, dec_hidden = layers.GRU(latent_dim, return_sequences=True, return_state=True, activation="tanh")(enc_out, initial_state=enc_hidden)
    if drop_out:
        dec_out = layers.Dropout(0.2)(dec_out)
        dec_hidden = layers.Dropout(0.2)(dec_hidden)
    if attention:
        dec_out = layers.Attention()([enc_out, dec_out])
    outputs = layers.Lambda(lambda x: x[:, warm_up:, :])(dec_out)   # used to remove the warm start time steps from the predictions
    outputs = layers.Dense(input_dim, activation='tanh')(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model

def create_model(params):
    '''Create an INDRA-like recurrent autoencoder using params
    Args:
        params (dict): dictionary containing model parameters
    Returns an uncompiled tf.keras.Model object
    '''
    model = autoencoder(
        params['time_steps'],
        params['warm_up'],
        params['input_dim'],
        params['latent_dim'],
        params['drop_out'],
        params['attention'])
    return model

def compile_model(model, params):
    '''Compile an INDRA-like Keras model using specified parameters
    Args:
        model (tf.keras.Model): Keras model used for compilation
        params (dict): dictionary containing model training parameters
    Returns a compiled version of the input model
    '''
    model.compile(optimizer=tf.optimizers.Adam(
        learning_rate=params['learning_rate']),
        loss=params['loss_function'],
        metrics=[params['metric']])
    return

def plot_loss(model):
    '''Plot the loss from a client
    '''
    # only works on FederatedLearning object
    loss_list = model.client_loss # list of lists for each clients' losses
    val_loss_list = model.val_loss_list # global loss list

    for lst in range(len(loss_list)):
      plt.plot([i for i in range(len(loss_list[0]))], loss_list[lst], label = 'Client ' + str(lst+1))
    plt.plot([i for i in range(len(loss_list[0]))], val_loss_list[1:], label = 'Global Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()
    return

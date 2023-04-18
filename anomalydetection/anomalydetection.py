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



'''
DEFINE CLASSES
_______________________________________________________________________________________________________________________
'''

class CentralizedModel:
    '''Class for creating and training a centralized (conventional) INDRA-like model
    '''
    def __init__(self, dataframe, params, file_name='model.h5', verbose=False):
        '''
        Args:
            dataframe (pd.DataFrame): SynCAN message DataFrame used for train, test and validation
            params (dict): dictionary containing model and training parameters
        '''
        self.datasets, self.dataframes = get_train_val_test(dataframe, params, verbose=verbose)
        self.params = params
        self.save_path = params['model_dir']+file_name
        self.verbose = verbose
        self.initialize_model()
        return

    def initialize_model(self):
        '''Initialize a Keras model using the create_model function
        '''
        if self.verbose:
            print(f"Saving model to {self.params['model_dir']}")
            print('Initializing centralized model...\n')
        self.model = create_model(self.params)
        compile_model(self.model, self.params)
        return
    
    def train_model(self, epochs=1, plot_loss=False, evaluate=False):
        '''Train the model using specified parameters
        Args:
            epochs (int): number of epochs used during training
            plot_loss (bool): plot the loss over epochs
            evaluate (bool): use the test set to evaluate loss
        '''
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.params['patience']),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.save_path,
                monitor="val_loss",
                save_best_only=True)]

        history = self.model.fit(self.datasets['train'],
                        epochs=epochs,
                        validation_data=self.datasets['val'],
                        use_multiprocessing=True,
                        workers=6,
                        shuffle=True,
                        callbacks=callbacks_list)
        if plot_loss and epochs > 1:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['training', 'validation'], loc='upper left')
            plt.show()
        if evaluate:
            if self.verbose:
                print('\nEvaluating...')
            loss, metric = self.model.evaluate(self.datasets['test'], verbose=0)
            print(f"\nTest loss: {loss:.5}\nTest {self.params['metric']}: {metric:.5}")
        return

#_______________________________________________________________________________________________________________________

class FederatedClient:
    '''Class for an individual client used for federated learning
    '''
    client_id = 0
    def __init__(self, dataframe, params, client_id=None, verbose=False):
        '''
        Args:
            dataframe (pd.DataFrame): DataFrame which contains SynCAN data for training
            params (dict): dictionary containing parameters used for federated learning models
            client_id (int): used to manually specify a unqiue client_id
            verbose (bool): verbose output
        '''
        self.dataset, self.dataframe = create_dataset(dataframe, params, verbose=verbose)
        self.params = params
        if client_id:
            self.client_id = client_id
        else:
            self.client_id = FederatedClient.client_id
            FederatedClient.client_id += 1
        self.verbose = verbose
        self.iteration = 1
        return
    
    def initialize_model(self):
        '''Initialize an INDRA-like model using the given params
        '''
        self.model = create_model(self.params)
        self.load_global_model()
        compile_model(self.model, self.params)
        return
    
    def load_global_model(self):
        '''Load global model from model_dir and set client model weights
        '''
        file_path = self.params['model_dir']+'global_model_'+str(self.iteration-1)+'.h5'
        interval = 2
        while not os.path.exists(file_path):
            if self.verbose:
                print(f'\rWaiting for global model {self.iteration-1}...', end='')
            time.sleep(interval)
        global_model = tf.keras.models.load_model(file_path)
        self.model.set_weights(global_model.get_weights())
        return
    
    def train_model(self):
        '''Train the client model for one iteration of federated learning
        '''
        if self.verbose:
            print('Training model...')
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.params['patience']),
            ]
        verbose = 'auto' if self.verbose else 0
        history = self.model.fit(self.dataset,
            epochs=self.params['epochs_per_round'],
            use_multiprocessing=True,
            workers=6,
            shuffle=True,
            callbacks=callbacks_list,
            verbose=verbose)
        loss = history.history['loss']
        # loss = 0
        return loss
    
    def iterate(self):
        '''Execute one iteration (round) of federated learning
        '''
        loss = self.train_model()
        save_path = self.params['model_dir']+'client'+str(self.client_id)+'_model_'+str(self.iteration)+'.h5'
        tf.keras.models.save_model(self.model, save_path)
        self.iteration += 1
        return loss

    def run_client(self):
        '''Run remote client over num_iterations (only needed for remote use, not local)
        '''
        if self.verbose:
            print(f'Starting Client {self.client_id}...')
        self.initialize_model()
        num_iterations = self.params['num_iterations']
        for i in range(num_iterations):
            print(f'\rIteration {self.iteration}/{num_iterations}', end='')
            if self.verbose:
                print()
            self.iterate()
            if self.verbose:
                print(f'Loading global model {self.iteration-1}...')
            self.load_global_model()
        return

#_______________________________________________________________________________________________________________________

class FederatedAggregator:
    '''This class is used to perform aggregation on saved clients models which may be training on a separate runtime
    '''
    def __init__(self, params, verbose=False):
        '''
        Args:
            params (dict): dictionary containing parameters for federated learning models
            verbose (bool): verbose output
        '''
        self.params = params
        self.verbose = verbose
        if verbose:
            print(f"Using model directory {self.params['model_dir']}")
        self.iteration = 1
        return
    
    def initialize_model(self):
        '''Initialze a global model using the given model parameters
        '''
        if self.verbose:
            print('Initializing global model...')
        self.global_model = create_model(self.params)
        compile_model(self.global_model, self.params)
        save_path = self.params['model_dir']+'global_model_0.h5'
        tf.keras.models.save_model(self.global_model, save_path)
        return
    
    def load_client_models(self):
        '''Load client model files from model_dir and remake client_models list
        '''
        interval = 2 #seconds
        num_clients = self.params['num_clients']
        self.client_models = []
        if self.verbose:
                print(f'Loading client models...')
        for i in range(num_clients):
            file_path = self.params['model_dir']+'client'+str(i)+'_model_'+str(self.iteration)+'.h5'
            while not os.path.exists(file_path):
                if self.verbose:
                    print(f'\rWaiting for client {i}...', end='')
                time.sleep(interval)
            if self.verbose:
                print()
            client_model = tf.keras.models.load_model(file_path)
            self.client_models.append(client_model)
        return
    
    def aggregate_client_models(self):
        '''Average all weights from the client models and set global model
        '''
        if self.verbose:
            print('Aggregating client models...')
        global_weights = self.global_model.get_weights()    # used only for reshaping the client weights
        num_layers = len(global_weights)
        clients_weights = []
        for client_model in self.client_models:
            weights = client_model.get_weights()
            clients_weights.append(weights)
        new_weights = []
        for i in range(num_layers):
            average = np.mean([w[i] for w in clients_weights], axis=0)
            new_weights.append(average.reshape(global_weights[i].shape))
        self.global_model.set_weights(new_weights)
        return
    
    def iterate(self):
        '''Execute one iteration (round) of federated learning
        '''
        self.load_client_models()
        self.aggregate_client_models()
        save_path = self.params['model_dir']+'global_model_'+str(self.iteration)+'.h5'
        tf.keras.models.save_model(self.global_model, save_path)
        self.iteration += 1
        return

#_______________________________________________________________________________________________________________________

class FederatedLearning:
    '''Class used for simulating federated learning on one or multiple runtimes
    '''
    def __init__(self, params, dataframe=None, verbose=False):
        '''
        Args:
            params (dict): dictionary of parameters used for federated learning models
            dataframe (pd.DataFrame): DataFrame used for train, val and test
            verbose (bool): verbose output
        '''
        self.params = params
        self.verbose = verbose
        self.val_loss_list = [float('inf')]
        self.iteration = 0
        self.aggregator = FederatedAggregator(params, verbose=verbose)
        if dataframe is not None:
            self.ds_dict, self.df_dict = get_train_val_test(dataframe, params, verbose=verbose)
        else:
            self.ds_dict = None
            self.df_dict = None
        self.client_loss = []
        self.clients = None
        return

    def initialize_clients(self, data_split=None):
        '''Initialize a set of locally created clients which run on the same runtime as the aggregator
        '''
        if self.df_dict is None:
            print("No training data given! Please set object parameter 'dataframe'")
            return
        train_df = self.df_dict['train']
        num_clients = self.params['num_clients']
        if data_split:
            if num_clients != len(data_split):
                print('Initialization error: Length of data split does not match number of clients!')
                return
        else:
            data_split = [1/num_clients for _ in range(num_clients)]
        self.clients = []
        if self.verbose:
            print('Initializing clients and generating models...')
        start = 0
        for i in range(num_clients):
            num_samples = int(len(train_df) * data_split[i])
            end = start + num_samples
            df = train_df.iloc[start:end]
            if self.verbose:
                print(f'Client {i}:')
            new_client = FederatedClient(
                df,
                self.params,
                client_id=i,
                verbose=self.verbose)
            new_client.initialize_model()
            self.clients.append(new_client)
            if self.verbose:
                print()
            start = end
            self.client_loss.append([])
        return
    
    def validate_global_model(self):
        '''Use validation set to validate global model before being sent to the clients
        '''
        if self.ds_dict is None:
            print("No validation data given! Please set object parameter 'dataframe'")
            return
        print('\rValidating global model...', end='')
        if self.verbose:
            print()
        verbose = 'auto' if self.verbose else 0
        loss, metric = self.aggregator.global_model.evaluate(self.ds_dict['val'], verbose=verbose)
        if self.verbose:
            print(f"Validation loss: {loss:.5}\nValidation {self.params['metric']}: {metric:.5}")
        self.val_loss_list.append(loss)
        return

    def iterate(self, validate=True):
        '''Execute one iteration (round) of federated learning
        Args:
            validate (bool): perform validation at the end of the iteration (round)
        '''
        if self.clients is not None:
            for j, client in enumerate(self.clients):
                print(f"\rIteration {self.iteration+1}/{self.params['num_iterations']} - Client {j+1}/{len(self.clients)}", end='')
                if self.verbose:
                    print()
                loss = client.iterate()
                self.client_loss[j].append(loss)
        else:
            print(f"\rIteration {self.iteration+1}/{self.params['num_iterations']}", end='')
        if self.verbose:
            print()
        self.aggregator.iterate()

        if validate:
            min_loss = np.min(self.val_loss_list)
            self.validate_global_model()
            if self.val_loss_list[-1] > min_loss:
                return True
        if self.clients is not None:
            for client in self.clients:
                client.load_global_model()
        self.iteration += 1
        return False

    def test_global_model(self):
        '''Use test set to test global model loss after all iterations are completed
        '''
        if self.ds_dict is None:
            print("No testing data given! Please set object parameter 'dataframe'")
            return
        print('\rTesting global model...')
        verbose = 'auto' if self.verbose else 0
        loss, metric = self.aggregator.global_model.evaluate(self.ds_dict['test'], verbose=verbose)
        print(f"Test loss: {loss:.5}\nTest {self.params['metric']}: {metric:.5}")
        return
    
    def run_server(self, validation=False, test=False):
        '''Run a sever which aggregates client models and distributes global model
        Args:
            validation (bool): perform validation at the end of each iteration (round)
            test (bool): perform testing after all iterations have completed
        '''
        if self.verbose:
            print('Starting Server...')
        self.aggregator.initialize_model()
        num_iterations = self.params['num_iterations']
        for i in range(num_iterations):
            if self.iterate(validation):    # early stop if validation loss increases
                break
        save_path = self.params['model_dir']+'global_model_final.h5'
        tf.keras.models.save_model(self.aggregator.global_model, save_path)
        if test:
            self.test_global_model()
        return
    
    def run_federated_learning(self, data_split=None, validation=False, test=False):
        '''Start a federated learning simulation
        Args:
            data_split (tuple): relative proportions of data given to each client (must add to 1.0)
            validation (bool): perform validation at the end of each iteration (round)
            test (bool): perform testing after all iterations have completed
        '''
        self.initialize_clients(data_split=data_split)
        self.run_server(validation, test)
        return

#_______________________________________________________________________________________________________________________

class SynCAN_Evaluator:
    '''Class used perform anomaly detection evaluation for INDRA-like models
    '''
    def __init__(self, thresh_df, params, verbose=False):
        '''
        Args:
            thresh_df (pd.DataFrame): DataFrame containing normal SynCAN messages, used for calculating thresholds/baseline
            params (dict): dictionary containing parameters for the models and evalutation
            verbose (bool): verbose output
        '''
        if verbose:
            print('Creating threshold dataset...')
        self.thresh_ds, self.thresh_df = create_dataset(thresh_df, params, verbose=verbose)
        self.params = params
        self.verbose = verbose
        self.batch_results = None
        return

    def reconstruct(self, ds, ret_subseqs=False):
        '''Used for reconstructing a tf.Dataset of subsequences into a single reconstruction DataFrame
        Args:
            ds (tf.Dataset): a dataset which contains SynCAN subsequences which will be reconstructed
            ret_subseqs (bool): return the reconstructed subsequences (np.array)
        Returns a pd.DataFrame which should be the same length as the returned df by create_dataset()
        '''
        # used for reconstructing signals with a saved model into a continuous dataframe
        predictions = self.model.predict(ds, verbose=self.verbose, workers=-1, use_multiprocessing=True)
        time_steps = self.params['time_steps']
        seq_stride = self.params['seq_stride']
        if time_steps == seq_stride:
            reconstruction = predictions.reshape(-1, predictions.shape[-1])
        else:
            # remove duplicate timesteps from predictions
            first = predictions[0]
            rest = predictions[1:, time_steps-seq_stride:]
            rest = rest.reshape(-1, rest.shape[-1])
            reconstruction = np.concatenate((first, rest))
        columns = ['Signal'+str(i+1) for i in range(reconstruction.shape[-1])]
        if ret_subseqs:
            return pd.DataFrame(reconstruction, columns=columns), predictions
        else:
            return pd.DataFrame(reconstruction, columns=columns)

    def reconstruct_threshold_data(self):
        '''Used to reconstructed the threshold selection dataset for threshold calculations
        '''
        real_values = self.thresh_df.to_numpy()[:,1:]
        if self.verbose:
            print('Reconstructing threshold-selection data...')
        reconstruction = self.reconstruct(self.thresh_ds)
        reconstructed_values = reconstruction.to_numpy()
        self.thresh_se = np.square(real_values - reconstructed_values)
        return
    
    def set_thresholds(self, num_stds, plot=False):
        '''Set a threshold for each message signal
        Args:
            num_stds (int/float): number of standard deviations from the mean used for thresholds
            plot (bool): create a plot showing the thresholds relative to the threshold reconstruction errors
        '''
        self.thresholds = self.thresh_se.mean(axis=0) + (num_stds * self.thresh_se.std(axis=0))
        # self.thresholds = np.max(thresh_se, axis=0)
        if self.verbose:
            print('Setting squared-error thresholds...')
            for i, t in enumerate(self.thresholds):
                print(f'Signal {str(i+1)}: {t:.5}')
        if plot:
            self.plot_error_thresholds(self.thresh_se)
        return
    
    def plot_error_thresholds(self, squared_error):
        '''Function used for showing the thresholds relative to the threshold reconstruction errors
        Args:
            squared_error (np.array): squared error values calculated from the threshold reconstruction
        '''
        num_signals = params['num_signals']
        fig, axes = plt.subplots(nrows=num_signals, ncols=1, figsize=(13, 2*num_signals))
        for i in range(num_signals): # Plot histograms of squared error values and mean + threshold lines
            ax = list(axes)[i]
            se = squared_error[:,i]
            sns.set(font_scale = 1)
            sns.set_style("white")
            ax.xlim([0, 2*self.thresholds[i]])
            sns.histplot(np.clip(se, 0, 2 * self.thresholds[i]), bins=50, kde=True, color='grey', ax=ax)
            ax.axvline(x=np.mean(se), color='g', linestyle='--', linewidth=3)
            ax.text(np.mean(se), 250, "Mean", horizontalalignment='left', 
                    size='small', color='black', weight='semibold')
            ax.axvline(x=self.thresholds[i], color='b', linestyle='--', linewidth=3)
            ax.text(self.thresholds[i], 250, "Threshold", horizontalalignment='left', 
                    size='small', color='Blue', weight='semibold')
            ax.xlabel('Squared Error')
            ax.title('Signal '+str(i+1))
            sns.despine()
        plt.tight_layout()
        plt.show()
        return

    def set_message_predictions(self, predictions):
        '''Create message-level predictions from windows predictions and set the evaluation squared error values
        '''
        stride = self.params['seq_stride']
        steps = self.params['time_steps']
        self.predictions = np.zeros(len(self.evaluation_df), dtype=int)
        for i, prediction in zip(range(0, len(self.evaluation_df)-steps+1, stride), predictions):
            if prediction == 1:
                self.predictions[i:i+steps] = 1
        real_values = self.evaluation_df.to_numpy()[:,1:]
        reconstructed_values = self.reconstructed_df.to_numpy()
        self.eval_se = np.square(real_values - reconstructed_values)
        return
    
    def create_window_labels(self, message_labels):
        '''Create one label for each evaluation subsequence
        Args:
            message_labels (np.array): array of message specific labels
        Returns an np.array containing one label for each input subsequence
        '''
        if self.verbose:
            print('Labeling reconstructed subsequences...')
        labels = []
        stride = self.params['seq_stride']
        steps = self.params['time_steps']
        for i in range(0, len(message_labels)-steps+1, stride):
            window = message_labels.iloc[i:i+steps]
            if len(window[window==1]) > 0:
                labels.append(1)
            else:
                labels.append(0)
        return np.array(labels)
    
    def create_window_predictions(self, reconstructions):
        '''Create one label for each reconstructed subsequence
        Args:
            reconstructions (np.array): array of reconstructed subsequences
        Returns an np.array containing one label for each reconstructed subsequence
        '''
        if self.verbose:
            print('Creating window predictions...')
        time_steps = self.params['time_steps']
        predictions = []
        indices = range(0, len(self.evaluation_df)-time_steps+1, self.params['seq_stride'])
        values = self.evaluation_df.to_numpy()
        predictions = np.zeros(len(indices), dtype=int)
        for i, (j, reconstruction) in enumerate(zip(indices, reconstructions)):
            real_values = values[j:j+time_steps,1:]
            se = np.square(real_values - reconstruction)
            pred = 1*(se > self.thresholds)
            predictions[i] = 1 if np.sum(pred) > 0 else 0
        return np.array(predictions)

    def get_results(self, reconstructions):
        '''Calculate metrics using the window labels and window predictions (assumes evaluate() was just called)
        Args:
            reconstructions (np.array): array of reconstructed subsequences
        Returns a dictionary with each metric score
        '''
        y = self.window_labels
        y_hat = self.create_window_predictions(reconstructions)
        if self.verbose:
            print(f'Percentage of anomalous predictions: {np.mean(y_hat==1)*100:.3f}%')
        self.set_message_predictions(y_hat)
        tn, fp, fn, tp = np.sum((y == 0) & (y_hat == 0)), np.sum((y == 0) & (y_hat == 1)), np.sum((y == 1) & (y_hat == 0)), np.sum((y == 1) & (y_hat == 1))
        fp_rate = fp / (fp + tn)
        tp_rate = tp / (tp + fn)
        accuracy = np.mean(np.array(y_hat)==np.array(y))
        bal_accuracy = metrics.balanced_accuracy_score(y, y_hat)
        f1_score = metrics.f1_score(y, y_hat)
        precision = metrics.precision_score(y, y_hat)
        recall = metrics.recall_score(y, y_hat)
        if self.verbose:
            print(f'Accuracy: {accuracy:.5}')
            print(f'Balanced Accuracy: {bal_accuracy:.5}')
            print(f'F1 Score: {f1_score:.5}')
            print(f'Precision Score: {precision:.5}')
            print(f'Recall Score: {recall:.5}')
        return {'False Positive Rate': fp_rate,
                'True Positive Rate': tp_rate,
                'Accuracy': accuracy,
                'Balanced Accuracy': bal_accuracy,
                'F1 Score': f1_score,
                'Precision': precision,
                'Recall': recall}
    
    def evaluate(self, model, eval_df, thresh_stds):
        '''Evaluate the given model using the given evalutaion data
        Args:
            model (tf.keras.Model): a trained INDRA-like Keras model
            eval_df (pd.DataFrame): dataframe containing values from a SynCAN attack test
            thresh_stds (list-like): a list of std values used to evaluate the model
        Returns a pd.DataFrame containing the metric results for each threshold value
        '''
        self.model = model
        self.reconstruct_threshold_data()
        evaluation_ds, self.evaluation_df = create_dataset(eval_df, self.params, verbose=self.verbose)
        if self.verbose:
            print(f'Reconstructing evaluation data...')
        self.reconstructed_df, reconstructions = self.reconstruct(evaluation_ds, ret_subseqs=True)
        self.reconstructed_df.set_index(self.evaluation_df.index, inplace=True)
        message_labels = self.evaluation_df['Label']
        self.window_labels = self.create_window_labels(message_labels)
        if self.verbose:
            print(f'Percentage of anomalous windows: {np.mean(self.window_labels==1)*100:.3f}%')
        # if self.verbose:
        #     print(f'Percentage of anomalous messages: {np.mean(self.evaluation_df.Label==1)*100:.3f}%')
        results = []
        for ts in list(thresh_stds):
            if self.verbose:
                print(f'\nThresholds set to {ts} standard deviations')
            self.set_thresholds(ts, plot=False)
            results.append(self.get_results(reconstructions))
        self.results_df = pd.DataFrame(results).set_index(thresh_stds)
        return self.results_df

    # def set_message_predictions(self):
    #     '''Create message-level predictions and set the evaluation squared error values
    #     '''
    #     real_values = self.evaluation_df.to_numpy()[:,1:]
    #     reconstructed_values = self.reconstructed_df.to_numpy()
    #     self.eval_se = np.square(real_values - reconstructed_values)
    #     self.predictions = 1*(self.eval_se > self.thresholds)
    #     return
    
    # def create_window_predictions(self, reconstructions):
    #     '''Create one label for each reconstructed subsequence
    #     Args:
    #         reconstructions (np.array): array of reconstructed subsequences
    #     Returns an np.array containing one label for each reconstructed subsequence
    #     '''
    #     if self.verbose:
    #         print('Creating window predictions...')
    #     time_steps = self.params['time_steps']
    #     predictions = []
    #     indices = range(0, len(self.evaluation_df)-time_steps+1, self.params['seq_stride'])
    #     values = self.evaluation_df.to_numpy()
    #     predictions = np.zeros(len(indices), dtype=int)
    #     for i, (j, reconstruction) in enumerate(zip(indices, reconstructions)):
    #         real_values = values[j:j+time_steps,1:]
    #         se = np.square(real_values - reconstruction)
    #         pred = 1*(se > self.thresholds)
    #         predictions[i] = 1 if np.sum(pred) > 0 else 0
    #     return np.array(predictions)

    # def get_results(self, reconstructions):
    #     '''Calculate metrics using the window labels and window predictions (assumes evaluate() was just called)
    #     Args:
    #         reconstructions (np.array): array of reconstructed subsequences
    #     Returns a dictionary with each metric score
    #     '''
    #     self.set_message_predictions()
    #     y = self.evaluation_df['Label']
    #     y_hat = self.predictions
    #     if self.verbose:
    #         print(f'Percentage of anomalous predictions: {np.mean(y_hat==1)*100:.3f}%')
    #     tn, fp, fn, tp = np.sum((y == 0) & (y_hat == 0)), np.sum((y == 0) & (y_hat == 1)), np.sum((y == 1) & (y_hat == 0)), np.sum((y == 1) & (y_hat == 1))
    #     fp_rate = fp / (fp + tn)
    #     tp_rate = tp / (tp + fn)
    #     accuracy = np.mean(np.array(y_hat)==np.array(y))
    #     bal_accuracy = metrics.balanced_accuracy_score(y, y_hat)
    #     f1_score = metrics.f1_score(y, y_hat)
    #     precision = metrics.precision_score(y, y_hat)
    #     recall = metrics.recall_score(y, y_hat)
    #     if self.verbose:
    #         print(f'Accuracy: {accuracy:.5}')
    #         print(f'Balanced Accuracy: {bal_accuracy:.5}')
    #         print(f'F1 Score: {f1_score:.5}')
    #         print(f'Precision Score: {precision:.5}')
    #         print(f'Recall Score: {recall:.5}')
    #     return {'False Positive Rate': fp_rate,
    #             'True Positive Rate': tp_rate,
    #             'Accuracy': accuracy,
    #             'Balanced Accuracy': bal_accuracy,
    #             'F1 Score': f1_score,
    #             'Precision': precision,
    #             'Recall': recall}

    # def evaluate(self, model, eval_df, thresh_stds):
    #     '''Evaluate the given model using the given evalutaion data
    #     Args:
    #         model (tf.keras.Model): a trained INDRA-like Keras model
    #         eval_df (pd.DataFrame): dataframe containing values from a SynCAN attack test
    #         thresh_stds (list-like): a list of std values used to evaluate the model
    #     Returns a pd.DataFrame containing the metric results for each threshold value
    #     '''
    #     self.model = model
    #     self.reconstruct_threshold_data()
    #     evaluation_ds, self.evaluation_df = create_dataset(eval_df, self.params, verbose=self.verbose)
    #     if self.verbose:
    #         print(f'Reconstructing evaluation data...')
    #     self.reconstructed_df, reconstructions = self.reconstruct(evaluation_ds, ret_subseqs=True)
    #     self.reconstructed_df.set_index(self.evaluation_df.index, inplace=True)
    #     if self.verbose:
    #         print(f'Percentage of anomalous messages: {np.mean(self.evaluation_df.Label==1)*100:.3f}%')
    #     results = []
    #     for ts in list(thresh_stds):
    #         if self.verbose:
    #             print(f'\nThresholds set to {ts} standard deviations')
    #         self.set_thresholds(ts, plot=False)
    #         results.append(self.get_results(reconstructions))
    #     self.results_df = pd.DataFrame(results).set_index(thresh_stds)
    #     return self.results_df
    
    def batch_evaluate(self, models, model_names, eval_df, thresh_stds):
        '''Evaluate a list of models using the given evaluation data and store results
        Args:
            models (list(tf.keras.Model)): list of trained INDRA-like Keras models
            eval_df (pd.DataFrame): dataframe containing values from a SynCAN attack test
            thresh_stds (list-like): a list of std values used to evaluate the models
        '''
        self.model_names = model_names
        self.batch_results = []
        for model in models:
            self.batch_results.append(self.evaluate(model, eval_df, thresh_stds))
        return
    
    def plot_ROC(self, title=None, filename=None):
        '''Plot the receiver operating characteristic curve (assumes evaluate() or batch_evaluate() was recently called)
        Args:
            title (str): optional title to use for the plot
            filename (str): specify a filename to save a .png file
        '''
        if self.batch_results:
            for results_df, name in zip(self.batch_results, self.model_names):
                fp_rate = results_df['False Positive Rate']
                tp_rate = results_df['True Positive Rate']
                label = f'{name}, AUC: {metrics.auc(fp_rate, tp_rate):.3f}'
                plt.plot(fp_rate, tp_rate, label=label)
            plt.legend()
        else:
            fp_rate = self.results_df['False Positive Rate']
            tp_rate = self.results_df['True Positive Rate']
            label = f'AUC: {metrics.auc(fp_rate, tp_rate):.3f}'
            plt.text(0.9, 0.1, label)
            plt.plot(fp_rate, tp_rate)
        if title:
            plt.title(title)
        else:
            plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if filename:
            plt.savefig(filename)
        plt.show()
        return
    
    def plot_PR(self, title=None, filename=None):
        '''Plot the precision-recall curve (assumes evaluate() or batch_evaluate() was recently called)
        Args:
            title (str): optional title to use for the plot
            filename (str): specify a filename to save a .png file
        '''
        if self.batch_results:
            for results_df, name in zip(self.batch_results, self.model_names):
                precision = results_df['Precision']
                recall = results_df['Recall']
                label = f'{name}, AUC: {metrics.auc(recall, precision):.3f}'
                plt.plot(recall, precision, label=label)
            plt.legend()
        else:
            precision = self.results_df['Precision']
            recall = self.results_df['Recall']
            plt.plot(recall, precision)
            label = f'AUC: {metrics.auc(recall, precision):.3f}'
            plt.text(0.1, 0.1, label)
        if title:
            plt.title(title)
        else:
            plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if filename:
            plt.savefig(filename)
        plt.show()
        return
    
    def visualize_reconstruction(self, start_time=0, end_time=None, highlight_anomalies=False, highlight_predictions=False, plot_squared_error=False):
        '''Visualize the current evaluation data and reconstruction
        Args:
            start_time (int/float): specify a starting time used for the plot
            end_time (int/float): specify a end time used for the plot
            highlight_anomalies (bool): highlight the actual anomalous ranges
            highlight_predictions (bool): highlight the predicted anomalous ranges
            plot_squared_error (bool): plot the calculated squared error for each signal
        '''
        # accepts two dataframes of the same length with the same number of signals - keys must be Signal1, Signal2,...
        if self.verbose:
            print('Plotting reconstruction...')
        in_df = self.evaluation_df
        out_df = self.reconstructed_df
        end_time = in_df.index.max() if not end_time else end_time
        data = in_df[((in_df.index >= start_time) & (in_df.index < end_time))]
        reconstructed_data = out_df[((out_df.index >= start_time) & (out_df.index < end_time))]

        num_signals = reconstructed_data.shape[1]
        labels = data.Label
        real_ranges = find_ranges(labels.to_numpy(), data.index)
        pred_ranges = find_ranges(self.predictions, in_df.index)
        msg_id = self.params['msg_id']
        fig, axes = plt.subplots(nrows=num_signals, ncols=1, figsize=(13, 3*num_signals), sharex=True)
        for i in range(num_signals):
            key = 'Signal'+str(i+1)
            t_data = data[key]
            t_reconstructed_data = reconstructed_data[key]
            ax = list(axes)[i]
            ax0 = t_data.plot(ax=ax, color="black", title=msg_id.upper()+'_'+key, rot=25)
            ax1 = t_reconstructed_data.plot(ax=ax, color="red", rot=10)
            ax1.legend(['Original Signal', 'Reconstructed Signal'], loc='upper left')
            if highlight_anomalies:
                for start, end in real_ranges:
                    ax0.axvspan(start, end, color='grey', alpha=0.3)
            if highlight_predictions:
                for start, end in pred_ranges:
                    if end > data.index.min() and start < data.index.max():
                        if start < data.index.min():
                            start = data.index.min()
                        if end > data.index.max():
                            end = data.index.max()
                        ax1.axvspan(start, end, color='red', alpha=0.3)
            if plot_squared_error: # plot squared error
                ax2 = ax0.twinx()
                se = pd.DataFrame(self.eval_se[:,i], index=in_df.index)
                se = se[((in_df.index >= start_time) & (in_df.index < end_time))]
                se.plot(ax=ax2, alpha=0.5)
                ax2.set_ylim([0,self.thresholds[i]*1.5])
                ax2.axhline(self.thresholds[i], linestyle='--', c='red', alpha=0.5)
                ax2.set_ylabel('Squared Error')
                ax2.legend(['Squared Error'], loc='lower left')
        plt.tight_layout()
        plt.show()
        return      


import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt

def import_train_data(csv_path, msg_id, start_time=0, end_time=None):    # imports unlabeled training data into dataframe
    df = pd.read_csv(csv_path)
    df = df.set_index(df.Time)
    df.index = df.index - df.index.min()      # set starting time to 0
    end_time = df.index.max() if not end_time else end_time
    df = df[((df.index >= start_time) & (df.index < end_time))]
    print(f'{len(df):,} total messages (id1,id2,...,id10)')
    id_df = df[df.ID==msg_id]
    id_df = id_df.dropna(axis=1, how='all')
    print(f'{len(id_df):,} messages used ({msg_id})')
    return id_df.iloc[:,3:]

def import_eval_data(csv_path, msg_id, start_time=0, end_time=None):  # imports evaluation (abnormal) data into dataframe
    df = pd.read_csv(csv_path, header=None, skiprows=1, names=['Label',  'Time', 'ID',
                                                               'Signal1',  'Signal2',  'Signal3',  'Signal4'])
    df = pd.DataFrame(df.set_index(df.Time))
    df.index = df.index - df.index.min()      # set starting time to 0
    end_time = df.index.max() if not end_time else end_time
    df = df[((df.index >= start_time) & (df.index < end_time))]
    print(f'{len(df):,} total messages (id1,id2,...,id10)')
    id_df = df[df.ID==msg_id]
    id_df = id_df.dropna(axis=1, how='all')
    print(f'{len(id_df):,} messages used ({msg_id})')
    id_df_labels = id_df.iloc[:,0:1].astype(int)
    id_df = id_df.iloc[:,3:]
    num_anomalous = len(id_df_labels[id_df_labels['Label']==1])
    print(f'{num_anomalous:,} anomalous messages out of {len(id_df):,}\n')
    return id_df_labels.join(id_df)

def visualize_data(df, start_time=0, end_time=None):
    num_signals = df.shape[1]
    end_time = df.index.max() if not end_time else end_time
    data = df[((df.index >= start_time) & (df.index < end_time))]
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
    plt.tight_layout()
    plt.show()
    return

def sequences_from_indices(data, indices_ds, start_index, end_index):
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
    start_idx = warm_up
    end_idx = (((len(df)-start_idx) // seq_stride) * seq_stride) + start_idx
    df = df.iloc[start_idx:end_idx]
    if verbose:
        num_sequences = ((len(df)-time_steps-warm_up) // seq_stride) + 1
        print(f"{num_sequences:,} subsequences of length {time_steps}")
        print(f"{ds.__len__().numpy():,} batches (batch size {batch_size})")
    return ds, df

def get_train_val_test(df, params, verbose=False):
    data_length = len(df)
    train_size = int(data_length*params['train_split'])
    val_size = int(data_length*params['val_split'])
    test_size = data_length - train_size - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
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
    if verbose:
        print(f'\nTest:')
    test_ds, test_df = create_dataset(test_df, params, verbose=verbose)
    if verbose:
        print()

    dataset_dict = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    dataframe_dict = {'train': train_df, 'val': val_df, 'test': test_df}
    return dataset_dict, dataframe_dict

def autoencoder(time_steps, warm_up, input_dim, latent_dim, drop_out=False):
    inputs = layers.Input(shape=(time_steps+warm_up, input_dim)) # shape = (time_steps, data_dimension/num_features)
    # encoder
    x = layers.Dense(latent_dim*2, activation='tanh')(inputs)
    if drop_out: x = layers.Dropout(0.2)(x)
    enc_out, enc_hidden = layers.GRU(latent_dim, return_sequences=True, return_state=True, activation="tanh")(x)
    if drop_out:
        enc_out = layers.Dropout(0.2)(enc_out)
        enc_hidden = layers.Dropout(0.2)(enc_hidden)
    # decoder
    dec_out, dec_hidden = layers.GRU(latent_dim, return_sequences=True, return_state=True, activation="tanh")(enc_out)
    if drop_out:
        dec_out = layers.Dropout(0.2)(dec_out)
        dec_hidden = layers.Dropout(0.2)(dec_hidden)
    outputs = layers.Lambda(lambda x: x[:, warm_up:, :])(dec_out)   # used to remove the warm start time steps from the predictions
    outputs = layers.Dense(input_dim, activation='tanh')(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model

def create_model(params):
    model = autoencoder(
        params['time_steps'],
        params['warm_up'],
        params['input_dim'],
        params['latent_dim'],
        params['drop_out'])
    return model

def compile_model(model, params):
    model.compile(optimizer=tf.optimizers.Adam(
        learning_rate=params['learning_rate']),
        loss=params['loss_function'],
        metrics=[params['metric']])
    return

def find_ranges(predictions, index): # used for highlighting in plots
    # accepts a dataframe of 1s and 0s
    # returns a list of range tuples which contain a start and end index
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
            end = index[i]
            # if end - start > 100:
            ranges.append((start, end))
        i += 1
    return ranges



class CentralizedModel:
    def __init__(self, dataframe, params, file_name='model.h5', verbose=False):
        self.datasets, self.dataframes = get_train_val_test(dataframe, params, verbose=verbose)
        self.params = params
        self.save_path = params['model_dir']+file_name
        self.verbose = verbose
        self.initialize_model()
        return

    def initialize_model(self):
        if self.verbose:
            print('Initializing centralized model...\n')
        self.model = create_model(self.params)
        compile_model(self.model, self.params)
        return
    
    def train_model(self, epochs=1, plot_loss=False, evaluate=False):
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



class FederatedClient:
    def __init__(self, dataframe, params, model=None, save_path='client_model.h5', verbose=False):
        self.dataset, self.dataframe = create_dataset(dataframe, params, verbose=verbose)
        self.params = params
        self.save_path = save_path
        self.verbose = verbose
        if model:
            self.set_model(model)
        else:
            self.initialize_model()
        return

    def initialize_model(self):
        self.model = create_model(self.params)
        compile_model(self.model, self.params)
        return
    
    def set_model(self, model):
        self.model = model
        return
    
    def train_model(self, epochs=1):
        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
            filepath=self.save_path)]
        verbose = 'auto' if self.verbose else 0
        self.model.fit(self.dataset,
            epochs=epochs,
            use_multiprocessing=True,
            workers=6,
            shuffle=True,
            callbacks=callbacks_list,
            verbose=verbose)
        return




class FederatedLearning:
    def __init__(self, dataframe, params, verbose=False):
        self.params = params
        self.ds_dict, self.df_dict = get_train_val_test(dataframe, params, verbose=verbose)
        self.global_model = create_model(params)
        compile_model(self.global_model, params)
        self.verbose = verbose
        self.val_loss = float('inf')
        return

    def initialize_clients(self, load_models=False, data_split=None):
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
            print('Initializing clients', end=' ')
            if load_models:
                print(f'using existing models...\n')
            else:
                print(f'and generating models...\n')
        start = 0
        for i in range(num_clients):
            num_samples = int(len(train_df) * data_split[i])
            end = start + num_samples
            df = train_df.iloc[start:end]
            save_path = self.params['model_dir']+'client'+str(i+1)+'_model.h5'
            if load_models:
                client_model = tf.keras.models.load_model(save_path)
            else:
                client_model = tf.keras.models.clone_model(self.global_model)
                client_model.set_weights(self.global_model.get_weights())
            compile_model(client_model, self.params)
            if self.verbose:
                print(f'Client {i+1} dataset:')
            new_client = FederatedClient(df, self.params, client_model,
                                         save_path=save_path, verbose=self.verbose)
            self.clients.append(new_client)
            if self.verbose:
                print()
            start = end
        return
    
    def train_client_models(self, epochs=1):
        num_clients = len(self.clients)
        for i, client in enumerate(self.clients, 1):
            if self.verbose:
                print(f'Training model for client {i}/{num_clients}')
            client.train_model(epochs)
        return
    
    def aggregate_client_models(self, save_global_model=False):
        global_weights = self.global_model.get_weights()
        num_layers = len(global_weights)
        clients_weights = []
        for client in self.clients:
            weights = client.model.get_weights()
            clients_weights.append(weights)
        new_weights = []
        for i in range(num_layers):
            average = np.mean([w[i] for w in clients_weights], axis=0)
            new_weights.append(average.reshape(global_weights[i].shape))
        self.global_model.set_weights(new_weights)
        return
    
    def distribute_global_model(self):
        weights = self.global_model.get_weights()
        for client in self.clients:
            client.model.set_weights(weights)
        return
    
    def validate_global_model(self):
        if self.verbose:
            print('Validating global model...')
        verbose = 'auto' if self.verbose else 0
        loss, metric = self.global_model.evaluate(self.ds_dict['val'], verbose=verbose)
        if self.verbose:
            print(f"Validation loss: {loss:.5}\nValidation {self.params['metric']}: {metric:.5}")
        self.val_loss = loss

    def iterate(self, validate=True):
        self.train_client_models()
        self.aggregate_client_models()
        if validate:
            old_val_loss = self.val_loss
            self.validate_global_model()
            if old_val_loss < self.val_loss:
                return
        save_path = self.params['model_dir']+'global_model.h5'
        tf.keras.models.save_model(self.global_model, save_path)
        self.distribute_global_model()
        return

    def test_global_model(self):
        print('Testing global model...')
        verbose = 'auto' if self.verbose else 0
        loss, metric = self.global_model.evaluate(self.ds_dict['test'], verbose=verbose)
        print(f"Test loss: {loss:.5}\nTest {self.params['metric']}: {metric:.5}")

    def run_federated_learning(self, num_iterations, validation=False, test=False):
        for i in range(num_iterations):
            print(f'\rIteration {i+1}/{num_iterations}', end='')
            if self.verbose:
                print()
            self.iterate(validate=validation)
            if self.verbose:
                print()
        print()
        if test:
            self.test_global_model()



class SynCAN_Evaluator:
    def __init__(self, thresh_df, params, verbose=False):
        if verbose:
            print('Creating threshold dataset...')
        self.thresh_ds, self.thresh_df = create_dataset(thresh_df, params, verbose=verbose)
        self.thresh_df.drop(['Label'], axis=1, errors='ignore')    # remove labels if they exist
        self.params = params
        self.verbose = verbose
        return

    def reconstruct(self, ds): # used for reconstructing signals with a saved model into a continuous dataframe
        reconstruction = self.model.predict(ds, verbose=self.verbose, workers=-1, use_multiprocessing=True)
        time_steps = self.params['time_steps']
        seq_stride = self.params['seq_stride']
        if time_steps == seq_stride:
            reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])
        else:
            # remove duplicate timesteps from predictions
            first = reconstruction[0]
            rest = reconstruction[1:, time_steps-seq_stride:]
            rest = rest.reshape(-1, rest.shape[-1])
            reconstruction = np.concatenate((first, rest))
        columns = ['Signal'+str(i+1) for i in range(reconstruction.shape[-1])]
        return pd.DataFrame(reconstruction, columns=columns)

    def set_thresholds(self, num_stds, plot=False): # used for detecting reconstructions that are significantly different from the orignal
        # returns a numpy array containing a threshold for each signal
        real_values = self.thresh_df.to_numpy()
        if self.verbose:
            print('Reconstructing threshold-selection data...')
        reconstruction = self.reconstruct(self.thresh_ds)
        reconstructed_values = reconstruction.to_numpy()
        thresh_se = np.square(real_values - reconstructed_values)
        self.thresholds = thresh_se.mean(axis=0) + (num_stds * thresh_se.std(axis=0))
        if self.verbose:
            print('Setting squared-error thresholds...')
            for i, t in enumerate(self.thresholds):
                print(f'Signal {str(i+1)}: {t:.5}')
        if plot:
            self.plot_error_thresholds(thresh_se)
        return
    
    def plot_error_thresholds(self, squared_error):
        num_signals = params['num_signals']
        fig, axes = plt.subplots(nrows=num_signals, ncols=1, figsize=(13, 1*num_signals))
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

    def set_predictions(self): # used to label messages as normal or anomalous
        # accepts two dataframes of the same length containing the same number of signals
        # returns a numpy array of 1s and 0s similar to the labels in df
        real_values = self.evaluation_df.to_numpy()[:,1:]      # remove labels before reconstruction
        reconstructed_values = self.reconstructed_df.to_numpy()
        self.eval_se = np.square(real_values - reconstructed_values)
         # if any signal is above respective threshold, prediction is 1 for that timestep
        self.predictions = np.max(1*(self.eval_se > self.thresholds), axis=1)
        return

    def evaluate(self, model, eval_data, thresh_stds=2, plot_thresholds=False):
        self.model = model
        self.set_thresholds(thresh_stds, plot_thresholds)
        self.evaluation_df = eval_data['df']
        labels = eval_data['df']['Label']
        if self.verbose:
            print(f'Reconstructing evaluation data...')
        self.reconstructed_df = self.reconstruct(eval_data['ds'])
        self.reconstructed_df.set_index(self.evaluation_df.index, inplace=True)
        if self.verbose:
            print('Predicting anomalies using thresholds...')
        self.set_predictions()
        print(f'\nAccuracy: {np.mean(self.predictions==labels):.5}\n\n')
        return

    def visualize_reconstruction(self, start_time=0, end_time=None, highlight_anomalies=False, highlight_predictions=False, plot_squared_error=False):
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
                real_ranges = find_ranges(labels.to_numpy(), data.index)
                for start, end in real_ranges:
                    ax0.axvspan(start, end, color='grey', alpha=0.3)
            if highlight_predictions:
                pred_ranges = find_ranges(self.predictions, in_df.index)
                for start, end in pred_ranges:
                    if start > data.index.min() and end < data.index.max():
                        ax1.axvspan(start, end, color='red', alpha=0.3)
            if plot_squared_error: # plot squared error
                ax2 = ax0.twinx()
                se = pd.DataFrame(self.eval_se[:,i], index=in_df.index)
                se = se[((in_df.index >= start_time) & (in_df.index < end_time))]
                se.plot(ax=ax2, alpha=0.5)
                ax2.set_ylim([0,self.thresholds[i]*1.5])
                ax2.axhline(self.thresholds[i], linestyle='--', c='red', alpha=0.5)
                ax2.set_ylabel('Squared Error')
                ax2.legend(['Squared Error'], loc='upper left')
        plt.tight_layout()
        plt.show()
        return


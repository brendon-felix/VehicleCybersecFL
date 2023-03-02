import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt

def import_train_data(csv_path, msg_id, start_time=0, end_time=None):    # imports unlabeled training data into dataframe
    df = pd.read_csv(csv_path)
    df = pd.DataFrame(df.set_index(df.Time))
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
        ax.legend([f'Signal {i}'])
    plt.tight_layout()
    plt.show()
    return

def sequences_from_indices(data, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(data[start_index : end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),  # pylint: disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def timeseries_dataset(data, time_steps, seq_stride,
                       warm_up=0,
                       targets=None,
                       data_is_target=False,
                       shuffle=False,
                       seed=None,
                       start_index=None,
                       end_index=None):
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)
    
    stop_idx = (end_index - start_index) - (time_steps + warm_up)
    # Generate start positions
    start_positions = np.arange(0, stop_idx, seq_stride)
    if shuffle:
        if seed is None:
            seed = np.random.randint(int(1e6))
    rng = np.random.RandomState(seed)
    rng.shuffle(start_positions)
    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat() # infinite times

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
        lambda i, positions: tf.range(positions[i], positions[i] + (time_steps + warm_up)),
        num_parallel_calls=tf.data.AUTOTUNE)
    # create dataset
    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
                lambda i, positions: positions[i],
                num_parallel_calls=tf.data.AUTOTUNE)
        target_ds = sequences_from_indices(targets, indices, start_index, end_index)
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    else:
        if data_is_target:
            if warm_up is None:
                dataset = tf.data.Dataset.zip((dataset, dataset))
            else:
                start_positions = np.arange(warm_up, stop_idx + warm_up, seq_stride)
                positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()
                indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
                    lambda i, positions: tf.range(positions[i], positions[i] + time_steps),
                    num_parallel_calls=tf.data.AUTOTUNE)
                target_ds = sequences_from_indices(data, indices, start_index, end_index)
                dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset

def create_dataset(df, params, train=True):
    time_steps = params['time_steps']
    seq_stride = params['seq_stride']
    warm_up = params['warm_up']
    batch_size = params['batch_size']
    data = df.to_numpy() if train else df.to_numpy()[:,1:]
    ds = timeseries_dataset(data, time_steps, seq_stride, warm_up, data_is_target=train).batch(batch_size)
    start_idx = warm_up
    end_idx = (((len(df)-start_idx) // seq_stride) * seq_stride) + start_idx
    df = df.iloc[start_idx:end_idx]
    return ds, df

def get_training_data(df, params, verbose=False):
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
    train_ds, train_df = create_dataset(train_df, params, True)
    val_ds, val_df = create_dataset(val_df, params, True)
    test_ds, test_df = create_dataset(test_df, params, True)
    dataset_dict = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    dataframe_dict = {'train': train_df, 'val': val_df, 'test': test_df}
    if verbose:
        for key, item in dataset_dict.items():
            num_sequences = ((len(dataframe_dict[key])-time_steps-warm_up) // seq_stride) + 1
            print(f"{key.upper()}: \t{num_sequences:,} subsequences of length {time_steps}")
    return dataset_dict, dataframe_dict

def get_evaluation_data(df, params, verbose=False):
    time_steps = params['time_steps']
    seq_stride = params['seq_stride']
    warm_up = params['warm_up']
    eval_ds, eval_df = create_dataset(df, params, False)
    eval_data = {'ds': eval_ds, 'df': eval_df}
    if verbose:
        num_sequences = ((len(df)-time_steps-warm_up) // seq_stride) + 1
        print(f"{num_sequences:,} subsequences of length {time_steps}")
    return eval_data

def autoencoder(time_steps, warm_up, input_dim, latent_dim, drop_out):
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

class FederatedClient:
    def __init__(self, dataframe, params, save_path, verbose=False):
        self.verbose = verbose
        self.data = dataframe.to_numpy()
        self.params = params
        self.save_path = save_path
        self.datasets, self.dataframes = get_training_data(dataframe, params, verbose)
        self.model = self.create_model()
        return

    def create_model(self):
        model = autoencoder(
            self.params['time_steps'],
            self.params['warm_up'],
            self.params['input_dim'],
            self.params['latent_dim'],
            self.params['drop_out'])
        model.compile(optimizer=tf.optimizers.Adam(
            learning_rate=self.params['learning_rate']),
            loss=self.params['loss_function'],
            metrics=[self.params['metric']])
        if self.verbose:
            model.summary()
        return model
    
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
            print('Evaluating...')
            loss, metric = self.model.evaluate(self.datasets['test'], verbose=0)
            print(f"Test loss: {loss:.5}\nTest {self.params['metric']}: {metric:.5}")
        return

class FederatedAggregator:
    def __init__(self, clients):
        self.clients = clients
        return


class SynCAN_Evaluator:
    def __init__(self, model, params, verbose=False):
        self.verbose = verbose
        self.model = model
        self.params = params
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

    def set_thresholds(self, data, num_stds, plot=False): # used for detecting reconstructions that are significantly different from the orignal
        # returns a numpy array containing a threshold for each signal
        real_values = data['df'].to_numpy()[:,1:]
        if self.verbose:
            print('Reconstructing threshold-selection data...')
        reconstruction = self.reconstruct(data['ds'])
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
        for signal in range(squared_error.shape[-1]): # Plot histograms of squared error values and mean + threshold lines
            se = squared_error[:,signal]
            plt.figure(figsize=(12,3))
            sns.set(font_scale = 1)
            sns.set_style("white")
            plt.xlim([0, 2*self.thresholds[signal]])
            sns.histplot(np.clip(se, 0, 2 * self.thresholds[signal]), bins=50, kde=True, color='grey')
            plt.axvline(x=np.mean(se), color='g', linestyle='--', linewidth=3)
            plt.text(np.mean(se), 250, "Mean", horizontalalignment='left', 
                    size='small', color='black', weight='semibold')
            plt.axvline(x=self.thresholds[signal], color='b', linestyle='--', linewidth=3)
            plt.text(self.thresholds[signal], 250, "Threshold", horizontalalignment='left', 
                    size='small', color='Blue', weight='semibold')
            plt.xlabel('Squared Error')
            plt.title('Signal '+str(signal+1))
            sns.despine()
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

    def evaluate(self, eval_data):
        self.evaluation_df = eval_data['df']
        if self.verbose:
            print(f'Reconstructing evaluation data...')
        self.reconstructed_df = self.reconstruct(eval_data['ds'])
        self.reconstructed_df.set_index(self.evaluation_df.index, inplace=True)
        self.set_predictions()
        return

    def visualize_reconstruction(self, start_time=0, end_time=None, highlight_anomalies=False, highlight_predictions=False, plot_se=False):
        # accepts two dataframes of the same length with the same number of signals - keys must be Signal1, Signal2,...
        in_df = self.evaluation_df
        out_df = self.reconstructed_df
        end_time = in_df.index.max() if not end_time else end_time
        data = in_df[((in_df.index >= start_time) & (in_df.index < end_time))]
        reconstructed_data = out_df[((out_df.index >= start_time) & (out_df.index < end_time))]

        num_signals = reconstructed_data.shape[1]
        labels = data.Label
        msg_id = params['msg_id']

        fig, axes = plt.subplots(nrows=num_signals, ncols=1, figsize=(13, 2*num_signals), sharex=True)
        for i in range(num_signals):
            key = 'Signal'+str(i+1)
            t_data = data[key]
            t_reconstructed_data = reconstructed_data[key]
            ax = list(axes)[i]
            ax0 = t_data.plot(ax=ax, color="black", title=msg_id.upper()+'_'+key, rot=25)
            ax1 = t_reconstructed_data.plot(ax=ax, color="red", rot=10)
            ax1.legend(['Original Signal', 'Reconstructed Signal'], loc='upper left')

            if plot_se: # plot squared error
                ax2 = ax0.twinx()
                se = pd.DataFrame(self.eval_se[:,i], index=in_df.index)
                se = se[((in_df.index >= start_time) & (in_df.index < end_time))]
                se.plot(ax=ax2, alpha=0.5)
                ax2.axhline(self.thresholds[i], linestyle='--', alpha=0.5)
                ax2.set_ylabel('Squared Error')
                ax2.legend(['Squared Error'], loc='upper right')

            if highlight_anomalies:
                real_ranges = find_ranges(labels.to_numpy(), data.index)
                for start, end in real_ranges:
                    ax0.axvspan(start, end, color='grey', alpha=0.3)
            if highlight_predictions:
                pred_ranges = find_ranges(self.predictions, in_df.index)
                for start, end in pred_ranges:
                    if start > data.index.min() and end < data.index.max():
                        ax1.axvspan(start, end, color='red', alpha=0.3)
        plt.tight_layout()
        plt.show()
        return


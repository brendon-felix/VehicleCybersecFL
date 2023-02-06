import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Download SynCAN dataset from the ETAS github
# !git clone https://github.com/etas/SynCAN.git
# !unzip ./SynCAN/\*.zip -d ./SynCAN/. &> /dev/null
# !rm ./SynCAN/*.zip &> /dev/null

def train_csv2df(dir_path): # imports training files into a dataframe
    data_frames = []
    csv_path = dir_path + 'train_1.csv'    # train 1 contains header
    df_temp = pd.read_csv(csv_path)
    data_frames.append(df_temp)
    for i in range(2, 5):
        csv_path = dir_path + 'train_' + str(i) + '.csv'
        df_temp = pd.read_csv(csv_path, header=None, names=['Label',  'Time', 'ID',
                                                            'Signal1', 'Signal2',  'Signal3', 'Signal4'])
        data_frames.append(df_temp)
    df = pd.concat(data_frames)
    return df



def prepare_train_data(syncan_dir, start_time=0, end_time=None, msg_id=None):
    df = train_csv2df(syncan_dir)
    df['Time'] = df['Time'] - df['Time'].min()      # set starting time to 0
    df = df.set_index(df['Time'])
    end_time = df.Time.max() if not end_time else end_time
    df = df[((df.Time >= start_time) & (df.Time < end_time))]
    # display(df)
    print(f'{len(df):,} total messages (id1,id2,...,id10)\n')
    if msg_id is None:
        dataframes = []
        for i in range(10):
            msg_id = 'id'+str(i+1)
            id_df = df[df.ID==msg_id]
            id_df = id_df.dropna(axis=1, how='all') # remove unused signal columns
            dataframes.append(id_df)
            print(f'{msg_id.upper()}: {len(id_df):,} messages')
            return dataframes
    else:
        id_df = df[df.ID==msg_id]
        id_df = id_df.dropna(axis=1, how='all')
        print(f'{msg_id.upper()}: {len(id_df):,} messages')
        return id_df



# visualize syncan signals for a single message ID
def visualize_data(df, start_time=0, end_time=None):
    end_time = df.Time.max() if not end_time else end_time
    data = df[((df.Time >= start_time) & (df.Time < end_time))]
    msg_id = data.ID.iloc[0]
    colors = ['blue', 'orange', 'green', 'red']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(df.shape[1]-3):
        key = 'Signal'+str(i+1)
        c = colors[i % (len(colors))]
        t_data = data[key]
        ax = t_data.plot(
            ax = axes[i // 2, i % 2],
            color = c,
            title = msg_id.upper()+'_'+key,
            rot = 25,
        )
    plt.tight_layout()
    plt.show()
    return
  
  

# derived from the keras timeseries_dataset_from_array function
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array
def timeseries_dataset(data, targets, sequence_length,
                       sequence_stride=1,
                       data_is_target=False,
                       warm_up=0,
                       batch_size=1,
                       shuffle=False,
                       seed=None,
                       start_index=None,
                       end_index=None):
  
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = (end_index - start_index) - (sequence_length + warm_up) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = 'int32'
    else:
        index_dtype = 'int64'

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
    rng = np.random.RandomState(seed)
    rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    warm_up = tf.cast(warm_up, dtype=index_dtype)
    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat() # infinite times

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
          lambda i, positions: tf.range(positions[i], positions[i] + (sequence_length + warm_up)),
          num_parallel_calls=tf.data.AUTOTUNE)

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
                start_positions = np.arange(warm_up, num_seqs + warm_up, sequence_stride, dtype=index_dtype)
                positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()
                indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
                    lambda i, positions: tf.range(positions[i], positions[i] + sequence_length),
                    num_parallel_calls=tf.data.AUTOTUNE)
                target_ds = sequences_from_indices(data, indices, start_index, end_index)
                dataset = tf.data.Dataset.zip((dataset, target_ds))
            
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
            dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
            
    return dataset


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index : end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),  # pylint: disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset



def reconstruct(ds, model_path, time_steps, seq_stride): # used for reconstructing signals with a saved model into a continuous dataframe
    saved_model = tf.keras.models.load_model(model_path)
    reconstruction = saved_model.predict(ds, verbose=1, workers=-1, use_multiprocessing=True)
    if time_steps == seq_stride:
        reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])
    else:
        # remove duplicate timesteps from predictions
        # first = reconstruction[0,:TIME_STEPS-SEQ_STRIDE]    # e.g. [0..10] out of [0..20]
        first = reconstruction[0]
        rest = reconstruction[1:, time_steps-seq_stride:]    # e.g. [10..20],[20..30].. out of [10..30],[20..40]..
        rest = rest.reshape(-1, rest.shape[-1])             # e.g. [10....len(df)]
        reconstruction = np.concatenate((first, rest))      # e.g. [0....len(df)]
    columns = ['Signal'+str(i+1) for i in range(reconstruction.shape[-1])]
    return pd.DataFrame(reconstruction, columns=columns)



def get_thresholds(df, reconstruction): # used for detecting reconstructions that are significantly different from the orignal
    # returns a numpy array containing a threshold for each signal
    real_values = df.to_numpy()[:,1:]
    reconstructed_values = reconstruction.to_numpy()
    squared_error = np.square(real_values - reconstructed_values)
    thresholds = squared_error.mean(axis=0) + (2 * squared_error.std(axis=0))
    return thresholds, squared_error



def get_predictions(df, reconstruction, thresholds): # used to label messages as normal or anomalous
    # accepts two dataframes of the same length containing the same number of signals
    # returns a numpy array of 1s and 0s similar to the labels in df
    real_values = df.to_numpy()[:,1:]
    reconstructed_values = reconstruction.to_numpy()
    squared_error = np.square(real_values - reconstructed_values)
     # if any signal is above respective threshold, prediction is 1 for that timestep
    predictions = 1*(squared_error > thresholds)
    return predictions



def find_ranges(predictions): # used for highlighting in plots
    # accepts a list/vector of 1s and 0s
    # returns a list of range tuples which contain a start and end index
    ranges = []
    i = 0
    while i < len(predictions):
        if predictions[i]:
            start = i
            while True:
                i += 1
                if i >= len(predictions):
                    break
                if not predictions[i]:
                    break
            end = i
            if end - start > 10:
                ranges.append((start, end))
        i += 1
    return ranges



def visualize_reconstructed_signal(in_df, out_df, start_time=0, end_time=None, predictions=None, show_se=False):
    # accepts two dataframes of the same length with the same number of signals - keys must be Signal1, Signal2,...
    out_df.set_index(in_df.Time)
    end_time = in_df.Time.max() if not end_time else end_time
    data = in_df[((in_df.Time >= start_time) & (in_df.Time < end_time))]
    reconstructed_data = out_df[((out_df.index >= start_time) & (out_df.index < end_time))]

    num_signals = reconstructed_data.shape[1]
    labels = data.Label
    msg_id = data.ID.iloc[0]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(num_signals):
        key = 'Signal'+str(i+1)
        t_data = data[key]
        t_reconstructed_data = reconstructed_data[key]
        ax = axes[i // 2, i % 2]
        ax0 = t_data.plot(ax=ax, color="black", title=msg_id.upper()+'_'+key, rot=25)
        ax1 = t_reconstructed_data.plot(ax=ax, color="red", rot=10)
        ax1.legend(['Original Signal', 'Reconstructed Signal'], loc='upper left')

        if show_se: # plot squared error
            se = np.square(t_data - t_reconstructed_data)
            ax2 = ax0.twinx()
            ax2.plot(se, alpha=0.5)
            ax2.set_ylabel('squared error')
            ax2.legend(['Squared Error'], loc='upper right')

        ranges = find_ranges(labels.to_numpy())
        for start, end in ranges:
            ax0.axvspan(start, end, color='grey', alpha=0.3)
        if predictions is not None:
            ranges = find_ranges(predictions)
            print(ranges)
            for start, end in ranges:
                ax1.axvspan(start, end, color='red', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return

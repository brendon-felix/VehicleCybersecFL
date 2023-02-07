
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt



class IntrusionDetector:
	def __init__(self):
		self.SIGNAL_COUNTS = [2, 3, 2, 1, 2, 2, 2, 1, 1, 4]
		self.set_constants()

	def set_constants(self,
					  msg_id_number=5,
					  train_split=0.8,
					  val_split=0.1,
					  warm_up=20,
					  time_steps=100,
					  seq_stride=None,
					  epochs=3,
					  batch_size=64,
					  learning_rate=0.001,
					  drop_out=True,
					  loss_function='mse',
					  metric='mean_squared_logarithmic_error',
					  patience=1,
					  latent_dim=None,
					  directory=None,
					  eval_set='continuous'
					  ):
		self.ID = msg_id_number
		self.MSG_ID = 'id'+str(ID)	  # each message contains 1-4 signals that are associated with the specified message ID
		self.NUM_SIGNALS = self.SIGNAL_COUNTS[self.ID-1]
		self.TRAIN_SPLIT = train_split
		self.VAL_SPLIT = val_split
		self.WARM_UP = warm_up
		self.TIME_STEPS = time_steps
		self.SEQ_STRIDE = time_steps // 2 if not seq_stride else seq_stride
		self.EPOCHS = epochs
		self.BATCH_SIZE = batch_size
		self.LEARNING_RATE = learning_rate
		self.DROP_OUT = drop_out
		self.LOSS_FUNCTION = loss_function
		self.METRIC = metric
		self.PATIENCE = patience
		self.LATENT_DIM = 32 * self.NUM_SIGNALS if not latent_dim else latent_dim
		self.DIRECTORY = "/content/drive/MyDrive/Colab Notebooks/models/cids/" if not directory else directory
		self.FILEPATH = self.DIRECTORY+self.MSG_ID+"_model_"+str(time_steps)+'-'+str(warm_up)+".h5"
		self.EVAL_SET = 'continuous'
		# self.PNGFILEPATH = self.DIRECTORY+self.MSG_ID+"_model.png"


	def train_csv2df(self, dir_path): # imports training files into a dataframe
		data_frames = []
		csv_path = dir_path + 'train_1.csv'	# train 1 contains header
		df_temp = pd.read_csv(csv_path)
		data_frames.append(df_temp)
		for i in range(2, 5):
			csv_path = dir_path + 'train_' + str(i) + '.csv'
			df_temp = pd.read_csv(csv_path, header=None, names=['Label',  'Time', 'ID',
														'Signal1', 'Signal2',  'Signal3', 'Signal4'])
			data_frames.append(df_temp)
		df = pd.concat(data_frames)
		return df



	def prepare_train_data(self, syncan_dir, start_time=0, end_time=None, msg_id=None):
		df = self.train_csv2df(syncan_dir)
		df['Time'] = df['Time'] - df['Time'].min()	  # set starting time to 0
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
			print(f'{msg_id.upper()}: {len(id_df):,} messages used')
			return id_df



	# visualize syncan signals for a single message ID
	def visualize_data(self, df, start_time=0, end_time=None):
		end_time = df.Time.max() if not end_time else end_time
		data = df[((df.Time >= start_time) & (df.Time < end_time))]
		msg_id = data.ID.iloc[0]
		colors = ['blue', 'orange', 'green', 'red']
		fig, axes = plt.subplots(nrows=self.NUM_SIGNALS, ncols=1, figsize=(13, 5), dpi=80, facecolor='w', edgecolor='k')
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
		print(msg_id.upper())
		plt.show()
		return



	# derived from the keras timeseries_dataset_from_array function
	# https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array
	def timeseries_dataset(self, data, targets, sequence_length,
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
		# create dataset
		dataset = self.sequences_from_indices(data, indices, start_index, end_index)
		if targets is not None:
			indices = tf.data.Dataset.zip(
				(tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
					lambda i, positions: positions[i],
					num_parallel_calls=tf.data.AUTOTUNE)
			target_ds = self.sequences_from_indices(targets, indices, start_index, end_index)
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
					target_ds = self.sequences_from_indices(data, indices, start_index, end_index)
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


	def sequences_from_indices(self, array, indices_ds, start_index, end_index):
		dataset = tf.data.Dataset.from_tensors(array[start_index : end_index])
		dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
			lambda steps, inds: tf.gather(steps, inds),  # pylint: disable=unnecessary-lambda
			num_parallel_calls=tf.data.AUTOTUNE)
		return dataset



	def SynCAN_train_ds(self, df):
		values = df.to_numpy()[:,3:].astype(np.float32)
		ds = self.timeseries_dataset(
			values,
			None,
			data_is_target = True,
			warm_up = self.WARM_UP,
			sequence_length = self.TIME_STEPS,
			sequence_stride = self.SEQ_STRIDE,
			# batch_size = self.BATCH_SIZE
		)
		return ds



	def get_centralized_datasets(self, df):
		data_length = len(df)
		train_size = ((int(data_length*self.TRAIN_SPLIT) // self.SEQ_STRIDE) * self.SEQ_STRIDE) + self.WARM_UP
		val_size = ((int(data_length*self.VAL_SPLIT) // self.SEQ_STRIDE) * self.SEQ_STRIDE) + self.WARM_UP
		test_size = (((data_length - train_size - val_size) // self.SEQ_STRIDE) * self.SEQ_STRIDE) + self.WARM_UP

		train_df = df.iloc[:train_size]
		val_df = df[train_size:train_size+val_size]
		test_df = df.iloc[-test_size:]

		train_ds = SynCAN_train_ds(train_df)
		val_ds = SynCAN_train_ds(val_df)
		test_ds = SynCAN_train_ds(test_df)

		datasets = {'train': train_ds, 'val': val_ds, 'test': test_ds}
		dataframes = {'train': train_df.iloc[self.WARM_UP:], 'val': val_df.iloc[self.WARM_UP:], 'test': test_df.iloc[self.WARM_UP:]}
		for key, item in datasets.items():
			print(f"{key.upper()}: \t{item.__len__().numpy():,} subsequences of length {self.TIME_STEPS}")
		return datasets, dataframes



	def reconstruct(self, ds, model_path): # used for reconstructing signals with a saved model into a continuous dataframe
		saved_model = tf.keras.models.load_model(model_path)
		reconstruction = saved_model.predict(ds, verbose=1, workers=-1, use_multiprocessing=True)
		if self.TIME_STEPS == self.SEQ_STRIDE:
			reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])
		else:
			# remove duplicate timesteps from predictions
			# first = reconstruction[0,:self.TIME_STEPS-self.SEQ_STRIDE]	# e.g. [0..10] out of [0..20]
			first = reconstruction[0]
			rest = reconstruction[1:, self.TIME_STEPS-self.SEQ_STRIDE:]	# e.g. [10..20],[20..30].. out of [10..30],[20..40]..
			rest = rest.reshape(-1, rest.shape[-1])			 # e.g. [10....len(df)]
			reconstruction = np.concatenate((first, rest))	  # e.g. [0....len(df)]
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



	def get_CIDS_model(time_steps, warm_up, input_dim, latent_dim, drop_out=False):
		inputs = layers.Input(shape=(time_steps+warm_up, input_dim)) # shape = (time_steps, data_dimension/num_features)
		# encoder
		x = layers.Dense(latent_dim*2, activation='tanh')(inputs)
		if drop_out: x = layers.Dropout(0.2)(x)
		enc_out, enc_hidden = layers.GRU(latent_dim, return_sequences=True, return_state=True, activation="tanh")(x)
		if drop_out:
			enc_out = layers.Dropout(0.2)(enc_out)
			enc_hidden = layers.Dropout(0.2)(enc_hidden)
		# decoder
		dec_out , dec_hidden = layers.GRU(latent_dim, return_sequences=True, return_state=True, activation="tanh")(enc_out)
		if drop_out:
			dec_out = layers.Dropout(0.2)(dec_out)
			dec_hidden = layers.Dropout(0.2)(dec_hidden)
		outputs = layers.Lambda(lambda x: x[:, warm_up:, :])(dec_out)   # used to remove the warm start time steps from the predictions
		outputs = layers.Dense(input_dim, activation='tanh')(outputs)
		model = tf.keras.Model(inputs, outputs)
		return model



	def create_model(time_steps, warm_up, input_dim, latent_dim, drop_out=False):
		model = get_CIDS_model(time_steps, warm_up, input_dim, latent_dim, drop_out)
		model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.LEARNING_RATE),
			loss=self.LOSS_FUNCTION,
			metrics=[self.METRIC])
		model.summary()
		return model



	def train_model(model, train_ds, val_ds=None, plot=False, evaluate=False):
		callbacks_list = [
			tf.keras.callbacks.EarlyStopping(
				monitor="val_loss",
				patience=self.PATIENCE,),
			tf.keras.callbacks.ModelCheckpoint(
				filepath=self.FILEPATH,
				monitor="val_loss",
				save_best_only=True)]
		history = model.fit(train_ds,
						epochs=self.EPOCHS,
						validation_data=val_ds,
						use_multiprocessing=True,
						workers=6,
						batch_size=self.BATCH_SIZE,
						shuffle=True,
						callbacks=callbacks_list)
		if plot:
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.title('model loss')
			plt.ylabel('loss')
			plt.xlabel('epoch')
			plt.legend(['train', 'test'], loc='upper left')
			plt.show()
		if evaluate:
			print('Evaluating...')
			loss, metric = model.evaluate(test_ds, verbose=0)
			print(f'Test loss: {loss:.5}\nTest {self.METRIC}: {metric:.5}')		# current best: 8.8552e-05
		return



	def plot_error_thresholds(thresholds, squared_error):
		for signal in range(squared_error.shape[-1]): # Plot histograms of squared error values and mean + threshold lines
			se = squared_error[:,signal]
			plt.figure(figsize=(12,3))
			sns.set(font_scale = 1)
			sns.set_style("white")
			plt.xlim([0, 2*thresholds[signal]])
			sns.histplot(np.clip(se, 0, 2 * thresholds[signal]), bins=50, kde=True, color='grey')
			plt.axvline(x=np.mean(se), color='g', linestyle='--', linewidth=3)
			plt.text(np.mean(se), 250, "Mean", horizontalalignment='left', 
					size='small', color='black', weight='semibold')
			plt.axvline(x=thresholds[signal], color='b', linestyle='--', linewidth=3)
			plt.text(thresholds[signal], 250, "Threshold", horizontalalignment='left', 
					size='small', color='Blue', weight='semibold')
			plt.xlabel('Squared Error')
			plt.title('Signal '+str(signal+1))
			sns.despine()
		return

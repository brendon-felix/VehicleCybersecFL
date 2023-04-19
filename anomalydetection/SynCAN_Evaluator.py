from functions import *
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

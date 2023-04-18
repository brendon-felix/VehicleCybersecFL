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

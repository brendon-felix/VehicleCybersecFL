from functions import *
class FederatedClient:
    '''Class for an individual client used in federated learning
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

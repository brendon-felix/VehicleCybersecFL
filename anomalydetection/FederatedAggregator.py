from functions import *
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

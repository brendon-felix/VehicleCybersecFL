from functions import *
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

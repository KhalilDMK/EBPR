import pandas as pd
import random
import itertools
from Code.EBPR_model import BPREngine
from Code.data import SampleGenerator, read_data

# Read dataset
dataset_name = 'lastfm-2k'  # 'ml-100k' for Movielens 100K. 'ml-1m' for the Movielens 1M dataset. 'lastfm-2k' for the Last.FM 2K dataset.
dataset = read_data(dataset_name)

# Define hyperparameters

model_name = 'UEBPR'  # Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'.
loo_eval = True  # True: LOO evaluation with HR@k and NDCG@k. False: Random train/test split
latent_factors = [5, 10, 20, 50, 100]
batch_sizes = [50, 100, 500]
l2_regularizations = [0, 0.00001, 0.001]
neighborhood_sizes = [5, 10, 15, 20, 25, 50]
num_reps = 5  # Number of replicates per hyperparameter configuration.
num_epochs = 100  # Number of epochs.
num_configurations = 15  # Number of random hyperparameter configurations.

hyper_tun_configurations = random.sample(set(itertools.product(latent_factors, batch_sizes, l2_regularizations, neighborhood_sizes)), num_configurations)

# Define results dataframe

if loo_eval:
    results = pd.DataFrame(columns=['latent', 'batch_size', 'l2_reg', 'neighborhood', 'rep', 'ndcg', 'hr', 'mep', 'wmep'])
else:
    results = pd.DataFrame(columns=['latent', 'batch_size', 'l2_reg', 'neighborhood', 'rep', 'ndcg', 'map', 'mep', 'wmep'])

# Hyperparameter tuning experiments

for hyper_tun_configuration in hyper_tun_configurations:
    for rep in range(num_reps):
        print('config ' + str(hyper_tun_configuration) + ', rep ' + str(rep))
        config = {'model': model_name,
                  'dataset': dataset_name,
                  'num_epoch': num_epochs,
                  'batch_size': hyper_tun_configuration[1],
                  'lr': 0.001,  # Learning rate.
                  #'optimizer': 'sgd',
                  #'sgd_momentum': 0.9,
                  #'optimizer': 'rmsprop',
                  #'rmsprop_alpha': 0.99,
                  #'rmsprop_momentum': 0,
                  'optimizer': 'adam',
                  'num_users': len(dataset['userId'].unique()),
                  'num_items': len(dataset['itemId'].unique()),
                  'test_rate': 0.2,  # Test rate for random train/test split. Used when 'loo_eval' is set to False.
                  'num_latent': hyper_tun_configuration[0],
                  'weight_decay': 0,
                  'l2_regularization': hyper_tun_configuration[2],
                  'use_cuda': True,
                  'device_id': 0,
                  'top_k': 10,  # k in MAP@k, HR@k and NDCG@k.
                  'loo_eval': loo_eval,
                  # evaluation with MAP@k and NDCG@k.
                  'neighborhood': hyper_tun_configuration[3],
                  'model_dir_explicit':'../Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}.model',
                  'model_dir_implicit':'../Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}.model'}

        # DataLoader
        sample_generator = SampleGenerator(dataset, config)
        evaluation_data = sample_generator.test_data_loader(config['batch_size'])

        # Create explainability matrix
        explainability_matrix = sample_generator.create_explainability_matrix()

        # Create popularity vector
        popularity_vector = sample_generator.create_popularity_vector()

        #Create item neighborhood
        neighborhood = sample_generator.create_neighborhood()

        # Specify the exact model
        engine = BPREngine(config)

        # Initialize list of optimal results
        best_performance = [0] * 5

        best_model = ''
        for epoch in range(config['num_epoch']):
            print('Training epoch {}'.format(epoch))
            train_loader = sample_generator.train_data_loader(config['batch_size'])
            engine.train_an_epoch(train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id=epoch)
            if config['loo_eval']:
                hr, ndcg, mep, wmep = engine.evaluate(evaluation_data, explainability_matrix, epoch_id=epoch)
                print('-' * 80)
                best_model, best_performance = engine.save_implicit(epoch, ndcg, hr, mep, wmep, config['num_epoch'], best_model, best_performance)
            else:
                map, ndcg, mep, wmep = engine.evaluate(evaluation_data, explainability_matrix, epoch_id=epoch)
                print('-' * 80)
                best_model, best_performance = engine.save_explicit(epoch, map, ndcg, mep, wmep, config['num_epoch'], best_model, best_performance)

        # Save results to dataframe
        if config['loo_eval']:
            results = results.append(
                {'latent': config['num_latent'], 'batch_size': config['batch_size'], 'l2_reg': config['l2_regularization'],
                 'neighborhood': config['neighborhood'], 'rep': rep, 'ndcg': best_performance[0],
                 'hr': best_performance[1], 'mep': best_performance[2], 'wmep': best_performance[3]},
                ignore_index=True)
        else:
            results = results.append(
                {'latent': config['num_latent'], 'batch_size': config['batch_size'], 'l2_reg': config['l2_regularization'],
                 'neighborhood': config['neighborhood'], 'rep': rep, 'ndcg': best_performance[1],
                 'map': best_performance[0], 'mep': best_performance[2], 'wmep': best_performance[3]},
                ignore_index=True)

# Save dataframe
results.to_csv('../Output/Hyperparameter_tuning_' + model_name + '_' + dataset_name + '.csv')
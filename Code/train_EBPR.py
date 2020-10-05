from Code.EBPR_model import BPREngine
from Code.data import SampleGenerator, read_data

# Read dataset
dataset_name = 'ml-100k'  # For Movielens 100K. Also try 'ml-1m' for the Movielens 1M dataset
dataset = read_data(dataset_name)

# Define hyperparameters
BPR_config = {'alias': 'BPR_' + dataset_name,
              'num_epoch': 5,
              'batch_size': 500,
              'lr': 0.001,
              #'optimizer': 'sgd',
              #'sgd_momentum': 0.9,
              #'optimizer': 'rmsprop',
              #'rmsprop_alpha': 0.99,
              #'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'num_users': len(dataset['userId'].unique()),
              'num_items': len(dataset['itemId'].unique()),
              'test_rate': 0.2,  # Test rate for random train/test split. Used when 'loo_eval' is set to False.
              'num_latent': 50,
              'weight_decay': 0,
              'l2_regularization': 0,
              'use_cuda': True,
              'device_id': 0,
              'top_k': 10,  # k in MAP@k, HR@k and NDCG@k.
              'loo_eval': False,  # True: LOO evaluation with HR@k and NDCG@k. False: Random train/test split
              # evaluation with MAP@k and NDCG@k.
              'neighborhood': 5,  # Neighborhood size for explainability.
              'epsilon': 0.001,  # Epsilon smoothing parameter in explainability matrix.
              'model_dir_explicit':'../Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}.model',
              'model_dir_implicit':'../Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}.model'}

config = BPR_config

# DataLoader
sample_generator = SampleGenerator(dataset, config)
evaluation_data = sample_generator.test_data_loader(config['batch_size'])

#Create explainability matrix
explainability_matrix = sample_generator.create_explainability_matrix()

# Specify the exact model
engine = BPREngine(config)

# Initialize list of optimal results
best_performance = [0] * 3

best_model = ''
for epoch in range(config['num_epoch']):
    print('Training epoch {}'.format(epoch))
    train_loader = sample_generator.train_data_loader(config['batch_size'])
    engine.train_an_epoch(train_loader, explainability_matrix, epoch_id=epoch)
    if config['loo_eval']:
        hr, ndcg = engine.evaluate(evaluation_data, epoch_id=epoch)
        print('-' * 80)
        best_model, best_performance = engine.save_implicit(config['alias'], epoch, ndcg, hr, config['num_epoch'], best_model, best_performance)
    else:
        map, ndcg = engine.evaluate(evaluation_data, epoch_id=epoch)
        print('-' * 80)
        best_model, best_performance = engine.save_explicit(config['alias'], epoch, map, ndcg, config['num_epoch'], best_model, best_performance)
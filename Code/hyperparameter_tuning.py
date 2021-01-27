import pandas as pd
import random
import itertools
from Code.EBPR_model import BPREngine
from Code.data import SampleGenerator, read_data
import argparse

def main(args):
    # Read dataset
    dataset_name = args.dataset  # 'ml-100k' for Movielens 100K. 'ml-1m' for the Movielens 1M dataset. 'lastfm-2k' for the
    # Last.FM 2K dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
    dataset = read_data(dataset_name)

    # Define hyperparameters

    model_name = args.model  # Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'.
    loo_eval = args.loo_eval  # True: LOO evaluation with HR@k and NDCG@k. False: Random train/test split
    latent_factors = [5, 10, 20, 50, 100]
    batch_sizes = [50, 100, 500]
    l2_regularizations = [0, 0.00001, 0.001]
    #neighborhood_sizes = [5, 10, 15, 20, 25, 50]
    neighborhood_sizes = [args.neighborhood]
    num_reps = args.num_reps  # Number of replicates per hyperparameter configuration.
    num_epochs = args.num_epoch  # Number of epochs.
    num_configurations = args.num_configurations  # Number of random hyperparameter configurations.

    hyper_tun_configurations = random.sample(set(itertools.product(latent_factors, batch_sizes, l2_regularizations, neighborhood_sizes)), num_configurations)

    # Define results dataframe

    if loo_eval:
        results = pd.DataFrame(columns=['latent', 'batch_size', 'l2_reg', 'neighborhood', 'rep', 'ndcg', 'hr', 'mep', 'wmep', 'avg_pop', 'efd', 'avg_pair_sim'])
    else:
        results = pd.DataFrame(columns=['latent', 'batch_size', 'l2_reg', 'neighborhood', 'rep', 'ndcg', 'map', 'mep', 'wmep', 'avg_pop', 'efd', 'avg_pair_sim'])

    # Hyperparameter tuning experiments

    for hyper_tun_configuration in hyper_tun_configurations:
        for rep in range(num_reps):
            print('config ' + str(hyper_tun_configuration) + ', rep ' + str(rep))
            config = {'model': model_name,
                      'dataset': dataset_name,
                      'num_epoch': num_epochs,
                      'batch_size': hyper_tun_configuration[1],
                      'lr': args.lr,  # Learning rate.
                      #'optimizer': 'sgd',
                      'sgd_momentum': args.sgd_momentum,
                      #'optimizer': 'rmsprop',
                      'rmsprop_alpha': args.rmsprop_alpha,
                      'rmsprop_momentum': args.rmsprop_momentum,
                      'optimizer': args.optimizer,
                      'num_users': len(dataset['userId'].unique()),
                      'num_items': len(dataset['itemId'].unique()),
                      'test_rate': args.test_rate,  # Test rate for random train/val/test split. test_rate is the rate of test + validation. Used when 'loo_eval' is set to False.
                      'num_latent': hyper_tun_configuration[0],
                      'weight_decay': args.weight_decay,
                      'l2_regularization': hyper_tun_configuration[2],
                      'use_cuda': args.use_cuda,
                      'device_id': args.device_id,
                      'top_k': args.top_k,  # k in MAP@k, HR@k and NDCG@k.
                      'loo_eval': loo_eval,
                      # evaluation with MAP@k and NDCG@k.
                      'neighborhood': hyper_tun_configuration[3],
                      'model_dir_explicit':'Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model',
                      'model_dir_implicit':'Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model'}

            # DataLoader
            sample_generator = SampleGenerator(dataset, config, split_val=True)
            validation_data = sample_generator.val_data_loader(config['batch_size'])

            # Create explainability matrix
            explainability_matrix = sample_generator.create_explainability_matrix()
            val_explainability_matrix = sample_generator.create_explainability_matrix(include_test=True)

            # Create popularity vector
            popularity_vector = sample_generator.create_popularity_vector()
            val_popularity_vector = sample_generator.create_popularity_vector(include_test=True)

            # Create item neighborhood
            neighborhood, item_similarity_matrix = sample_generator.create_neighborhood()
            _, val_item_similarity_matrix = sample_generator.create_neighborhood(include_test=True)

            # Specify the exact model
            engine = BPREngine(config)

            # Initialize list of optimal results
            best_performance = [0] * 8

            best_model = ''
            for epoch in range(config['num_epoch']):
                print('Training epoch {}'.format(epoch))
                train_loader = sample_generator.train_data_loader(config['batch_size'])
                engine.train_an_epoch(train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id=epoch)
                if config['loo_eval']:
                    ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(validation_data,
                                                                                      val_explainability_matrix,
                                                                                      val_popularity_vector,
                                                                                      val_item_similarity_matrix,
                                                                                      epoch_id=str(epoch) + ' on val data')
                    print('-' * 80)
                    best_model, best_performance = engine.save_implicit(epoch, ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim, config['num_epoch'], best_model, best_performance, save_models = args.save_models)
                else:
                    map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(validation_data,
                                                                                       val_explainability_matrix,
                                                                                       val_popularity_vector,
                                                                                       val_item_similarity_matrix,
                                                                                       epoch_id=str(epoch) + ' on val data')
                    print('-' * 80)
                    best_model, best_performance = engine.save_explicit(epoch, map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim, config['num_epoch'], best_model, best_performance, save_models = args.save_models)

            # Save results to dataframe
            if config['loo_eval']:
                results = results.append(
                    {'latent': config['num_latent'], 'batch_size': config['batch_size'], 'l2_reg': config['l2_regularization'],
                     'neighborhood': config['neighborhood'], 'rep': rep, 'ndcg': best_performance[0],
                     'hr': best_performance[1], 'mep': best_performance[2], 'wmep': best_performance[3], 'avg_pop': best_performance[4], 'efd': best_performance[5], 'avg_pair_sim': best_performance[6]},
                    ignore_index=True)
            else:
                results = results.append(
                    {'latent': config['num_latent'], 'batch_size': config['batch_size'], 'l2_reg': config['l2_regularization'],
                     'neighborhood': config['neighborhood'], 'rep': rep, 'ndcg': best_performance[1],
                     'map': best_performance[0], 'mep': best_performance[2], 'wmep': best_performance[3], 'avg_pop': best_performance[4], 'efd': best_performance[5], 'avg_pair_sim': best_performance[6]},
                    ignore_index=True)

    # Save dataframe
    print(results)
    if args.save_results:
        results.to_csv('Output/Hyperparameter_tuning_' + model_name + '_' + dataset_name + '.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument("--model", type =str, default='EBPR', help="Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', "
                                                                   "'UEBPR'.")
    parser.add_argument("--dataset", type =str, default='ml-100k', help="'ml-100k' for Movielens 100K. 'ml-1m' for "
                                                                        "the Movielens 1M dataset. 'lastfm-2k' for "
                                                                        "the Last.FM 2K dataset. 'yahoo-r3' for the "
                                                                        "Yahoo! R3 dataset.")
    parser.add_argument("--num_configurations", type=int, default=7, help="Number of random hyperparameter "
                                                                          "configurations.")
    parser.add_argument("--num_reps", type=int, default=3, help="Number of replicates per hyperparameter configuration.")
    parser.add_argument("--num_epoch", type =int, default=50, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient.")
    parser.add_argument("--neighborhood", type=int, default=20, help="Neighborhood size for explainability.")
    parser.add_argument("--top_k", type=int, default=10, help="Cutoff k in MAP@k, HR@k and NDCG@k, etc.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer: 'adam', 'sgd', 'rmsprop'.")
    parser.add_argument("--sgd_momentum", type =float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--rmsprop_alpha", type =float, default=0.9, help="alpha hyperparameter for RMSProp optimizer.")
    parser.add_argument("--rmsprop_momentum", type =float, default=0.0, help="Momentum for RMSProp optimizer.")
    parser.add_argument("--loo_eval", default=True, type=lambda x: (str(x).lower() == 'true'), help="True: LOO evaluation. False: Random "
                                                                            "train/test split")
    parser.add_argument("--test_rate", type=float, default=0.1, help="Test rate for random train/val/test "
                                                                            "split. test_rate is the rate of test and "
                                                                            "validation. Used when 'loo_eval' is set "
                                                                            "to False.")
    parser.add_argument("--use_cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="True if you want to use a CUDA device.")
    parser.add_argument("--device_id", type=int, default=0, help="ID of CUDA device if 'use_cuda' is True.")
    parser.add_argument("--save_models", type=lambda x: (str(x).lower() == 'true'), default=True, help="True if you want to save the best model(s).")
    parser.add_argument("--save_results", type=lambda x: (str(x).lower() == 'true'), default=True, help="True if you want to save the results in a csv file.")

    args = parser.parse_args()
    main(args)
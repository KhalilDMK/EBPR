from Code.EBPR_model import BPREngine
from Code.data import SampleGenerator, read_data
import argparse

def main(args):
    # Read dataset
    dataset_name = args.dataset  # 'ml-100k' for Movielens 100K. 'ml-1m' for the Movielens 1M dataset. 'lastfm-2k' for the
    # Last.FM 2K dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
    dataset = read_data(dataset_name, args.int_per_item)

    # Define hyperparameters
    config = {'model': args.model,  # Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'.
              'dataset': dataset_name,
              'num_epoch': args.num_epoch,  # Number of training epochs.
              'batch_size': args.batch_size,  # Batch size.
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
              'num_latent': args.num_latent,  # Number of latent factors.
              'weight_decay': args.weight_decay,
              'l2_regularization': args.l2_regularization,
              'use_cuda': args.use_cuda,
              'device_id': args.device_id,
              'top_k': args.top_k,  # k in MAP@k, HR@k and NDCG@k.
              'loo_eval': args.loo_eval,  # True: LOO evaluation with HR@k and NDCG@k. False: Random train/test split
              # evaluation with MAP@k and NDCG@k.
              'neighborhood': args.neighborhood,  # Neighborhood size for explainability.
              'model_dir_explicit':'Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model',
              'model_dir_implicit':'Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model'}

    # DataLoader
    sample_generator = SampleGenerator(dataset, config, split_val=False)
    test_data = sample_generator.test_data_loader(config['batch_size'])

    # Create explainability matrix
    explainability_matrix = sample_generator.create_explainability_matrix()
    test_explainability_matrix = sample_generator.create_explainability_matrix(include_test=True)

    # Create popularity vector
    popularity_vector = sample_generator.create_popularity_vector()
    test_popularity_vector = sample_generator.create_popularity_vector(include_test=True)

    #Create item neighborhood
    neighborhood, item_similarity_matrix = sample_generator.create_neighborhood()
    _, test_item_similarity_matrix = sample_generator.create_neighborhood(include_test=True)

    # Specify the exact model
    engine = BPREngine(config)

    # Initialize list of optimal results
    best_performance = [0] * 8
    best_ndcg = 0

    best_model = ''
    for epoch in range(config['num_epoch']):
        print('Training epoch {}'.format(epoch))
        train_loader = sample_generator.train_data_loader(config['batch_size'])
        engine.train_an_epoch(train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id=epoch)
        if config['loo_eval']:
            ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(test_data, test_explainability_matrix, test_popularity_vector, test_item_similarity_matrix, epoch_id=str(epoch) + ' on test data')
            print('-' * 80)
            best_model, best_performance = engine.save_implicit(epoch, ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim, config['num_epoch'], best_model, best_performance, save_models = args.save_models)
        else:
            map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(test_data, test_explainability_matrix, test_popularity_vector, test_item_similarity_matrix, epoch_id=str(epoch) + ' on test data')
            print('-' * 80)
            best_model, best_performance = engine.save_explicit(epoch, map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim, config['num_epoch'], best_model, best_performance, save_models = args.save_models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument("--model", type =str, default='EBPR', help="Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', "
                                                                   "'UEBPR'.")
    parser.add_argument("--dataset", type =str, default='ml-100k', help="'ml-100k' for Movielens 100K. 'ml-1m' for "
                                                                        "the Movielens 1M dataset. 'lastfm-2k' for "
                                                                        "the Last.FM 2K dataset. 'yahoo-r3' for the "
                                                                        "Yahoo! R3 dataset.")
    parser.add_argument("--num_epoch", type =int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type =int, default=100, help="Batch size.")
    parser.add_argument("--num_latent", type=int, default=50, help="Number of latent features.")
    parser.add_argument("--l2_regularization", type=float, default=0.0, help="L2 regularization coefficient.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient.")
    parser.add_argument("--neighborhood", type=int, default=20, help="Neighborhood size for explainability.")
    parser.add_argument("--top_k", type=int, default=10, help="Cutoff k in MAP@k, HR@k and NDCG@k, etc.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer: 'adam', 'sgd', 'rmsprop'.")
    parser.add_argument("--sgd_momentum", type =float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--rmsprop_alpha", type =float, default=0.9, help="alpha hyperparameter for RMSProp optimizer.")
    parser.add_argument("--rmsprop_momentum", type =float, default=0.0, help="Momentum for RMSProp optimizer.")
    parser.add_argument("--loo_eval", type=lambda x: (str(x).lower() == 'true'), default=True, help="True: LOO evaluation. False: Random "
                                                                            "train/test split")
    parser.add_argument("--test_rate", type=float, default=0.2, help="Test rate for random train/val/test "
                                                                            "split. test_rate is the rate of test + "
                                                                            "validation. Used when 'loo_eval' is set "
                                                                            "to False.")
    parser.add_argument("--use_cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="True is you want to use a CUDA device.")
    parser.add_argument("--device_id", type=int, default=0, help="ID of CUDA device if 'use_cuda' is True.")
    parser.add_argument("--save_models", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="True if you want to save the best model(s).")
    parser.add_argument("--int_per_item", type =int, default=0, help="Minimum number of interactions per item for studying effect sparsity on the lastfm-2k dataset.")

    args = parser.parse_args()
    main(args)
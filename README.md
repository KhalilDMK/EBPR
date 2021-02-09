# EBPR
Pytorch implementation of the paper "Debiased Explainable Pairwise Ranking from Implicit Feedback".

This is an anonymous copy of the private Github repository for The Web Conference submission.

## Environment settings
We use Pytorch 1.7.1.

## Description
This repository includes the code necessary to:
* <b>Train BPR [1], UBPR [2], EBPR, pUEBPR and UEBPR:</b>

```
python -m Code.train_EBPR [-h] [--model MODEL] [--dataset DATASET]
                          [--num_epoch NUM_EPOCH] [--batch_size BATCH_SIZE]
                          [--num_latent NUM_LATENT]
                          [--l2_regularization L2_REGULARIZATION]
                          [--weight_decay WEIGHT_DECAY]
                          [--neighborhood NEIGHBORHOOD] [--top_k TOP_K] [--lr LR]
                          [--optimizer OPTIMIZER] [--sgd_momentum SGD_MOMENTUM]
                          [--rmsprop_alpha RMSPROP_ALPHA]
                          [--rmsprop_momentum RMSPROP_MOMENTUM]
                          [--loo_eval LOO_EVAL] [--test_rate TEST_RATE]
                          [--use_cuda USE_CUDA] [--device_id DEVICE_ID]
                          [--save_models SAVE_MODELS] [--int_per_item INT_PER_ITEM]
```

The code is set up to train EBPR on the Movielens 100K dataset. You can change the model using the "model" argument. Also, you can change the "dataset" argument to choose between the "Movielens 100K", "Movielens 1M", "Yahoo! R3" or "Last.FM 2K" datasets. The model will train and output the NDCG@K, HR@K, MEP@K and WMEP@K results on the test set for every epoch using the Leave-One-Out (LOO) evaluation procedure. You can choose the standard random train/test split by changing the parameter "loo_eval" in the "config" dictionary. The list of arguments is presented below:

```
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR',
                        'UEBPR'.
  --dataset DATASET     'ml-100k' for Movielens 100K. 'ml-1m' for the
                        Movielens 1M dataset. 'lastfm-2k' for the Last.FM 2K
                        dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
  --num_epoch NUM_EPOCH
                        Number of training epochs.
  --batch_size BATCH_SIZE
                        Batch size.
  --num_latent NUM_LATENT
                        Number of latent features.
  --l2_regularization L2_REGULARIZATION
                        L2 regularization coefficient.
  --weight_decay WEIGHT_DECAY
                        Weight decay coefficient.
  --neighborhood NEIGHBORHOOD
                        Neighborhood size for explainability.
  --top_k TOP_K         Cutoff k in MAP@k, HR@k and NDCG@k, etc.
  --lr LR               Learning rate.
  --optimizer OPTIMIZER
                        Optimizer: 'adam', 'sgd', 'rmsprop'.
  --sgd_momentum SGD_MOMENTUM
                        Momentum for SGD optimizer.
  --rmsprop_alpha RMSPROP_ALPHA
                        alpha hyperparameter for RMSProp optimizer.
  --rmsprop_momentum RMSPROP_MOMENTUM
                        Momentum for RMSProp optimizer.
  --loo_eval LOO_EVAL   True: LOO evaluation. False: Random train/test split
  --test_rate TEST_RATE
                        Test rate for random train/val/test split. test_rate
                        is the rate of test + validation. Used when 'loo_eval'
                        is set to False.
  --use_cuda USE_CUDA   True is you want to use a CUDA device.
  --device_id DEVICE_ID
                        ID of CUDA device if 'use_cuda' is True.
  --save_models SAVE_MODELS
                        True if you want to save the best model(s).
  --int_per_item INT_PER_ITEM
                        Minimum number of interactions per item for studying
                        effect sparsity on the lastfm-2k dataset.
```


* <b>Tune the hyperparameters of the models:</b>

```
python -m Code.hyperparameter_tuning [-h] [--model MODEL] [--dataset DATASET]
                                     [--num_configurations NUM_CONFIGURATIONS]
                                     [--num_reps NUM_REPS] [--num_epoch NUM_EPOCH]
                                     [--weight_decay WEIGHT_DECAY]
                                     [--neighborhood NEIGHBORHOOD] [--top_k TOP_K]
                                     [--lr LR] [--optimizer OPTIMIZER]
                                     [--sgd_momentum SGD_MOMENTUM]
                                     [--rmsprop_alpha RMSPROP_ALPHA]
                                     [--rmsprop_momentum RMSPROP_MOMENTUM]
                                     [--loo_eval LOO_EVAL] [--test_rate TEST_RATE]
                                     [--use_cuda USE_CUDA] [--device_id DEVICE_ID]
                                     [--save_models SAVE_MODELS]
                                     [--save_results SAVE_RESULTS]
                                     [--int_per_item INT_PER_ITEM]
```

Similarly, you can choose the dataset and the model. The code is set to perform a random hyperparameter tuning as presented in the paper. You can choose the number of experiments and replicates of each experiment. The hyperparameters tuned are the number of latent features, batch size and l2 regularization. The list of arguments is presented below:

```
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR',
                        'UEBPR'.
  --dataset DATASET     'ml-100k' for Movielens 100K. 'ml-1m' for the
                        Movielens 1M dataset. 'lastfm-2k' for the Last.FM 2K
                        dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
  --num_configurations NUM_CONFIGURATIONS
                        Number of random hyperparameter configurations.
  --num_reps NUM_REPS   Number of replicates per hyperparameter configuration.
  --num_epoch NUM_EPOCH
                        Number of training epochs.
  --weight_decay WEIGHT_DECAY
                        Weight decay coefficient.
  --neighborhood NEIGHBORHOOD
                        Neighborhood size for explainability.
  --top_k TOP_K         Cutoff k in MAP@k, HR@k and NDCG@k, etc.
  --lr LR               Learning rate.
  --optimizer OPTIMIZER
                        Optimizer: 'adam', 'sgd', 'rmsprop'.
  --sgd_momentum SGD_MOMENTUM
                        Momentum for SGD optimizer.
  --rmsprop_alpha RMSPROP_ALPHA
                        alpha hyperparameter for RMSProp optimizer.
  --rmsprop_momentum RMSPROP_MOMENTUM
                        Momentum for RMSProp optimizer.
  --loo_eval LOO_EVAL   True: LOO evaluation. False: Random train/test split
  --test_rate TEST_RATE
                        Test rate for random train/val/test split. test_rate
                        is the rate of test and validation. Used when
                        'loo_eval' is set to False.
  --use_cuda USE_CUDA   True if you want to use a CUDA device.
  --device_id DEVICE_ID
                        ID of CUDA device if 'use_cuda' is True.
  --save_models SAVE_MODELS
                        True if you want to save the best model(s).
  --save_results SAVE_RESULTS
                        True if you want to save the results in a csv file.
  --int_per_item INT_PER_ITEM
                        Minimum number of interactions per item for studying
                        effect sparsity on the lastfm-2k dataset.
```

## Datasets
We provide code ready to run on the:
* Movielens 100K dataset.
* Movielens 1M dataset.
* Last.FM 2K dataset.
* Yahoo! R3 dataset.

Note that, due to a privacy consent, you need to add the Yahoo! R3 dataset in a folder "Data/yahoo-r3" to be able to use it.

## References
[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).<br>
[2] Saito, Yuta. "Unbiased Pairwise Learning from Implicit Feedback." NeurIPS 2019 Workshop on Causal Machine Learning. 2019.

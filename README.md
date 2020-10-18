# EBPR
Pytorch implementation of the paper "Investigating Explainability and Exposure Bias in Pairwise Ranking from Implicit Feedback".

This is an anonymous copy of the private Github repository for The Web Conference submission.

## Environment settings
We use Pytorch 1.1.0.

## Description
This repository includes the code necessary to:
* <b>Train BPR [1], UBPR [2], EBPR, pUEBPR and UEBPR:</b>
Run "train_EBPR.py". The code is set up to train EBPR on the Movielens 100K dataset. You can change the "dataset_name" variable to choose between the "Movielens 100K" or the "Last.FM 2K" datasets. You can also change the "config" dictionary to choose the model you would like to train and its hyperparameters. The model will train and output the NDCG@K, HR@K, MEP@K and WMEP@K results on the test set for every epoch using the Leave-One-Out (LOO) evaluation procedure. You can choose the standard random train/test split by changing the parameter "loo_eval" in the "config" dictionary.
* <b>Tune the hyperparameters of the models:</b>
Run "hyperparameter_tuning.py". Similarly, you can choose the dataset and the model. The code is set to do a random hyperparameter tuning. You can choose the number of experiments and replicates of each experiment. The hyperparameters tuned are the number of latent features, batch size, l2 regularization and neighborhood size. You can choose the lists of values to try for these hyperparameters.

## Datasets
We provide code ready to run on the:
* Movielens 100K dataset.
* Movielens 1M dataset.
* Last.FM 2K dataset.
* Yahoo! R3 dataset.

Note that you need to add the Yahoo! R3 dataset in a folder "Data/yahoo-r3" to be able to use it.

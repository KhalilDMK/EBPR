import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

random.seed(0)

def read_data(dataset_name):
    """Read dataset"""

    dataset = pd.DataFrame()
    if dataset_name == 'ml-100k':
        # Load Movielens 100K Data
        ml1m_dir = '../Data/ml-100k/u.data'
        dataset = pd.read_csv(ml1m_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                  engine='python')
    elif dataset_name == 'ml-1m':
        # Load Movielens 1M Data
        ml1m_dir = '../Data/ml-1m/ratings.dat'
        dataset = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    # Reindex data
    user_id = dataset[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    dataset = pd.merge(dataset, user_id, on=['uid'], how='left')
    item_id = dataset[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    dataset = pd.merge(dataset, item_id, on=['mid'], how='left')
    dataset = dataset[['userId', 'itemId', 'rating', 'timestamp']]

    return dataset


class data_loader(Dataset):
    """Convert user, item, negative and target Tensors into Pytorch Dataset"""

    def __init__(self, user_tensor, positive_item_tensor, negative_item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.positive_item_tensor = positive_item_tensor
        self.negative_item_tensor = negative_item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.positive_item_tensor[index], self.negative_item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class data_loader_implicit(Dataset):
    """Convert user and item Tensors into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class data_loader_test_explicit(Dataset):
    """Convert user, item and target Tensors into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class data_loader_negatives(Dataset):
    """Convert user and item negative Tensors into Pytorch Dataset"""

    def __init__(self, user_neg_tensor, item_neg_tensor):
        self.user_neg_tensor = user_neg_tensor
        self.item_neg_tensor = item_neg_tensor

    def __getitem__(self, index):
        return self.user_neg_tensor[index], self.item_neg_tensor[index]

    def __len__(self):
        return self.user_neg_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset"""

    def __init__(self, ratings, config):
        """
        args:
            ratings: pd.DataFrame containing 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
            config: dictionary containing the configuration hyperparameters
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.config = config
        self.ratings = ratings
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples
        self.negatives = self._sample_negative(ratings)
        if self.config['loo_eval']:
            self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
        else:
            self.test_rate = self.config['test_rate']
            self.train_ratings, self.test_ratings = self.train_test_split_random(self.preprocess_ratings)  # also try train_test_split_latest

    def _binarize(self, ratings):
        """binarize into 0 or 1 for imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'] = 1.0
        return ratings

    def train_test_split_latest(self, ratings):
        """First (100 *  (1 - test_rate))%/Last (test_rate * 100)% (for every user) Train/test split"""
        ratings['rank'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        rating_count = ratings.groupby(['userId'])['timestamp'].count()
        ratings['num_ratings'] = [rating_count[user] for user in ratings['userId']]
        ratings['rank/num_ratings'] = ratings['rank'] / ratings['num_ratings']
        train = ratings[ratings['rank/num_ratings'] <= (1 - self.test_rate)]
        print('Train size: ' + str(len(train)))
        test = ratings[ratings['rank/num_ratings'] > (1 - self.test_rate)]
        print('Test size: ' + str(len(test)))
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def train_test_split_random(self, ratings):
        """Random train/test split"""
        train, test = train_test_split(ratings, test_size=self.test_rate)
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _split_loo(self, ratings):
        """leave-one-out train/test split"""
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 100))
        interact_status['negative_items'] = interact_status.apply(lambda x: (x.negative_items - set(x.negative_samples)), axis=1)
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def train_data_loader(self, batch_size):
        """instance train loader for one training epoch"""
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        users = [int(x) for x in train_ratings['userId']]
        items = [int(x) for x in train_ratings['itemId']]
        ratings = [float(x) for x in train_ratings['rating']]
        neg_items = [random.choice(list(neg_list)) for neg_list in train_ratings['negative_items']]
        dataset = data_loader(user_tensor=torch.LongTensor(users),
                              positive_item_tensor=torch.LongTensor(items),
                              negative_item_tensor=torch.LongTensor(neg_items),
                              target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_data_loader(self, batch_size):
        """create evaluation data"""
        if self.config['loo_eval']:
            test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
            test_users, test_items, negative_users, negative_items = [], [], [], []
            for row in test_ratings.itertuples():
                test_users.append(int(row.userId))
                test_items.append(int(row.itemId))
                for i in range(len(row.negative_samples)):
                    negative_users.append(int(row.userId))
                    negative_items.append(int(row.negative_samples[i]))
            dataset = data_loader_implicit(user_tensor=torch.LongTensor(test_users),
                                           item_tensor=torch.LongTensor(test_items))
            dataset_negatives = data_loader_negatives(user_neg_tensor=torch.LongTensor(negative_users),
                                                      item_neg_tensor=torch.LongTensor(negative_items))
            return [DataLoader(dataset, batch_size=batch_size, shuffle=False), DataLoader(dataset_negatives, batch_size=50, shuffle=False)]
        else:
            test_ratings = self.test_ratings
            test_users = [int(x) for x in test_ratings['userId']]
            test_items = [int(x) for x in test_ratings['itemId']]
            test_ratings = [float(x) for x in test_ratings['rating']]
            dataset = data_loader_test_explicit(user_tensor=torch.LongTensor(test_users),
                                                item_tensor=torch.LongTensor(test_items),
                                                target_tensor=torch.FloatTensor(test_ratings))
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
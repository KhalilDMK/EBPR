import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

random.seed(1)

def read_data(dataset_name, int_per_item):
    """Read dataset"""

    dataset = pd.DataFrame()
    if dataset_name == 'ml-100k':
        # Load Movielens 100K Data
        data_dir = 'Data/ml-100k/u.data'
        dataset = pd.read_csv(data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                  engine='python')
    elif dataset_name == 'ml-1m':
        # Load Movielens 1M Data
        data_dir = 'Data/ml-1m/ratings.dat'
        dataset = pd.read_csv(data_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    elif dataset_name == 'lastfm-2k':
        # Load Last.FM 2K Data
        data_dir = 'Data/lastfm-2k/user_artists.dat'
        dataset = pd.read_csv(data_dir, sep='\t', header=0, names=['uid', 'mid', 'rating'],  engine='python')
        dataset['timestamp'] = [1 for i in range(len(dataset))]
        # Filtering items with more than int_per_item interactions
        item_count = dataset[['uid', 'mid']].groupby('mid').count()['uid'].rename('count').reset_index()
        dataset = dataset.merge(item_count, how='left', on='mid')
        dataset = dataset.loc[dataset['count'] >= int_per_item][['uid', 'mid', 'rating', 'timestamp']]
        # Filtering users with more than 10 interactions
        user_count = dataset[['uid', 'mid']].groupby('uid').count()['mid'].rename('count').reset_index()
        dataset = dataset.merge(user_count, how='left', on='uid')
        dataset = dataset.loc[dataset['count'] >= 10][['uid', 'mid', 'rating', 'timestamp']]

    elif dataset_name == 'yahoo-r3':
        # Load Yahoo! R3 Data
        data_dir = 'Data/yahoo-r3/ydata-ymusic-rating-study-v1_0-train.txt'
        dataset = pd.read_csv(data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating'],  engine='python')
        dataset['timestamp'] = [1 for i in range(len(dataset))]

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

    def __init__(self, ratings, config, split_val):
        """
        args:
            ratings: pd.DataFrame containing 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
            config: dictionary containing the configuration hyperparameters
            split_val: boolean that takes True if we are using a validation set
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.config = config
        self.ratings = ratings
        self.split_val = split_val
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples
        self.negatives = self._sample_negative(ratings, self.split_val)
        if self.config['loo_eval']:
            if self.split_val:
                self.train_ratings, self.val_ratings = self._split_loo(self.preprocess_ratings, split_val=True)
            else:
                self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings, split_val=False)
        else:
            self.test_rate = self.config['test_rate']
            if self.split_val:
                self.train_ratings, self.val_ratings = self.train_test_split_random(self.preprocess_ratings, split_val=True)
            else:
                self.train_ratings, self.test_ratings = self.train_test_split_random(self.preprocess_ratings, split_val=False)

    def _binarize(self, ratings):
        """binarize into 0 or 1 for imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'] = 1.0
        return ratings

    def train_test_split_random(self, ratings, split_val):
        """Random train/test split"""
        train, test = train_test_split(ratings, test_size=self.test_rate)
        if split_val:
            train, val = train_test_split(train, test_size=self.test_rate / (1 - self.test_rate))
            return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']]
        else:
            return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _split_loo(self, ratings, split_val):
        """leave-one-out train/test split"""
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        if split_val:
            val = ratings[ratings['rank_latest'] == 2]
            train = ratings[ratings['rank_latest'] > 2]
            assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
            return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']]
        else:
            train = ratings[ratings['rank_latest'] > 1]
            assert train['userId'].nunique() == test['userId'].nunique()
            return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings, split_val):
        """return all negative items & 100 sampled negative test items & 100 sampled negative val items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['test_negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 100))
        interact_status['negative_items'] = interact_status.apply(lambda x: (x.negative_items - set(x.test_negative_samples)), axis=1)
        if split_val:
            interact_status['val_negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 100))
            interact_status['negative_items'] = interact_status.apply(lambda x: (x.negative_items - set(x.val_negative_samples)), axis=1)
            return interact_status[['userId', 'negative_items', 'test_negative_samples', 'val_negative_samples']]
        else:
            return interact_status[['userId', 'negative_items', 'test_negative_samples']]

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
            test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'test_negative_samples']], on='userId')
            test_users, test_items, negative_users, negative_items = [], [], [], []
            for row in test_ratings.itertuples():
                test_users.append(int(row.userId))
                test_items.append(int(row.itemId))
                for i in range(len(row.test_negative_samples)):
                    negative_users.append(int(row.userId))
                    negative_items.append(int(row.test_negative_samples[i]))
            dataset = data_loader_implicit(user_tensor=torch.LongTensor(test_users),
                                           item_tensor=torch.LongTensor(test_items))
            dataset_negatives = data_loader_negatives(user_neg_tensor=torch.LongTensor(negative_users),
                                                      item_neg_tensor=torch.LongTensor(negative_items))
            return [DataLoader(dataset, batch_size=batch_size, shuffle=False), DataLoader(dataset_negatives, batch_size=batch_size, shuffle=False)]
        else:
            test_ratings = self.test_ratings
            test_users = [int(x) for x in test_ratings['userId']]
            test_items = [int(x) for x in test_ratings['itemId']]
            test_ratings = [float(x) for x in test_ratings['rating']]
            dataset = data_loader_test_explicit(user_tensor=torch.LongTensor(test_users),
                                                item_tensor=torch.LongTensor(test_items),
                                                target_tensor=torch.FloatTensor(test_ratings))
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def val_data_loader(self, batch_size):
        """create validation data"""
        if self.config['loo_eval']:
            val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'val_negative_samples']], on='userId')
            val_users, val_items, negative_users, negative_items = [], [], [], []
            for row in val_ratings.itertuples():
                val_users.append(int(row.userId))
                val_items.append(int(row.itemId))
                for i in range(len(row.val_negative_samples)):
                    negative_users.append(int(row.userId))
                    negative_items.append(int(row.val_negative_samples[i]))
            dataset = data_loader_implicit(user_tensor=torch.LongTensor(val_users),
                                           item_tensor=torch.LongTensor(val_items))
            dataset_negatives = data_loader_negatives(user_neg_tensor=torch.LongTensor(negative_users),
                                                      item_neg_tensor=torch.LongTensor(negative_items))
            return [DataLoader(dataset, batch_size=batch_size, shuffle=False), DataLoader(dataset_negatives, batch_size=batch_size, shuffle=False)]
        else:
            val_ratings = self.val_ratings
            val_users = [int(x) for x in val_ratings['userId']]
            val_items = [int(x) for x in val_ratings['itemId']]
            val_ratings = [float(x) for x in val_ratings['rating']]
            dataset = data_loader_test_explicit(user_tensor=torch.LongTensor(val_users),
                                                item_tensor=torch.LongTensor(val_items),
                                                target_tensor=torch.FloatTensor(val_ratings))
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def create_explainability_matrix(self, include_test=False):
        """create explainability matrix"""
        if not include_test:
            print('Creating explainability matrix...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId, self.train_ratings.itemId)
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        elif not self.split_val:
            print('Creating test explainability matrix...')
            interaction_matrix = np.array(pd.crosstab(self.preprocess_ratings.userId, self.preprocess_ratings.itemId)[
                                              list(range(self.config['num_items']))].sort_index())
        else:
            print('Creating val explainability matrix...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId.append(self.val_ratings.userId), self.train_ratings.itemId.append(self.val_ratings.itemId))
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        #item_similarity_matrix = 1 - pairwise_distances(interaction_matrix.T, metric = "hamming")
        item_similarity_matrix = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(item_similarity_matrix, 0)
        neighborhood = [np.argpartition(row, - self.config['neighborhood'])[- self.config['neighborhood']:]
                        for row in item_similarity_matrix]
        explainability_matrix = np.array([[sum([interaction_matrix[user, neighbor] for neighbor in neighborhood[item]])
                                           for item in range(self.config['num_items'])] for user in
                                          range(self.config['num_users'])]) / self.config['neighborhood']
        #explainability_matrix[explainability_matrix < 0.1] = 0
        #explainability_matrix = explainability_matrix + self.config['epsilon']
        return explainability_matrix

    def create_popularity_vector(self, include_test=False):
        """create popularity vector"""
        if not include_test:
            print('Creating popularity vector...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId, self.train_ratings.itemId)
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        elif not self.split_val:
            print('Creating test popularity vector...')
            interaction_matrix = np.array(pd.crosstab(self.preprocess_ratings.userId, self.preprocess_ratings.itemId)[
                                              list(range(self.config['num_items']))].sort_index())
        else:
            print('Creating val popularity vector...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId.append(self.val_ratings.userId),
                                             self.train_ratings.itemId.append(self.val_ratings.itemId))
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        popularity_vector = np.sum(interaction_matrix, axis=0)
        popularity_vector = (popularity_vector / max(popularity_vector)) ** 0.5
        return popularity_vector

    def create_neighborhood(self, include_test=False):
        """Determine item neighbors"""
        if not include_test:
            print('Determining item neighborhoods...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId, self.train_ratings.itemId)
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        elif not self.split_val:
            print('Determining test item neighborhoods...')
            interaction_matrix = np.array(pd.crosstab(self.preprocess_ratings.userId, self.preprocess_ratings.itemId)[
                                              list(range(self.config['num_items']))].sort_index())
        else:
            print('Determining val item neighborhoods...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId.append(self.val_ratings.userId),
                                             self.train_ratings.itemId.append(self.val_ratings.itemId))
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        item_similarity_matrix = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(item_similarity_matrix, 0)
        neighborhood = np.array([np.argpartition(row, - self.config['neighborhood'])[- self.config['neighborhood']:]
                        for row in item_similarity_matrix])
        return neighborhood, item_similarity_matrix
import torch
from Code.utils import save_checkpoint, use_optimizer, resume_checkpoint
from Code.metrics import MetronAtK
import pyprind


class Engine(object):
    """Meta Engine for training & evaluating BPR"""

    def __init__(self, config):
        self.config = config
        self._metron = MetronAtK(top_k=config['top_k'], loo_eval=self.config['loo_eval'])
        self.opt = use_optimizer(self.model, config)

    def train_single_batch_EBPR(self, users, pos_items, neg_items, ratings, explainability_matrix, popularity_vector, neighborhood):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        assert self.config['model'] in ['BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'], 'Please specify the right model!'
        if self.config['use_cuda'] is True:
            users, pos_items, neg_items, ratings = users.cuda(), pos_items.cuda(), neg_items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        pos_prediction, neg_prediction = self.model(users, pos_items, neg_items)
        if self.config['model'] == 'BPR':
            loss = - (pos_prediction - neg_prediction).sigmoid().log().sum()
        elif self.config['model'] == 'UBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items]).sum()
        elif self.config['model'] == 'EBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() * explainability_matrix[users, pos_items] * (
                        1 - explainability_matrix[users, neg_items])).sum()
        elif self.config['model'] == 'pUEBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items] *
                      explainability_matrix[users, pos_items] * (1 - explainability_matrix[users, neg_items])).sum()
        elif self.config['model'] == 'UEBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items] *
                      explainability_matrix[users, pos_items] / popularity_vector[
                          neighborhood[pos_items].flatten()].view(len(pos_items), self.config['neighborhood']).sum(
                        1) * (1 - explainability_matrix[users, neg_items] / popularity_vector[
                        neighborhood[neg_items].flatten()].view(len(neg_items), self.config['neighborhood']).sum(
                        1))).sum()
        if self.config['l2_regularization'] > 0:
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += self.config['l2_regularization'] * l2_reg
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.model.train()
        if self.config['use_cuda'] is True:
            explainability_matrix = torch.from_numpy(explainability_matrix).float().cuda()
            popularity_vector = torch.from_numpy(popularity_vector).float().cuda()
            neighborhood = torch.from_numpy(neighborhood).cuda()
        total_loss = 0
        bar = pyprind.ProgBar(len(train_loader))
        for batch_id, batch in enumerate(train_loader):
            bar.update()
            assert isinstance(batch[0], torch.LongTensor)
            user, pos_item, neg_item, rating = batch[0], batch[1], batch[2], batch[3]
            loss = self.train_single_batch_EBPR(user, pos_item, neg_item, rating, explainability_matrix, popularity_vector, neighborhood)
            total_loss += loss

    def evaluate(self, evaluate_data, explainability_matrix, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['loo_eval']:
            test_users_eval, test_items_eval, test_scores_eval, negative_users_eval, negative_items_eval, negative_scores_eval = [], [], [], [], [], []
        else:
            test_users_eval, test_items_eval, test_scores_eval, test_output_eval = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            if self.config['loo_eval']:
                for batch_id, batch in enumerate(evaluate_data[0]):
                    test_users, test_items = batch[0], batch[1]
                    if self.config['use_cuda'] is True:
                        test_users = test_users.cuda()
                        test_items = test_items.cuda()
                    test_scores, _ = self.model(test_users, test_items, test_items)
                    if self.config['use_cuda'] is True:
                        test_users_eval += test_users.cpu().data.view(-1).tolist()
                        test_items_eval += test_items.cpu().data.view(-1).tolist()
                        test_scores_eval += test_scores.cpu().data.view(-1).tolist()
                for batch_id, batch in enumerate(evaluate_data[1]):
                    negative_users, negative_items = batch[0], batch[1]
                    if self.config['use_cuda'] is True:
                        negative_users = negative_users.cuda()
                        negative_items = negative_items.cuda()
                    negative_scores, _ = self.model(negative_users, negative_items, negative_items)
                    if self.config['use_cuda'] is True:
                        negative_users_eval += negative_users.cpu().data.view(-1).tolist()
                        negative_items_eval += negative_items.cpu().data.view(-1).tolist()
                        negative_scores_eval += negative_scores.cpu().data.view(-1).tolist()
                self._metron.subjects = [test_users_eval, test_items_eval, test_scores_eval, negative_users_eval,
                                         negative_items_eval, negative_scores_eval]
                hr, ndcg, mep, wmep = self._metron.cal_hit_ratio_loo(), self._metron.cal_ndcg_loo(), self._metron.cal_mep(explainability_matrix, theta=0), self._metron.cal_weighted_mep(explainability_matrix, theta=0)
                print('Evaluating Epoch {}: NDCG@{} = {:.4f}, HR@{} = {:.4f}, MEP@{} = {:.4f}, WMEP@{} = {:.4f}'.format(epoch_id, self.config['top_k'],
                                                                                     ndcg, self.config['top_k'], hr, self.config['top_k'], mep, self.config['top_k'], wmep))
                return ndcg, hr, mep, wmep
            else:
                for batch_id, batch in enumerate(evaluate_data):
                    test_users, test_items, test_output = batch[0], batch[1], batch[2]
                    if self.config['use_cuda'] is True:
                        test_users = test_users.cuda()
                        test_items = test_items.cuda()
                        test_output = test_output.cuda()
                    test_scores, _ = self.model(test_users, test_items, test_items)
                    if self.config['use_cuda'] is True:
                        test_users_eval += test_users.cpu().data.view(-1).tolist()
                        test_items_eval += test_items.cpu().data.view(-1).tolist()
                        test_scores_eval += test_scores.cpu().data.view(-1).tolist()
                        test_output_eval += test_output.cpu().data.view(-1).tolist()
            self._metron.subjects = [test_users_eval, test_items_eval, test_output_eval, test_scores_eval]
            map, ndcg, mep, wmep = self._metron.cal_map_at_k(), self._metron.cal_ndcg(), self._metron.cal_mep(explainability_matrix, theta=0), self._metron.cal_weighted_mep(explainability_matrix, theta=0)
            print('Evaluating Epoch {}: MAP@{} = {:.4f}, NDCG@{} = {:.4f}, MEP@{} = {:.4f}, WMEP@{} = {:.4f}'.format(epoch_id, self.config['top_k'], map, self.config['top_k'], ndcg, self.config['top_k'], mep, self.config['top_k'], wmep))
            return map, ndcg, mep, wmep

    def save_explicit(self, epoch_id, map, ndcg, mep, wmep, num_epoch, best_model, best_performance):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if ndcg > best_performance[1]:
            best_performance[0] = map
            best_performance[1] = ndcg
            best_performance[2] = mep
            best_performance[3] = wmep
            best_performance[4] = epoch_id
            best_model = self.model
        if epoch_id == num_epoch - 1:
            alias = self.config['model'] + '_' + self.config['dataset'] + '_batchsize_' + str(self.config['batch_size']) + '_opt_' + str(self.config['optimizer']) + '_lr_' + str(self.config['lr']) + '_latent_' + str(self.config['num_latent']) + '_l2reg_' + str(self.config['l2_regularization'])
            model_dir = self.config['model_dir_explicit'].format(alias, best_performance[4], self.config['top_k'], best_performance[0], self.config['top_k'], best_performance[1], self.config['top_k'], best_performance[2], self.config['top_k'], best_performance[3])
            save_checkpoint(best_model, model_dir)
        return best_model, best_performance

    def save_implicit(self, epoch_id, ndcg, hr, mep, wmep, num_epoch, best_model, best_performance):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if ndcg > best_performance[0]:
            best_performance[0] = ndcg
            best_performance[1] = hr
            best_performance[2] = mep
            best_performance[3] = wmep
            best_performance[4] = epoch_id
            best_model = self.model
        if epoch_id == num_epoch - 1:
            alias = self.config['model'] + '_' + self.config['dataset'] + '_batchsize_' + str(self.config['batch_size']) + '_opt_' + str(self.config['optimizer']) + '_lr_' + str(self.config['lr']) + '_latent_' + str(self.config['num_latent']) + '_l2reg_' + str(self.config['l2_regularization'])
            model_dir = self.config['model_dir_implicit'].format(alias, best_performance[4], self.config['top_k'], best_performance[1], self.config['top_k'], best_performance[0], self.config['top_k'], best_performance[2], self.config['top_k'], best_performance[3])
            save_checkpoint(best_model, model_dir)
        return best_model, best_performance

    def load_model(self, test_model_path):
        resume_checkpoint(self.model, test_model_path, self.config['device_id'])
        return self.model
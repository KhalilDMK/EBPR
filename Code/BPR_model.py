import torch
from Code.engine_BPR import Engine
from Code.utils import use_cuda


class BPR(torch.nn.Module):
    """"BPR model definition"""

    def __init__(self, config):
        super(BPR, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_latent = config['num_latent']
        self.loo_eval = config['loo_eval']

        self.embed_user = torch.nn.Embedding(self.num_users, self.num_latent)
        self.embed_item = torch.nn.Embedding(self.num_items, self.num_latent)

        # torch.nn.init.xavier_uniform_(self.embed_user.weight)
        # torch.nn.init.xavier_uniform_(self.embed_item.weight)
        torch.nn.init.normal_(self.embed_user.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):

        user_latent = self.embed_user(user_indices)
        pos_item_latent = self.embed_item(pos_item_indices)
        neg_item_latent = self.embed_item(neg_item_indices)

        pos_prediction = (user_latent * pos_item_latent).sum(dim=-1)
        neg_prediction = (user_latent * neg_item_latent).sum(dim=-1)
        return pos_prediction, neg_prediction

    def init_weight(self):
        pass

class BPREngine(Engine):
    """Engine for training & evaluating BPR"""
    def __init__(self, config):
        self.model = BPR(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(BPREngine, self).__init__(config)
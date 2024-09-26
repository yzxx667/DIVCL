# -*- coding: utf-8 -*-

import numpy as np
import torch
import faiss
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(
        self,
    ):
        super(Aggregator, self).__init__()

    def forward(
        self,
        entity_emb,
        user_emb,
        relation_emb,
        edge_index,
        edge_type,
        interact_mat,
        relation_intent_emb=None,
        history_intent_emb=None,
        adj_mat=None,
    ):
        from torch_scatter import scatter_mean

        n_entities = entity_emb.shape[0]

        """user common aggregate"""
        user_agg = torch.sparse.mm(
            interact_mat, entity_emb
        )
        if relation_intent_emb is not None:
            """KG aggregate"""
            head, tail = edge_index
            edge_relation_emb = relation_emb[edge_type]
            neigh_relation_emb = (
                    entity_emb[tail] * edge_relation_emb
            )  # [-1, embedding_size]
            entity_agg = scatter_mean(
                src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
            )

            """relation intent"""
            relation_score_ = torch.mm(user_emb, relation_intent_emb.T)
            relation_score = nn.Softmax(dim=1)(relation_score_)  # [n_users, intents]
            relation_user_agg = (torch.mm(relation_score, relation_intent_emb)) * user_agg + user_agg  # [n_users, embedding_size]
            return entity_agg, relation_user_agg

        elif history_intent_emb is not None:
            """item aggregate"""
            user_index, item_index = adj_mat.nonzero()
            user_index = torch.tensor(user_index).type(torch.long).cuda()
            item_index = torch.tensor(item_index).type(torch.long).cuda()
            neigh_emb = user_emb[user_index]
            entity_agg = scatter_mean(src=neigh_emb, index=item_index, dim_size=n_entities, dim=0)
            """history intent"""
            # item_intent_matrix = torch.matmul(entity_emb, history_intent_emb.T)
            # history_score_ = torch.sparse.mm(interact_mat, item_intent_matrix) # [n_users, intents]
            history_score_ = torch.mm(user_emb, history_intent_emb.T)
            history_score = nn.Softmax(dim=1)(history_score_)  # [n_users, intents]
            history_user_agg = (torch.mm(history_score, history_intent_emb)) * user_agg + user_agg
            return entity_agg, history_user_agg

        # # [n_users, n_factors]
        # score_ = torch.mm(user_emb, intent_emb.t())
        # # [n_factors(argmax_users), embedding_size]
        # argmax_user_indices = torch.argmax(score_,dim=0)
        # argmax_user_emb = user_emb[argmax_user_indices]
        #
        # current_user_pre = nn.Softmax(dim=1)(score_)
        # # [n_users, embedding_size]
        # user_aug = torch.matmul(current_user_pre, argmax_user_emb)
        # user_agg = user_aug + user_agg
        # return entity_agg, user_agg

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        embedding_size,
        n_hops,
        n_users,
        n_items,
        n_factors,
        n_relations,
        edge_index,
        edge_type,
        interact_mat,
        tmp,
        device,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1,
        adj_mat=None,
        threshold=None,
    ):
        super(GraphConv, self).__init__()

        self.embedding_size = embedding_size
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items,
        self.n_factors = n_factors
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.interact_mat = interact_mat
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.threshold = threshold
        self.adj_mat = adj_mat
        self.temperature = tmp
        self.device = device

        # rela
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # KG intent project
        relation_intent_project = nn.init.xavier_uniform_(torch.empty(self.n_factors, self.n_relations))
        self.relation_intent_project = nn.Parameter(relation_intent_project)

        # history intent project
        history_intent = nn.init.xavier_uniform_(torch.empty(self.n_factors, self.embedding_size))
        self.history_intent_embedding = nn.Parameter(history_intent)


        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices]

        # sampling kG
    def _sim_kg_edge_sampling(self, edge_index, edge_type, entity_embeddings, threshold=0.2,CHUNK_SIZE = 5000):
        #project
        intent_project = torch.matmul(self.relation_intent_project, self.relation_embedding.weight)
        intent_project_T = intent_project.T
        head, tail = edge_index
        n_edges = edge_index.shape[1]
        head_project_emb = torch.matmul(entity_embeddings[head], intent_project_T)
        tail_project_emb = torch.matmul(entity_embeddings[tail], intent_project_T)
        # rela_project_emb = relation_embeddings[edge_type]
        similarity_value = []
        CHUNK_SIZE = 5000
        for i in range(0, n_edges, CHUNK_SIZE):
            if i + CHUNK_SIZE <= n_edges:
                head_batch = head_project_emb[i:i + CHUNK_SIZE]
                tail_batch = tail_project_emb[i:i + CHUNK_SIZE]
                # rela_batch = rela_project_emb[i:i + CHUNK_SIZE]
            else:
                head_batch = head_project_emb[i:]
                tail_batch = tail_project_emb[i:]
                # rela_batch = rela_project_emb[i:]
            similarity_batch = torch.nn.functional.cosine_similarity(tail_batch, head_batch, dim=-1)
            similarity_value.append(similarity_batch)
        similarity_value = torch.cat(similarity_value, dim=0)
        # fixed
        similarity_value = torch.div(torch.add(similarity_value,1),2)
        selected_indices = (similarity_value > threshold).nonzero().squeeze()
        return edge_index[:, selected_indices], edge_type[selected_indices]


    def forward(self, user_emb, entity_emb,):
        """node dropout"""
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self._sim_kg_edge_sampling(
                self.edge_index, self.edge_type, entity_emb, self.threshold
            )
            # edge_index, edge_type = self.edge_sampling(
            #    self.edge_index, self.edge_type, self.node_dropout_rate
            # )
            interact_mat = self.node_dropout(self.interact_mat)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat

        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        relation_intent_emb = torch.matmul(self.relation_intent_project, relation_emb)  # [n_factors, embedding_size]
        history_intent_emb = self.history_intent_embedding # [n_factors, embedding_size]

        """history intent"""
        h_i_entity_emb = entity_emb
        h_i_user_emb = user_emb
        h_i_entity_res_emb = entity_emb  # [n_entities, embedding_size]
        h_i_user_res_emb = user_emb  # [n_users, embedding_size]
        for i in range(len(self.convs)):
            h_i_entity_emb, h_i_user_emb = self.convs[i](
                h_i_entity_emb,
                h_i_user_emb,
                relation_emb,
                edge_index,
                edge_type,
                interact_mat,
                relation_intent_emb=None,
                history_intent_emb=history_intent_emb,
                adj_mat=self.adj_mat,
            )
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                h_i_entity_emb = self.mess_dropout(h_i_entity_emb)
                h_i_user_emb = self.mess_dropout(h_i_user_emb)
            h_i_entity_emb = F.normalize(h_i_entity_emb)
            h_i_user_emb = F.normalize(h_i_user_emb)
            """result emb"""
            h_i_entity_res_emb = torch.add(h_i_entity_res_emb, h_i_entity_emb)
            h_i_user_res_emb = torch.add(h_i_user_res_emb, h_i_user_emb)

        r_i_entity_emb = entity_emb
        r_i_user_emb = user_emb
        r_i_entity_res_emb = entity_emb  # [n_entities, embedding_size]
        r_i_user_res_emb = user_emb  # [n_users, embedding_size]
        for i in range(len(self.convs)):
            r_i_entity_emb, r_i_user_emb = self.convs[i](
                r_i_entity_emb,
                r_i_user_emb,
                relation_emb,
                edge_index,
                edge_type,
                interact_mat,
                relation_intent_emb=relation_intent_emb,
                history_intent_emb=None,
                adj_mat=None,
            )
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                r_i_entity_emb = self.mess_dropout(r_i_entity_emb)
                r_i_user_emb = self.mess_dropout(r_i_user_emb)
            r_i_entity_emb = F.normalize(r_i_entity_emb)
            r_i_user_emb = F.normalize(r_i_user_emb)
            """result emb"""
            r_i_entity_res_emb = torch.add(r_i_entity_res_emb, r_i_entity_emb)
            r_i_user_res_emb = torch.add(r_i_user_res_emb, r_i_user_emb)

        entity_res_emb = torch.cat((h_i_entity_res_emb, r_i_entity_res_emb),dim=-1)
        user_res_emb = torch.cat((h_i_user_res_emb, r_i_user_res_emb),dim=-1)
        # entity_res_emb = h_i_entity_res_emb
        #entity_res_emb = r_i_entity_res_emb
        # user_res_emb = h_i_user_res_emb
        #user_res_emb = r_i_user_res_emb

        return (
            entity_res_emb,
            user_res_emb,
            [h_i_entity_res_emb,r_i_entity_res_emb,h_i_user_res_emb,r_i_user_res_emb]
        )

class KGINCL(KnowledgeRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGINCL, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_factors = config["n_factors"]
        self.context_hops = config["context_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.sim_decay = config["sim_regularity"]
        self.reg_weight = config["reg_weight"]
        self.ssl_reg = config["ssl_reg"]
        self.temperature = config["temperature"]
        self.threshold = config["kg_threshold"]

        self.k = config['num_clusters']

        # load dataset info
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32
        )  # [n_users, n_items]
        # inter_matrix: [n_users, n_entities]; inter_graph: [n_users + n_entities, n_users + n_entities]
        self.interact_mat, _ = self.get_norm_inter_matrix(mode="si")
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)


        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)

        self.gcn = GraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_items=self.n_items,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            interact_mat=self.interact_mat,
            tmp=self.temperature,
            device=self.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
            adj_mat=self.inter_matrix,
            threshold=self.threshold,
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_norm_inter_matrix(self, mode="bi"):
        # Get the normalized interaction matrix of users and items.

        def _bi_norm_lap(A):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(A.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(A):
            # D^{-1}A
            rowsum = np.array(A.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(A)
            return norm_adj.tocoo()

        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=np.float32,
        )
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        if mode == "bi":
            L = _bi_norm_lap(A)
        elif mode == "si":
            L = _si_norm_lap(A)
        else:
            raise NotImplementedError(
                f"Normalize mode [{mode}] has not been implemented."
            )
        # covert norm_inter_graph to tensor
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        norm_graph = torch.sparse.FloatTensor(i, data, L.shape)

        # interaction: user->item, [n_users, n_entities]
        L_ = L.tocsr()[: self.n_users, self.n_users :].tocoo()
        # covert norm_inter_matrix to tensor
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        norm_matrix = torch.sparse.FloatTensor(i_, data_, L_.shape)

        return norm_matrix.to(self.device), norm_graph.to(self.device)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def forward(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        # entity_gcn_emb: [n_entities, embedding_size]
        # user_gcn_emb: [n_users, embedding_size]
        # latent_gcn_emb: [n_factors, embedding_size]
        entity_gcn_emb, user_gcn_emb, emb_list = self.gcn(
            user_embeddings, entity_embeddings
        )

        return user_gcn_emb, entity_gcn_emb, emb_list

    def ProtoNCE_loss(self, user_embeddings_all, item_embeddings_all, user, item):

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]  # [B,]
        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.temperature)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.temperature).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.temperature)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.temperature).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        # proto_nce_loss = self.proto_reg * (proto_nce_loss_user)
        return proto_nce_loss

    def cul_ssl_loss(self, x, y):
        nrom_x = F.normalize(x)
        nrom_y = F.normalize(y)
        pos_score_ = torch.mul(nrom_x, nrom_y).sum(dim=1)
        pos_score_ = torch.exp(pos_score_ / self.temperature)
        ttl_score_ = torch.matmul(nrom_x, nrom_y.transpose(0, 1))
        ttl_score_ = torch.exp(ttl_score_ / self.temperature).sum(dim=1)

        nce_loss_ = -torch.log(pos_score_ / ttl_score_).sum()
        return nce_loss_

    def cul_ssl_loss2(self, x, y):
        f = lambda x:torch.exp(x / self.temperature)
        A = F.normalize(x)
        B = F.normalize(y)
        refl_sim = f(torch.mm(A, A.t()))
        between_sim = f(torch.mm(A, B.t()))
        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

        refl_sim_1 = f(torch.mm(B, B.t()))
        between_sim_1 = f(torch.mm(B, A.t()))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag())
        )
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()

        return ret


    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings, emb_list = self.forward()

        h_i_entity_res_emb, r_i_entity_res_emb, h_i_user_res_emb, r_i_user_res_emb = emb_list
        # ssl_user_loss = self.cul_ssl_loss(h_i_user_res_emb[user],r_i_user_res_emb[user])
        # ssl_item_loss = self.cul_ssl_loss(h_i_entity_res_emb[pos_item],r_i_entity_res_emb[pos_item])

        ssl_user_loss = self.cul_ssl_loss2(h_i_user_res_emb[user],r_i_user_res_emb[user])
        ssl_item_loss = self.cul_ssl_loss2(h_i_entity_res_emb[pos_item], r_i_entity_res_emb[pos_item])
        ssl_loss = ssl_user_loss + ssl_item_loss

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        # calculate BPR loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        return mf_loss + self.reg_weight * reg_loss + self.ssl_reg * ssl_loss, self.ssl_reg * ssl_loss
        # return mf_loss + self.reg_weight * reg_loss


    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings, _ = self.forward()
        #u_emb = user_all_embeddings[:,:self.embedding_size]
        u_emb = user_all_embeddings
        #e_emb = entity_all_embeddings[:, :self.entity_embedding]
        e_emb = entity_all_embeddings
        u_embeddings = u_emb[user]
        i_embeddings = e_emb[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e,_ = self.forward()
        #u_embeddings = self.restore_user_e[user, : self.embedding_size]
        u_embeddings = self.restore_user_e[user]
        #i_embeddings = self.restore_entity_e[: self.n_items,: self.embedding_size]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
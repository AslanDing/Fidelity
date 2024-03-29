import random

import torch
import numpy as np
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod
from torch_geometric.utils import k_hop_subgraph

from itertools import combinations

class BaseExplainer(ABC):
    def __init__(self, model_to_explain, graphs, features, task,max_length = 50):
        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features
        self.type = task
        self.undirect = True
        self.max_length = max_length

    def set_undirect(self,undirect=False):
        self.undirect = undirect

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass

    def cal_fid(self, index, graph, explain, label):

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            # graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph)[index],dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            # graph = self.graphs[index].detach()
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph),dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()

        # add mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                         edge_weights=torch.sigmoid(explain).to(feats.device))[index],dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                          edge_weights=torch.sigmoid(explain).to(feats.device)),dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()

        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax( self.model_to_explain(feats, graph, edge_weights=(1 - torch.sigmoid(explain).to(feats.device)))[
                        index],dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax( self.model_to_explain(feats, graph,
                                    edge_weights=(1 - torch.sigmoid(explain).to(feats.device))),dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()

                fid_plus = original_pred[:,label] - mask_pred_plus[:,label]
                fid_minus = original_pred[:,label] - mask_pred_minus[:,label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return fid_plus, fid_minus, fid_plus_label, fid_minus_label

    def cal_fid_gt(self, index, graph, gt_graph, label):

        if self.type == 'node':
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][0] #.cpu().numpy()
            matrix_1 = gt_graph[0][1] #.cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (gt_graph[1],
                     (matrix_0,matrix_1)),shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph,matrix_1_graph].A[0]
            explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()
        else:
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][index][0]  # .cpu().numpy()
            matrix_1 = gt_graph[0][index][1]  # .cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (gt_graph[1][index],
                 (matrix_0, matrix_1)), shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph, matrix_1_graph].A[0]
            explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()

        # print("sparsity: ", 1 - sparsity)

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            # graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph)[index],dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            # graph = self.graphs[index].detach()
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph),dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
                # add mask

        if self.type == 'node':
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_eval(feats, graph,
                                                                      edge_weights=explain.to(
                                                                          feats.device))[index], dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_eval(feats, graph,
                                                                      edge_weights=explain.to(
                                                                          feats.device)), dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()
        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax(self.model_eval(feats, graph, edge_weights=(
                                1 - explain.to(feats.device)))[
                                      index], dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax(self.model_eval(feats, graph,
                                                                     edge_weights=(
                                                                                 1 - explain.to(
                                                                             feats.device))), dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()

                fid_plus = original_pred[:, label] - mask_pred_plus[:, label]
                fid_minus = original_pred[:, label] - mask_pred_minus[:, label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return fid_plus, fid_minus, fid_plus_label, fid_minus_label, 1- sparsity

    def cal_fid_edit_distance(self, index, graph, gt_graph, ed_weight, label):

        if self.type == 'node':
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][0] #.cpu().numpy()
            matrix_1 = gt_graph[0][1] #.cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (ed_weight,
                     (matrix_0,matrix_1)),shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph,matrix_1_graph].A[0]
            explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()
        else:
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][index][0]  # .cpu().numpy()
            matrix_1 = gt_graph[0][index][1]  # .cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (ed_weight,
                 (matrix_0, matrix_1)), shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph, matrix_1_graph].A[0]
            explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()

        # print("sparsity: ", 1 - sparsity)

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            # graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph)[index],dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            # graph = self.graphs[index].detach()
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph),dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
                # add mask

        if self.type == 'node':
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_eval(feats, graph,
                                                                      edge_weights=explain.to(
                                                                          feats.device))[index], dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_eval(feats, graph,
                                                                      edge_weights=explain.to(
                                                                          feats.device)), dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()
        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax(self.model_eval(feats, graph, edge_weights=(
                                1 - explain.to(feats.device)))[
                                      index], dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax(self.model_eval(feats, graph,
                                                                     edge_weights=(
                                                                                 1 - explain.to(
                                                                             feats.device))), dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()

                fid_plus = original_pred[:, label] - mask_pred_plus[:, label]
                fid_minus = original_pred[:, label] - mask_pred_minus[:, label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return fid_plus, fid_minus, fid_plus_label, fid_minus_label, 1- sparsity , weights

    def cal_fid_sparsity(self, index, graph, explain, label,sparsity = 0.5):

        retain = 1 - sparsity
        explain_retain = torch.kthvalue(explain,int(explain.shape[0]*sparsity)).values
        explain_01 = torch.where(explain > explain_retain,torch.ones_like(explain),torch.zeros_like(explain))
        explain = explain_01
        sparsity = torch.sum(explain)/torch.ones_like(explain).sum()
        # print("sparsity: ", 1 - sparsity)

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            # graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph)[index],dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            # graph = self.graphs[index].detach()
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph),dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()

        # add mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_eval(feats, graph,
                         edge_weights=explain.to(feats.device))[index],dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_eval(feats, graph,
                          edge_weights=explain.to(feats.device)),dim=-1)
                mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()

        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax( self.model_eval(feats, graph, edge_weights=(1 - explain.to(feats.device)))[
                        index],dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax( self.model_eval(feats, graph,
                                    edge_weights=(1 - explain.to(feats.device))),dim=-1)
                mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()

                fid_plus = original_pred[:,label] - mask_pred_plus[:,label]
                fid_minus = original_pred[:,label] - mask_pred_minus[:,label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return fid_plus, fid_minus, fid_plus_label, fid_minus_label,1 - sparsity

    def cosin_distance(self,embedding0,embedding1):

        return  torch.sqrt((embedding0-embedding1)**2+1E-8)

        # return torch.sum(embedding0*embedding1)/(torch.sqrt(torch.sum(embedding0**2)+1E-8)*
        #                                   torch.sqrt(torch.sum(embedding1**2)+1E-8))

    def edit_distance_gt_ratio_plus(self, index, graph, gt_graph, label,ed_weight,reverse=False,k=1,r_shuffle=True):
        # reverse = False ,fid+ , reverse= True fid-
        if self.type == 'node':
            # matrix_0_graph = graph[0].cpu().numpy().tolist()
            # matrix_1_graph = graph[1].cpu().numpy().tolist()
            matrix_0 = gt_graph[0][0] #.cpu().numpy()
            matrix_1 = gt_graph[0][1] #.cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (ed_weight, #gt_graph[1],
                     (matrix_0,matrix_1)),shape=(self.features.shape[0], self.features.shape[0])).tocsr()

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(index, 3,
                                                  graph,
                                                  relabel_nodes=False)
            edge_index = edge_index.cpu().detach().numpy()
            sample_matrix = coo_matrix(
                (np.ones_like(edge_index[0]),
                 (edge_index[0], edge_index[1])), shape=(self.features.shape[0], self.features.shape[0])).tocsr()

            # if not reverse:  # fid +
            #     graph_matrix = sample_matrix - sample_matrix * (gt_graph_matrix)
            # else:
            graph_matrix = sample_matrix.multiply(gt_graph_matrix)
            non_graph_matrix = sample_matrix -  graph_matrix

            weights = graph_matrix[edge_index[0], edge_index[1]].A[0]
            explain_weights = weights
            explain = torch.tensor(weights).float().to(graph.device)
            weights = non_graph_matrix[edge_index[0], edge_index[1]].A[0]
            non_explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()

        else:
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][index][0]  # .cpu().numpy()
            matrix_1 = gt_graph[0][index][1]  # .cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (ed_weight,  # weight
                 (matrix_0, matrix_1)), shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph, matrix_1_graph].A[0]
            explain_weights = weights
            # if not reverse:
            #     explain = torch.tensor(1 - weights).float().to(graph.device)
            # else:
            explain = torch.tensor(weights).float().to(graph.device)
            non_explain = torch.tensor(1-weights).float().to(graph.device)
            # explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()


        if self.undirect:
            maps = {}
            explain_list = []
            non_explain_list = []
            for i, (nodeid0, nodeid1, ex) in enumerate(zip(matrix_0, matrix_1, weights)):
                max_node = max(nodeid0, nodeid1)
                min_node = min(nodeid0, nodeid1)
                if (min_node, max_node) in maps.keys():
                    maps[(min_node, max_node)].append(i)
                    if ex > 0.5:
                        explain_list.append((min_node, max_node))
                    else:
                        non_explain_list.append((min_node, max_node))
                else:
                    maps[(min_node, max_node)] = [i]

            # if len(explain_list)> k:
            #     explain_indexs_combin = combinations(explain_list, len(explain_list)- k)
            # else:
            #     explain_indexs_combin = combinations(explain_list, len(explain_list))
            #
            # explain_indexs_combin = list(explain_indexs_combin)
            # if len(explain_list)>25:
            #     random.shuffle(explain_list)
            # explain_list = explain_list[:25]
            # if len(non_explain_list)>25:
            #     random.shuffle(non_explain_list)
            # non_explain_list = non_explain_list[:25]

            # non_explain_indexs_combin = combinations(non_explain_list,k)
            # non_explain_indexs_combin = list(non_explain_indexs_combin)
        else:
            explain_list = torch.nonzero(explain).cpu().detach().numpy().tolist()
            # non_explain_indexs = torch.nonzero(non_explain).cpu().detach().numpy().tolist()

            # if len(explain_indexs)> k:
            #     explain_indexs_combin = combinations(explain_indexs,len(explain_indexs) - k)
            # else:
            #     explain_indexs_combin = combinations(explain_indexs, len(explain_indexs) )
            # explain_indexs_combin = list(explain_indexs_combin)
            # if len(explain_indexs)>25:
            #     random.shuffle(explain_indexs)
            # explain_indexs = explain_indexs[:25]
            # if len(non_explain_indexs)>25:
            #     random.shuffle(non_explain_indexs)
            # non_explain_indexs = non_explain_indexs[:25]
            #
            # non_explain_indexs_combin = combinations(non_explain_indexs,k)
            # non_explain_indexs_combin = list(non_explain_indexs_combin)

        explaine_ratio = np.ones(len(explain_list))
        explaine_ratio = k * explaine_ratio.sum() * (explaine_ratio/explaine_ratio.sum())
        explaine_ratio_remove = np.random.binomial(1,explaine_ratio,size=(self.max_length,explaine_ratio.shape[0]))
        # explaine_length = len(explain_indexs_combin)
        # non_explaine_length = len(non_explain_indexs_combin)

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            # graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred , original_embedding= self.model_to_explain(feats, graph,embedding=True)
                original_embedding = original_embedding[index]
                original_pred = torch.softmax(original_pred[index], dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            # graph = self.graphs[index].detach()
            with torch.no_grad():
                original_pred , original_embedding= self.model_to_explain(feats, graph,embedding=True)
                original_pred = torch.softmax(original_pred, dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
                # add mask

        def cal_fid_embedding_plus():
            # global explain_indexs_combin
            list_explain = torch.zeros([self.max_length, explain.shape[0]])
            for i in range(self.max_length):
                remove_edges = explaine_ratio_remove[i]
                for idx , edge in enumerate(explain_list):
                    if remove_edges[idx]==1:
                        if self.undirect:  # undirect graph
                            id_lists = maps[edge]  # get two edges id
                            for id in id_lists:
                                list_explain[i, id] = 1.0
                        else:
                            list_explain[i, idx] = 1.0

            fid_plus_prob_list = []
            fid_plus_acc_list = []
            fid_plus_embedding_distance_list = []

            for i in range(self.max_length):
                if self.type == 'node':
                    with torch.no_grad():
                        mask_pred_plus, embedding_expl_src_plus = self.model_to_explain(feats, graph,
                                    edge_weights=(1 - list_explain[i].to(feats.device)),embedding=True)
                        mask_pred_plus, embedding_expl_src_plus = mask_pred_plus[index], \
                            embedding_expl_src_plus[index]#.view(1, -1)

                        mask_pred_plus = torch.softmax(mask_pred_plus, dim=-1)
                        mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()

                        fid_plus = original_pred[label] - mask_pred_plus[label]
                        fid_plus_label = int(original_label == label) - int(mask_label_plus == label)

                        # cosin distance
                        distance = self.cosin_distance(original_embedding,embedding_expl_src_plus)

                else:
                    with torch.no_grad():

                        mask_pred_plus , embedding_expl_src_plus= self.model_to_explain(feats, graph,
                                                edge_weights=(1 - list_explain[i].to(feats.device)),embedding=True)
                        mask_pred_plus = torch.softmax(mask_pred_plus, dim=-1)
                        mask_label_plus = mask_pred_plus.argmax(dim=-1).detach()

                        fid_plus = original_pred[:, label] - mask_pred_plus[:, label]
                        fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                        # cosin distance
                        distance = self.cosin_distance(original_embedding,embedding_expl_src_plus)

                fid_plus_prob_list.append(fid_plus)
                fid_plus_acc_list.append(fid_plus_label)
                fid_plus_embedding_distance_list.append(distance)
            if len(fid_plus_prob_list) <1:
                # print("list zeros")
                return 0, 0, 0
            else:
                fid_plus_mean = torch.stack(fid_plus_prob_list).mean().cpu().detach().numpy()
                fid_plus_label_mean = np.stack(fid_plus_acc_list).mean()
                fid_plus_embedding_distance_list = torch.stack(fid_plus_embedding_distance_list).mean().cpu().detach().numpy()
            return fid_plus_mean,fid_plus_label_mean,fid_plus_embedding_distance_list

        fid_plus_mean,fid_plus_label_mean,fid_plus_embedding_distance_list =  cal_fid_embedding_plus()
        return fid_plus_mean,fid_plus_label_mean,fid_plus_embedding_distance_list,explain_weights

    def edit_distance_gt_ratio_minus(self, index, graph, gt_graph, label,ed_weight,reverse=False,k=1,r_shuffle=True):
        # reverse = False ,fid+ , reverse= True fid-
        if self.type == 'node':
            # matrix_0_graph = graph[0].cpu().numpy().tolist()
            # matrix_1_graph = graph[1].cpu().numpy().tolist()
            matrix_0 = gt_graph[0][0] #.cpu().numpy()
            matrix_1 = gt_graph[0][1] #.cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (ed_weight , # gt_graph[1],
                     (matrix_0,matrix_1)),shape=(self.features.shape[0], self.features.shape[0])).tocsr()

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(index, 3,
                                                  graph,
                                                  relabel_nodes=False)
            edge_index = edge_index.cpu().detach().numpy()
            sample_matrix = coo_matrix(
                (np.ones_like(edge_index[0]),
                 (edge_index[0], edge_index[1])), shape=(self.features.shape[0], self.features.shape[0])).tocsr()

            # if not reverse:  # fid +
            #     graph_matrix = sample_matrix - sample_matrix * (gt_graph_matrix)
            # else:
            graph_matrix = sample_matrix.multiply(gt_graph_matrix)
            non_graph_matrix = sample_matrix -  graph_matrix

            weights = graph_matrix[edge_index[0], edge_index[1]].A[0]
            explain_weights = weights
            explain = torch.tensor(weights).float().to(graph.device)
            weights = non_graph_matrix[edge_index[0], edge_index[1]].A[0]
            non_explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()
        else:
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][index][0]  # .cpu().numpy()
            matrix_1 = gt_graph[0][index][1]  # .cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (ed_weight, # gt_graph[1][index],
                 (matrix_0, matrix_1)), shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph, matrix_1_graph].A[0]
            explain_weights = weights
            # if not reverse:
            #     explain = torch.tensor(1 - weights).float().to(graph.device)
            # else:
            explain = torch.tensor(weights).float().to(graph.device)
            non_explain = torch.tensor(1-weights).float().to(graph.device)
            # explain = torch.tensor(weights).float().to(graph.device)
            sparsity = torch.sum(explain) / torch.ones_like(explain).sum()
        if self.undirect:
            maps = {}
            explain_list = []
            non_explain_list = []
            for i, (nodeid0, nodeid1, ex) in enumerate(zip(matrix_0, matrix_1, weights)):
                max_node = max(nodeid0, nodeid1)
                min_node = min(nodeid0, nodeid1)
                if (min_node, max_node) in maps.keys():
                    maps[(min_node, max_node)].append(i)
                    if ex > 0.5:
                        explain_list.append((min_node, max_node))
                    else:
                        non_explain_list.append((min_node, max_node))
                else:
                    maps[(min_node, max_node)] = [i]


            # if len(explain_list)>25:
            #     random.shuffle(explain_list)
            # explain_list = explain_list[:25]
            # if len(non_explain_list)>25:
            #     random.shuffle(non_explain_list)
            # non_explain_list = non_explain_list[:25]
            #
            # # explain_indexs_combin = combinations(explain_list, k)
            # # explain_indexs_combin = list(explain_indexs_combin)
            # non_explain_indexs_combin = combinations(non_explain_list, k)
            # non_explain_indexs_combin = list(non_explain_indexs_combin)
        else:
            explain_indexs = torch.nonzero(explain).cpu().detach().numpy().tolist()
            non_explain_list = torch.nonzero(non_explain).cpu().detach().numpy().tolist()


            # if len(explain_indexs)>25:
            #     random.shuffle(explain_indexs)
            # explain_indexs = explain_indexs[:25]
            # if len(non_explain_indexs)>25:
            #     random.shuffle(non_explain_indexs)
            # non_explain_indexs = non_explain_indexs[:25]
            #
            # # explain_indexs_combin = combinations(explain_indexs, k)
            # # explain_indexs_combin = list(explain_indexs_combin)
            # non_explain_indexs_combin = combinations(non_explain_indexs, k)
            # non_explain_indexs_combin = list(non_explain_indexs_combin)



        non_explaine_ratio = np.ones(len(non_explain_list))
        non_explaine_ratio = k * non_explaine_ratio.sum() * (non_explaine_ratio/non_explaine_ratio.sum())
        non_explaine_ratio_remove = np.random.binomial(1,non_explaine_ratio,size=(self.max_length,non_explaine_ratio.shape[0]))
        # explaine_length = len(explain_indexs_combin)
        # non_explaine_length = len(non_explain_indexs_combin)

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            # graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred , original_embedding= self.model_to_explain(feats, graph,embedding=True)
                original_embedding = original_embedding[index]
                original_pred = torch.softmax(original_pred[index], dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            # graph = self.graphs[index].detach()
            with torch.no_grad():
                original_pred , original_embedding= self.model_to_explain(feats, graph,embedding=True)
                original_pred = torch.softmax(original_pred, dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()
                # add mask

        def cal_fid_embedding_minus():
            # global non_explain_indexs_combin
            list_explain = torch.zeros([self.max_length, non_explain.shape[0]])
            for i in range(self.max_length):
                remove_edges = non_explaine_ratio_remove[i]
                for idx, edge in enumerate(non_explain_list):
                    if remove_edges[idx] == 1:
                        if self.undirect:  # undirect graph
                            id_lists = maps[edge]  # get two edges id
                            for id in id_lists:
                                list_explain[i, id] = 1.0
                        else:
                            list_explain[i, idx] = 1.0

            # length = min(non_explaine_length, self.max_length)  # int(torch.sum(explain).item())
            # list_explain = torch.zeros([length, non_explain.shape[0]])
            # # indexes = torch.argwhere(explain > 0)
            # for i, com in enumerate(non_explain_indexs_combin):
            #     if not i<length:
            #         break
            #     if self.undirect:  # undirect graph
            #         for c in com:  # (min_node_id, max_node_id)
            #             id_lists = maps[c]  # get two edges id
            #             for id in id_lists:
            #                 list_explain[i, id] = 1
            #     else:
            #         for c in com:
            #             list_explain[i, c] = 1

            fid_minus_prob_list = []
            fid_minus_acc_list = []
            fid_minus_embedding_distance_list = []

            for i in range(self.max_length):
                if self.type == 'node':
                    with torch.no_grad():
                        mask_pred_minus, embedding_expl_src_minus = self.model_to_explain(feats, graph,
                                                            edge_weights=1 - list_explain[i].to(feats.device),
                                                                                        embedding=True)
                        mask_pred_minus, embedding_expl_src_minus = mask_pred_minus[index], \
                            embedding_expl_src_minus[index] #.view(1, -1)

                        mask_pred_minus = torch.softmax(mask_pred_minus, dim=-1)
                        mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()

                        fid_minus = original_pred[label] - mask_pred_minus[label]
                        fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

                        # cosin distance
                        distance = self.cosin_distance(original_embedding,embedding_expl_src_minus)

                else:
                    with torch.no_grad():

                        mask_pred_minus , embedding_expl_src_minus= self.model_to_explain(feats, graph,
                                            edge_weights=1 - list_explain[i].to(feats.device),embedding=True)
                        mask_pred_minus = torch.softmax(mask_pred_minus, dim=-1)
                        mask_label_minus = mask_pred_minus.argmax(dim=-1).detach()

                        fid_minus = original_pred[:, label] - mask_pred_minus[:, label]
                        fid_minus_label = int(original_label == label) - int(mask_label_minus == label)
                        # cosin distance
                        distance = self.cosin_distance(original_embedding,embedding_expl_src_minus)

                fid_minus_prob_list.append(fid_minus)
                fid_minus_acc_list.append(fid_minus_label)
                fid_minus_embedding_distance_list.append(distance)

            if len(fid_minus_prob_list) <1:
                # print("list zero")
                return 1, 1, 1
            else:
                fid_minus_mean = torch.stack(fid_minus_prob_list).mean().cpu().detach().numpy()
                fid_minus_label_mean = np.stack(fid_minus_acc_list).mean()
                fid_minus_embedding_distance_list = torch.stack(fid_minus_embedding_distance_list).mean().cpu().detach().numpy()
            return fid_minus_mean,fid_minus_label_mean,fid_minus_embedding_distance_list

        fid_minus_mean, fid_minus_label_mean, fid_minus_embedding_distance_list = cal_fid_embedding_minus()
        return fid_minus_mean, fid_minus_label_mean, fid_minus_embedding_distance_list,explain_weights


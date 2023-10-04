from math import sqrt

import numpy as np
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge

from scipy.sparse import coo_matrix
"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""


class GNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.

    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """
    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, reg_coefs=(0.05, 1.0)):
        super().__init__(model_to_explain, graphs, features, task)
        self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs


    def _set_masks(self, x, edge_index):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        (N, F), E = x.size(), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        # for module in self.model_to_explain.modules():
        #     if isinstance(module, MessagePassing):
        #         module.__explain__ = True
        #         module.__edge_mask__ = self.edge_mask


    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        # for module in self.model_to_explain.modules():
        #     if isinstance(module, MessagePassing):
        #         module.__explain__ = False
        #         module.__edge_mask__ = None
        self.edge_mask = None

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        # print(" loss , ",cce_loss.item(),size_loss.item()/size_reg,mask_ent_loss.item()/entropy_reg)
        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.cuda()
        self.model_to_explain.eval()
        self._clear_masks()           # remove mask

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)[index]
                pred_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            graph = self.graphs[index].detach()
            # Remove self-loops
            graph = graph[:, (graph[0] != graph[1])]
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)
                pred_label = original_pred.argmax(dim=-1).detach()

        self._set_masks(feats, graph)
        optimizer = Adam([self.edge_mask], lr=self.lr)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            # Sample possible explanation
            if self.type == 'node':
                masked_pred = self.model_to_explain(feats, graph, edge_weights=torch.sigmoid(self.edge_mask).to(feats.device))[index]
                loss = self._loss(masked_pred.unsqueeze(0), pred_label.unsqueeze(0), self.edge_mask, self.reg_coefs)
            else:
                masked_pred = self.model_to_explain(feats, graph, edge_weights=torch.sigmoid(self.edge_mask).to(feats.device))
                loss = self._loss(masked_pred, pred_label, self.edge_mask, self.reg_coefs)

            # if e % 10 == 0:
            #     print(loss.item(), end=" ")

            loss.backward()
            optimizer.step()

        # print(loss.item())

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)): # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights

    def cal_fid(self, index, graph, explain, label):

        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.cuda()
        self.model_to_explain.eval()
        self._clear_masks()           # remove mask

        # origial output
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
            # Remove self-loops
            graph = graph[:, (graph[0] != graph[1])]
            with torch.no_grad():
                original_pred = torch.softmax(self.model_to_explain(feats, graph),dim=-1)
                original_label = original_pred.argmax(dim=-1).detach()

        # add mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                         edge_weights=torch.sigmoid(explain).to(feats.device))[index],dim=-1)
                mask_label_minus = original_pred.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                          edge_weights=torch.sigmoid(explain).to(feats.device)),dim=-1)
                mask_label_minus = original_pred.argmax(dim=-1).detach()

        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax( self.model_to_explain(feats, graph, edge_weights=(1 - torch.sigmoid(explain).to(feats.device)))[
                        index],dim=-1)
                mask_label_plus = original_pred.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax( self.model_to_explain(feats, graph,
                                    edge_weights=(1 - torch.sigmoid(explain).to(feats.device))),dim=-1)
                mask_label_plus = original_pred.argmax(dim=-1).detach()
                fid_plus = original_pred[:, label] - mask_pred_plus[:, label]
                fid_minus = original_pred[:, label] - mask_pred_minus[:, label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return  fid_plus, fid_minus, fid_plus_label, fid_minus_label

    def cal_fid_gt(self, index, graph, gt_graph, label):

        if self.type == 'node':
            matrix_0_graph = graph[0].cpu().numpy().tolist()
            matrix_1_graph = graph[1].cpu().numpy().tolist()

            matrix_0 = gt_graph[0][0]  # .cpu().numpy()
            matrix_1 = gt_graph[0][1]  # .cpu().numpy()
            gt_graph_matrix = coo_matrix(
                (gt_graph[1],
                 (matrix_0, matrix_1)), shape=(self.features.shape[0], self.features.shape[0])).tocsr()
            weights = gt_graph_matrix[matrix_0_graph, matrix_1_graph].A[0]
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
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                                                                      edge_weights=explain.to(
                                                                          feats.device))[index], dim=-1)
                mask_label_minus = original_pred.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                                                                      edge_weights=explain.to(
                                                                          feats.device)), dim=-1)
                mask_label_minus = original_pred.argmax(dim=-1).detach()
        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax(self.model_to_explain(feats, graph, edge_weights=(
                                1 - explain.to(feats.device)))[
                                      index], dim=-1)
                mask_label_plus = original_pred.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax(self.model_to_explain(feats, graph,
                                                                     edge_weights=(
                                                                                 1 - explain.to(
                                                                             feats.device))), dim=-1)
                mask_label_plus = original_pred.argmax(dim=-1).detach()

                fid_plus = original_pred[:, label] - mask_pred_plus[:, label]
                fid_minus = original_pred[:, label] - mask_pred_minus[:, label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return fid_plus, fid_minus, fid_plus_label, fid_minus_label, 1- sparsity

    def cal_fid_sparsity(self, index, graph, explain, label,sparsity = 0.5):

        retain = 1- sparsity
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
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                         edge_weights=explain.to(feats.device))[index],dim=-1)
                mask_label_minus = original_pred.argmax(dim=-1).detach()
        else:
            with torch.no_grad():
                mask_pred_minus = torch.softmax(self.model_to_explain(feats, graph,
                          edge_weights=explain.to(feats.device)),dim=-1)
                mask_label_minus = original_pred.argmax(dim=-1).detach()

        # remove mask
        if self.type == 'node':
            with torch.no_grad():
                mask_pred_plus = \
                    torch.softmax( self.model_to_explain(feats, graph, edge_weights=(1 - explain.to(feats.device)))[
                        index],dim=-1)
                mask_label_plus = original_pred.argmax(dim=-1).detach()
                fid_plus = original_pred[label] - mask_pred_plus[label]
                fid_minus = original_pred[label] - mask_pred_minus[label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        else:
            with torch.no_grad():
                mask_pred_plus = torch.softmax( self.model_to_explain(feats, graph,
                                    edge_weights=(1 - explain.to(feats.device))),dim=-1)
                mask_label_plus = original_pred.argmax(dim=-1).detach()

                fid_plus = original_pred[:,label] - mask_pred_plus[:,label]
                fid_minus = original_pred[:,label] - mask_pred_minus[:,label]

                fid_plus_label = int(original_label == label) - int(mask_label_plus == label)
                fid_minus_label = int(original_label == label) - int(mask_label_minus == label)

        return fid_plus, fid_minus, fid_plus_label, fid_minus_label,1 - sparsity

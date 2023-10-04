import copy
import time
import json
import os

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from torch_geometric.utils import k_hop_subgraph

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

def visulaized(graph,weights,store=True,name='graph'):
    plt.cla()
    G = nx.Graph()
    for i in range(graph.shape[1]):
        if weights[i]>0.5:
            G.add_edge(graph[0, i], graph[1, i])
    nx.draw_spring(G)

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    if store:
        plt.savefig("./data/graph_%s.png"%(name))
        plt.savefig("./data/graph_%s.pdf"%(name))

def edit_distance_k_gen(dir='./data',undirect=True):
    visualization = False
    dataset_name = 'mutag'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)

    all_edit_distance_lists=[]

    def explain_mapping(matrix_0,matrix_1,weights,r_map=False):
        maps = {}
        explain_list = []
        explain_nodes = []
        non_explain_list = []
        for i, (nodeid0, nodeid1, ex) in enumerate(zip(matrix_0, matrix_1, weights)):
            max_node = max(nodeid0, nodeid1)
            min_node = min(nodeid0, nodeid1)
            if (min_node, max_node) in maps.keys():
                maps[(min_node, max_node)].append(i)
                if ex > 0.5:
                    explain_list.append((min_node, max_node))
                    explain_nodes.append(min_node)
                    explain_nodes.append(max_node)
                else:
                    non_explain_list.append((min_node, max_node))
            else:
                maps[(min_node, max_node)] = [i]
        explain_nodes = list(set(explain_nodes))

        if r_map:
            return maps,explain_list,explain_nodes,non_explain_list
        else:
            return explain_list, explain_nodes, non_explain_list

    def get_adding_edge_list(graph,explain_nodes):
        adding_k_1_edge_list = []  # the new edges
        for node in explain_nodes:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(int(node), 1, torch.from_numpy(graph))
            edge_index_np = edge_index.cpu().detach().numpy()
            # for node in subset.cpu().detach().numpy():
            #     adding_k_1_nodes.add(node)
            for edge_idx in range(edge_index_np.shape[1]):
                node_id0 = edge_index_np[0, edge_idx]
                node_id1 = edge_index_np[1, edge_idx]
                max_node = max(node_id0, node_id1)
                min_node = min(node_id0, node_id1)
                if (min_node, max_node) in explain_list:
                    pass
                else:
                    if (min_node, max_node) in adding_k_1_edge_list:
                        pass
                    else:
                        adding_k_1_edge_list.append((min_node, max_node))

        adding_k_1_edge_list = list(set(adding_k_1_edge_list))
        return adding_k_1_edge_list


    # only edit
    for i in tqdm(range(len(graphs))): # len(graphs) indices

        edit_distance_lists = [[], [], [], [], [], [], []]

        if dataset_name=="mutag" :
            graph = explanation_labels[0][i]
        else:
            graph = graphs[i]
        weights = explanation_labels[1][i]

        matrix_0 = graph[0]
        matrix_1 = graph[1]
        #weights = label[1]
        edit_distance_lists[0].append(tuple(weights.tolist()))
        if undirect:
            maps, explain_list, explain_nodes, non_explain_list = explain_mapping(matrix_0, matrix_1, weights,True)
            # visualize the gt
            # visulaized_three(graph,weights,name='%d'%i)
            # exit(0)
            # visulaized(graph,np.ones_like(weights),name='%d_graph'%i)
            if i<10 and visualization:
                visulaized(graph,weights,name='%d_exp'%i)
            # visulaized(graph,1-weights,name='%d_nonexp'%i)
            # generate sample
            # by deleting edge
            k=1
            explain_indexs_combin = combinations(explain_list, k)
            explain_indexs_combin = list(explain_indexs_combin)
            # length = len(explain_indexs_combin)
            # new weight
            # temp_remove_one_edge = []
            for count, com in enumerate(explain_indexs_combin):
                weight_n = weights.copy()
                for c in com:  # (min_node_id, max_node_id)
                    id_lists = maps[c]  # get two edges id
                    for id in id_lists:
                        weight_n[id] = 0
                edit_distance_lists[k].append(tuple(weight_n.tolist()))
                if i < 10 and visualization:
                    visulaized(graph, weight_n, name='%d_%d_k_%d_exp_remove' % (i,k,count))
                # temp_remove_one_edge.append(weight_n)

            # by adding edge
            adding_k_1_edge_list = get_adding_edge_list(graph,explain_nodes)
            adding_k_1_edge_list = list(set(adding_k_1_edge_list))
            explain_indexs_combin = combinations(adding_k_1_edge_list, k)
            explain_indexs_combin = list(explain_indexs_combin)
            # temp_adding_one_edge = []
            for count, com in enumerate(explain_indexs_combin):
                weight_n = weights.copy()
                for c in com:  # (min_node_id, max_node_id)
                    id_lists = maps[c]  # get two edges id
                    for id in id_lists:
                        weight_n[id] = 1
                edit_distance_lists[k].append(tuple(weight_n.tolist()))
                if i < 10 and visualization:
                    visulaized(graph, weight_n, name='%d_%d_k_%d_exp_add' % (i, k, count))
                # temp_adding_one_edge.append(weight_n)
            edit_distance_lists[k] = set(edit_distance_lists[k])

            ############################################################################################################
            #  k = 2
            #  deleting one edge adding one edge
            for k in range(2,6):
                if k ==5 :
                    pass
                # k = 2
                # 1. remove one edge from remove one edge weights
                for weight_t in edit_distance_lists[k-1]:
                    # relist the explanation nodes,
                    # maps remain the same
                    explain_list_t,explain_nodes_t,non_explain_list_t = explain_mapping(matrix_0, matrix_1, weight_t)

                    # option one , remove one edge again, only from motifs
                    # first.  get intersection set
                    intersetion_set = set(explain_list_t) & set(explain_list)
                    # second. remove one edge from
                    explain_indexs_combin = combinations(list(intersetion_set), 1)
                    explain_indexs_combin = list(explain_indexs_combin)
                    for count, com in enumerate(explain_indexs_combin):
                        weight_n = list(copy.copy(weight_t))
                        for c in com:  # (min_node_id, max_node_id)
                            id_lists = maps[c]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 0
                        edit_distance_lists[k].append(tuple(weight_n))
                        if i < 10 and visualization:
                            visulaized(graph, weight_n, name='%d_%d_k_%d_exp_remove' % (i,k, count))

                    # option two , remove one edge again, only from motifs
                    adding_k_1_edge_list_t = get_adding_edge_list(graph,explain_nodes_t)
                    adding_k_1_edge_list_t = list(set(adding_k_1_edge_list_t) - set(explain_list))
                    explain_indexs_combin = combinations(adding_k_1_edge_list_t, 1)
                    explain_indexs_combin = list(explain_indexs_combin)
                    for count, com in enumerate(explain_indexs_combin):
                        weight_n = list(copy.copy(weight_t))
                        for c in com:  # (min_node_id, max_node_id)
                            id_lists = maps[c]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 1
                        edit_distance_lists[k].append(tuple(weight_n))
                        if i < 10 and visualization:
                            visulaized(graph, weight_n, name='%d_%d_k_%d_exp_add' % (i,k, count))
                edit_distance_lists[k] = set(edit_distance_lists[k])
        all_edit_distance_lists.append(edit_distance_lists)
    np.save('./data/sample_weights',all_edit_distance_lists)

if __name__=="__main__":
    edit_distance_k_gen()
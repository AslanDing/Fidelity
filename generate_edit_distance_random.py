import copy
import random
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
from scipy.sparse import coo_matrix

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

def visulaized(graph,weights,store=True,name='graph', posx = None):
    plt.cla()
    dpi = 300
    figure_size = 2000
    node_size = 300
    # font_size = 10
    weight_font_size = 5

    fig = plt.figure( figsize=(10,5),dpi=300)  # , dpi=60

    G = nx.Graph()

    all_nodes = []
    explain_nodes = []
    non_explain_nodes = []

    for i in range(graph.shape[1]):
        if weights[i]>0.5:
            G.add_edge(graph[0, i], graph[1, i],weight_label = weights[i],color='r')
            explain_nodes.append(graph[0, i])
            explain_nodes.append(graph[1, i])
        else:
            G.add_edge(graph[0, i], graph[1, i], weight_label=weights[i], color='k')
            non_explain_nodes.append(graph[0, i])
            non_explain_nodes.append(graph[1, i])
        all_nodes.append(graph[0, i])
        all_nodes.append(graph[1, i])

    all_nodes = list(set(all_nodes))
    explain_nodes = list(set(explain_nodes))
    non_explain_nodes = list(set(non_explain_nodes))

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight_label').items())
    explain_edge_index = np.argwhere(np.array(list(weights))>0)[:,0].tolist()
    explain_edge_list = [edges[idx] for idx in explain_edge_index]
    non_explain_edge_list = list(set(edges)-set(explain_edge_list))

    pos = posx
    if posx == None :
        pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size,node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k')

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    if store:
        plt.savefig("./data/graph_%s.png"%(name))
        plt.savefig("./data/graph_%s.pdf"%(name))
    plt.close()
    return pos

def remap_ed_k_ratio():
    undirect = True
    visualization = False
    dataset_name = 'mutag'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)

    all_edit_distance_lists = []
    # remove, adding
    # remove, adding
    all_edit_distance_remap = {('0.0', '0.0'): [],
                               ('0.1', '0.0'): [],
                               ('0.2', '0.0'): [],
                               ('0.3', '0.0'): [],
                               ('0.4', '0.0'): [],
                               ('0.5', '0.0'): [],
                               ('0.6', '0.0'): [],
                               ('0.7', '0.0'): [],
                               ('0.8', '0.0'): [],
                               ('0.9', '0.0'): [],
                               ('1.0', '0.0'): [],

                               ('0.0', '0.1'): [],
                               ('0.1', '0.1'): [],
                               ('0.2', '0.1'): [],
                               ('0.3', '0.1'): [],
                               ('0.4', '0.1'): [],
                               ('0.5', '0.1'): [],
                               ('0.6', '0.1'): [],
                               ('0.7', '0.1'): [],
                               ('0.8', '0.1'): [],
                               ('0.9', '0.1'): [],
                               ('1.0', '0.1'): [],

                               ('0.0', '0.2'): [],
                               ('0.1', '0.2'): [],
                               ('0.2', '0.2'): [],
                               ('0.3', '0.2'): [],
                               ('0.4', '0.2'): [],
                               ('0.5', '0.2'): [],
                               ('0.6', '0.2'): [],
                               ('0.7', '0.2'): [],
                               ('0.8', '0.2'): [],
                               ('0.9', '0.2'): [],
                               ('1.0', '0.2'): [],

                               ('0.0', '0.3'): [],
                               ('0.1', '0.3'): [],
                               ('0.2', '0.3'): [],
                               ('0.3', '0.3'): [],
                               ('0.4', '0.3'): [],
                               ('0.5', '0.3'): [],
                               ('0.6', '0.3'): [],
                               ('0.7', '0.3'): [],
                               ('0.8', '0.3'): [],
                               ('0.9', '0.3'): [],
                               ('1.0', '0.3'): [],

                               ('0.0', '0.4'): [],
                               ('0.1', '0.4'): [],
                               ('0.2', '0.4'): [],
                               ('0.3', '0.4'): [],
                               ('0.4', '0.4'): [],
                               ('0.5', '0.4'): [],
                               ('0.6', '0.4'): [],
                               ('0.7', '0.4'): [],
                               ('0.8', '0.4'): [],
                               ('0.9', '0.4'): [],
                               ('1.0', '0.4'): [],

                               ('0.0', '0.5'): [],
                               ('0.1', '0.5'): [],
                               ('0.2', '0.5'): [],
                               ('0.3', '0.5'): [],
                               ('0.4', '0.5'): [],
                               ('0.5', '0.5'): [],
                               ('0.6', '0.5'): [],
                               ('0.7', '0.5'): [],
                               ('0.8', '0.5'): [],
                               ('0.9', '0.5'): [],
                               ('1.0', '0.5'): [],

                               ('0.0', '0.6'): [],
                               ('0.1', '0.6'): [],
                               ('0.2', '0.6'): [],
                               ('0.3', '0.6'): [],
                               ('0.4', '0.6'): [],
                               ('0.5', '0.6'): [],
                               ('0.6', '0.6'): [],
                               ('0.7', '0.6'): [],
                               ('0.8', '0.6'): [],
                               ('0.9', '0.6'): [],
                               ('1.0', '0.6'): [],

                               ('0.0', '0.7'): [],
                               ('0.1', '0.7'): [],
                               ('0.2', '0.7'): [],
                               ('0.3', '0.7'): [],
                               ('0.4', '0.7'): [],
                               ('0.5', '0.7'): [],
                               ('0.6', '0.7'): [],
                               ('0.7', '0.7'): [],
                               ('0.8', '0.7'): [],
                               ('0.9', '0.7'): [],
                               ('1.0', '0.7'): [],

                               ('0.0', '0.8'): [],
                               ('0.1', '0.8'): [],
                               ('0.2', '0.8'): [],
                               ('0.3', '0.8'): [],
                               ('0.4', '0.8'): [],
                               ('0.5', '0.8'): [],
                               ('0.6', '0.8'): [],
                               ('0.7', '0.8'): [],
                               ('0.8', '0.8'): [],
                               ('0.9', '0.8'): [],
                               ('1.0', '0.8'): [],

                               ('0.0', '0.9'): [],
                               ('0.1', '0.9'): [],
                               ('0.2', '0.9'): [],
                               ('0.3', '0.9'): [],
                               ('0.4', '0.9'): [],
                               ('0.5', '0.9'): [],
                               ('0.6', '0.9'): [],
                               ('0.7', '0.9'): [],
                               ('0.8', '0.9'): [],
                               ('0.9', '0.9'): [],
                               ('1.0', '0.9'): [],

                               ('0.0', '1.0'): [],
                               ('0.1', '1.0'): [],
                               ('0.2', '1.0'): [],
                               ('0.3', '1.0'): [],
                               ('0.4', '1.0'): [],
                               ('0.5', '1.0'): [],
                               ('0.6', '1.0'): [],
                               ('0.7', '1.0'): [],
                               ('0.8', '1.0'): [],
                               ('0.9', '1.0'): [],
                               ('1.0', '1.0'): [],
                                }  # edit distance = 5 }

    def explain_mapping(matrix_0, matrix_1, weights, r_map=False):
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
            return maps, explain_list, explain_nodes, non_explain_list
        else:
            return explain_list, explain_nodes, non_explain_list

    def get_adding_edge_list(graph, explain_nodes, explain_list_t):
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
                if (min_node, max_node) in explain_list_t:
                    pass
                else:
                    if (min_node, max_node) in adding_k_1_edge_list:
                        pass
                    else:
                        adding_k_1_edge_list.append((min_node, max_node))

        adding_k_1_edge_list = list(set(adding_k_1_edge_list))
        return adding_k_1_edge_list

    # only edit
    for i in tqdm(range(len(graphs))):  # len(graphs) indices
        edit_distance_rt = {('0.0', '0.0'): [],
                               ('0.1', '0.0'): [],
                               ('0.2', '0.0'): [],
                               ('0.3', '0.0'): [],
                               ('0.4', '0.0'): [],
                               ('0.5', '0.0'): [],
                               ('0.6', '0.0'): [],
                               ('0.7', '0.0'): [],
                               ('0.8', '0.0'): [],
                               ('0.9', '0.0'): [],
                               ('1.0', '0.0'): [],

                               ('0.0', '0.1'): [],
                               ('0.1', '0.1'): [],
                               ('0.2', '0.1'): [],
                               ('0.3', '0.1'): [],
                               ('0.4', '0.1'): [],
                               ('0.5', '0.1'): [],
                               ('0.6', '0.1'): [],
                               ('0.7', '0.1'): [],
                               ('0.8', '0.1'): [],
                               ('0.9', '0.1'): [],
                               ('1.0', '0.1'): [],

                               ('0.0', '0.2'): [],
                               ('0.1', '0.2'): [],
                               ('0.2', '0.2'): [],
                               ('0.3', '0.2'): [],
                               ('0.4', '0.2'): [],
                               ('0.5', '0.2'): [],
                               ('0.6', '0.2'): [],
                               ('0.7', '0.2'): [],
                               ('0.8', '0.2'): [],
                               ('0.9', '0.2'): [],
                               ('1.0', '0.2'): [],

                               ('0.0', '0.3'): [],
                               ('0.1', '0.3'): [],
                               ('0.2', '0.3'): [],
                               ('0.3', '0.3'): [],
                               ('0.4', '0.3'): [],
                               ('0.5', '0.3'): [],
                               ('0.6', '0.3'): [],
                               ('0.7', '0.3'): [],
                               ('0.8', '0.3'): [],
                               ('0.9', '0.3'): [],
                               ('1.0', '0.3'): [],

                               ('0.0', '0.4'): [],
                               ('0.1', '0.4'): [],
                               ('0.2', '0.4'): [],
                               ('0.3', '0.4'): [],
                               ('0.4', '0.4'): [],
                               ('0.5', '0.4'): [],
                               ('0.6', '0.4'): [],
                               ('0.7', '0.4'): [],
                               ('0.8', '0.4'): [],
                               ('0.9', '0.4'): [],
                               ('1.0', '0.4'): [],

                               ('0.0', '0.5'): [],
                               ('0.1', '0.5'): [],
                               ('0.2', '0.5'): [],
                               ('0.3', '0.5'): [],
                               ('0.4', '0.5'): [],
                               ('0.5', '0.5'): [],
                               ('0.6', '0.5'): [],
                               ('0.7', '0.5'): [],
                               ('0.8', '0.5'): [],
                               ('0.9', '0.5'): [],
                               ('1.0', '0.5'): [],

                               ('0.0', '0.6'): [],
                               ('0.1', '0.6'): [],
                               ('0.2', '0.6'): [],
                               ('0.3', '0.6'): [],
                               ('0.4', '0.6'): [],
                               ('0.5', '0.6'): [],
                               ('0.6', '0.6'): [],
                               ('0.7', '0.6'): [],
                               ('0.8', '0.6'): [],
                               ('0.9', '0.6'): [],
                               ('1.0', '0.6'): [],

                               ('0.0', '0.7'): [],
                               ('0.1', '0.7'): [],
                               ('0.2', '0.7'): [],
                               ('0.3', '0.7'): [],
                               ('0.4', '0.7'): [],
                               ('0.5', '0.7'): [],
                               ('0.6', '0.7'): [],
                               ('0.7', '0.7'): [],
                               ('0.8', '0.7'): [],
                               ('0.9', '0.7'): [],
                               ('1.0', '0.7'): [],

                               ('0.0', '0.8'): [],
                               ('0.1', '0.8'): [],
                               ('0.2', '0.8'): [],
                               ('0.3', '0.8'): [],
                               ('0.4', '0.8'): [],
                               ('0.5', '0.8'): [],
                               ('0.6', '0.8'): [],
                               ('0.7', '0.8'): [],
                               ('0.8', '0.8'): [],
                               ('0.9', '0.8'): [],
                               ('1.0', '0.8'): [],

                               ('0.0', '0.9'): [],
                               ('0.1', '0.9'): [],
                               ('0.2', '0.9'): [],
                               ('0.3', '0.9'): [],
                               ('0.4', '0.9'): [],
                               ('0.5', '0.9'): [],
                               ('0.6', '0.9'): [],
                               ('0.7', '0.9'): [],
                               ('0.8', '0.9'): [],
                               ('0.9', '0.9'): [],
                               ('1.0', '0.9'): [],

                               ('0.0', '1.0'): [],
                               ('0.1', '1.0'): [],
                               ('0.2', '1.0'): [],
                               ('0.3', '1.0'): [],
                               ('0.4', '1.0'): [],
                               ('0.5', '1.0'): [],
                               ('0.6', '1.0'): [],
                               ('0.7', '1.0'): [],
                               ('0.8', '1.0'): [],
                               ('0.9', '1.0'): [],
                               ('1.0', '1.0'): [],
                                }  # edit distance = 5 }
        if 'syn' in dataset_name:
            graph = graphs
        else:
            if dataset_name == "mutag":
                graph = explanation_labels[0][i]
            else:
                graph = graphs[i]
        weights = explanation_labels[1][i]

        matrix_0 = graph[0]
        matrix_1 = graph[1]
        maps, explain_list, explain_nodes, non_explain_list = explain_mapping(matrix_0, matrix_1, weights, True)
        explain_np = np.arange(0,len(explain_list),1)
        non_explain_np = np.arange(0, len(non_explain_list), 1)
        if i in indices:
            for key in edit_distance_rt.keys():
                # remove, adding
                remove_c = float(key[0])
                adding_c = float(key[1])
                if remove_c == 0 and adding_c == 0:
                    edit_distance_rt[(key[0], key[1])].append(weights)
                    continue

                # removing edges
                sample_ratio_explain = remove_c * len(explain_list) * np.ones_like(explain_np)/(len(explain_np)+1E-8)
                # adding edges
                sample_ratio_nonexplain = adding_c * len(explain_list) * np.ones_like(non_explain_np)/(len(non_explain_list)+1E-8)
                sample_ratio_nonexplain = np.where(sample_ratio_nonexplain>1,np.ones_like(sample_ratio_nonexplain),sample_ratio_nonexplain)

                # sample
                samples_explain_removing = np.random.binomial(1,sample_ratio_explain,size=(10,sample_ratio_explain.shape[0]))
                samples_nonexplain_removing = np.random.binomial(1,sample_ratio_nonexplain,size=(10,sample_ratio_nonexplain.shape[0]))


                for ii in range(samples_explain_removing.shape[0]):
                    weight_n = weights.copy()
                    # remove the edges
                    for idx, edge in enumerate(explain_list):
                        if samples_explain_removing[ii,idx]==1:
                            id_lists = maps[edge]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 0
                    # adding the edges
                    for idx, edge in enumerate(non_explain_list):
                        if samples_nonexplain_removing[ii, idx] == 1:
                            id_lists = maps[edge]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 1

                    edit_distance_rt[(key[0], key[1])].append(weight_n)


        for key in edit_distance_rt.keys():
            # if len(edit_distance_rt[key]) > 20:
            #     random.shuffle(edit_distance_rt[key])
            all_edit_distance_remap[key].append(edit_distance_rt[key][:])

    # np.save('./data/%s_sample_weights'%dataset_name, all_edit_distance_lists)

    np.save('./data/%s_random_sample_maps_ratio' % dataset_name, all_edit_distance_remap)

def remap_ed_k_nodes_undi_ratio():
    undirect = True
    visualization = False
    dataset_name = 'syn3'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)

    graph_tensor = torch.tensor(graphs).cuda()
    all_edit_distance_lists = []
    # remove, adding
    all_edit_distance_remap = {('0.0', '0.0'): [],
                               ('0.1', '0.0'): [],
                               ('0.2', '0.0'): [],
                               ('0.3', '0.0'): [],
                               ('0.4', '0.0'): [],
                               ('0.5', '0.0'): [],
                               ('0.6', '0.0'): [],
                               ('0.7', '0.0'): [],
                               ('0.8', '0.0'): [],
                               ('0.9', '0.0'): [],
                               ('1.0', '0.0'): [],

                               ('0.0', '0.1'): [],
                               ('0.1', '0.1'): [],
                               ('0.2', '0.1'): [],
                               ('0.3', '0.1'): [],
                               ('0.4', '0.1'): [],
                               ('0.5', '0.1'): [],
                               ('0.6', '0.1'): [],
                               ('0.7', '0.1'): [],
                               ('0.8', '0.1'): [],
                               ('0.9', '0.1'): [],
                               ('1.0', '0.1'): [],

                               ('0.0', '0.2'): [],
                               ('0.1', '0.2'): [],
                               ('0.2', '0.2'): [],
                               ('0.3', '0.2'): [],
                               ('0.4', '0.2'): [],
                               ('0.5', '0.2'): [],
                               ('0.6', '0.2'): [],
                               ('0.7', '0.2'): [],
                               ('0.8', '0.2'): [],
                               ('0.9', '0.2'): [],
                               ('1.0', '0.2'): [],

                               ('0.0', '0.3'): [],
                               ('0.1', '0.3'): [],
                               ('0.2', '0.3'): [],
                               ('0.3', '0.3'): [],
                               ('0.4', '0.3'): [],
                               ('0.5', '0.3'): [],
                               ('0.6', '0.3'): [],
                               ('0.7', '0.3'): [],
                               ('0.8', '0.3'): [],
                               ('0.9', '0.3'): [],
                               ('1.0', '0.3'): [],

                               ('0.0', '0.4'): [],
                               ('0.1', '0.4'): [],
                               ('0.2', '0.4'): [],
                               ('0.3', '0.4'): [],
                               ('0.4', '0.4'): [],
                               ('0.5', '0.4'): [],
                               ('0.6', '0.4'): [],
                               ('0.7', '0.4'): [],
                               ('0.8', '0.4'): [],
                               ('0.9', '0.4'): [],
                               ('1.0', '0.4'): [],

                               ('0.0', '0.5'): [],
                               ('0.1', '0.5'): [],
                               ('0.2', '0.5'): [],
                               ('0.3', '0.5'): [],
                               ('0.4', '0.5'): [],
                               ('0.5', '0.5'): [],
                               ('0.6', '0.5'): [],
                               ('0.7', '0.5'): [],
                               ('0.8', '0.5'): [],
                               ('0.9', '0.5'): [],
                               ('1.0', '0.5'): [],

                               ('0.0', '0.6'): [],
                               ('0.1', '0.6'): [],
                               ('0.2', '0.6'): [],
                               ('0.3', '0.6'): [],
                               ('0.4', '0.6'): [],
                               ('0.5', '0.6'): [],
                               ('0.6', '0.6'): [],
                               ('0.7', '0.6'): [],
                               ('0.8', '0.6'): [],
                               ('0.9', '0.6'): [],
                               ('1.0', '0.6'): [],

                               ('0.0', '0.7'): [],
                               ('0.1', '0.7'): [],
                               ('0.2', '0.7'): [],
                               ('0.3', '0.7'): [],
                               ('0.4', '0.7'): [],
                               ('0.5', '0.7'): [],
                               ('0.6', '0.7'): [],
                               ('0.7', '0.7'): [],
                               ('0.8', '0.7'): [],
                               ('0.9', '0.7'): [],
                               ('1.0', '0.7'): [],

                               ('0.0', '0.8'): [],
                               ('0.1', '0.8'): [],
                               ('0.2', '0.8'): [],
                               ('0.3', '0.8'): [],
                               ('0.4', '0.8'): [],
                               ('0.5', '0.8'): [],
                               ('0.6', '0.8'): [],
                               ('0.7', '0.8'): [],
                               ('0.8', '0.8'): [],
                               ('0.9', '0.8'): [],
                               ('1.0', '0.8'): [],

                               ('0.0', '0.9'): [],
                               ('0.1', '0.9'): [],
                               ('0.2', '0.9'): [],
                               ('0.3', '0.9'): [],
                               ('0.4', '0.9'): [],
                               ('0.5', '0.9'): [],
                               ('0.6', '0.9'): [],
                               ('0.7', '0.9'): [],
                               ('0.8', '0.9'): [],
                               ('0.9', '0.9'): [],
                               ('1.0', '0.9'): [],

                               ('0.0', '1.0'): [],
                               ('0.1', '1.0'): [],
                               ('0.2', '1.0'): [],
                               ('0.3', '1.0'): [],
                               ('0.4', '1.0'): [],
                               ('0.5', '1.0'): [],
                               ('0.6', '1.0'): [],
                               ('0.7', '1.0'): [],
                               ('0.8', '1.0'): [],
                               ('0.9', '1.0'): [],
                               ('1.0', '1.0'): [],
                                }  # edit distance = 5 }

    def explain_mapping(matrix_0, matrix_1, weights, r_map=False):
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
            return maps, explain_list, explain_nodes, non_explain_list
        else:
            return explain_list, explain_nodes, non_explain_list

    def get_adding_edge_list(graph, explain_nodes, explain_list_t):
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
                if (min_node, max_node) in explain_list_t:
                    pass
                else:
                    if (min_node, max_node) in adding_k_1_edge_list:
                        pass
                    else:
                        adding_k_1_edge_list.append((min_node, max_node))

        adding_k_1_edge_list = list(set(adding_k_1_edge_list))
        return adding_k_1_edge_list

    def remake_weight(nodes_num, graph_index0,graph_index1,subgraph_index0,subgraph_index1,subgraph_weight):
        sub_graph_matrix = coo_matrix(
            (subgraph_weight,
             (subgraph_index0, subgraph_index1)), shape=(nodes_num, nodes_num)).tocsr()
        weights_graph = sub_graph_matrix[graph_index0, graph_index1].A[0]
        return weights_graph


    # change the direct graph to undir
    graph = graphs
    weights = explanation_labels[1]
    for ii in range(graph.shape[1]):
        pair = graph[:,ii]
        idx_edge = np.where((graphs.T == pair.T).all(axis=1))
        idx_edge_rev = np.where((graphs.T == [pair[1], pair[0]]).all(axis=1))
        gt = weights[idx_edge] + weights[idx_edge_rev]
        if gt == 0:
            pass
        else:
            weights[idx_edge]=1.0
            weights[idx_edge_rev]=1.0
    np.save("./data/%s_gt_subgraphs"%dataset_name,weights)
    matrix_0 = graph[0]
    matrix_1 = graph[1]
    nodes_num = features.shape[0]
    # only edit
    for i in tqdm(range(nodes_num)):  # len(graphs) indices
        edit_distance_rt = {('0.0', '0.0'): [],
                               ('0.1', '0.0'): [],
                               ('0.2', '0.0'): [],
                               ('0.3', '0.0'): [],
                               ('0.4', '0.0'): [],
                               ('0.5', '0.0'): [],
                               ('0.6', '0.0'): [],
                               ('0.7', '0.0'): [],
                               ('0.8', '0.0'): [],
                               ('0.9', '0.0'): [],
                               ('1.0', '0.0'): [],

                               ('0.0', '0.1'): [],
                               ('0.1', '0.1'): [],
                               ('0.2', '0.1'): [],
                               ('0.3', '0.1'): [],
                               ('0.4', '0.1'): [],
                               ('0.5', '0.1'): [],
                               ('0.6', '0.1'): [],
                               ('0.7', '0.1'): [],
                               ('0.8', '0.1'): [],
                               ('0.9', '0.1'): [],
                               ('1.0', '0.1'): [],

                               ('0.0', '0.2'): [],
                               ('0.1', '0.2'): [],
                               ('0.2', '0.2'): [],
                               ('0.3', '0.2'): [],
                               ('0.4', '0.2'): [],
                               ('0.5', '0.2'): [],
                               ('0.6', '0.2'): [],
                               ('0.7', '0.2'): [],
                               ('0.8', '0.2'): [],
                               ('0.9', '0.2'): [],
                               ('1.0', '0.2'): [],

                               ('0.0', '0.3'): [],
                               ('0.1', '0.3'): [],
                               ('0.2', '0.3'): [],
                               ('0.3', '0.3'): [],
                               ('0.4', '0.3'): [],
                               ('0.5', '0.3'): [],
                               ('0.6', '0.3'): [],
                               ('0.7', '0.3'): [],
                               ('0.8', '0.3'): [],
                               ('0.9', '0.3'): [],
                               ('1.0', '0.3'): [],

                               ('0.0', '0.4'): [],
                               ('0.1', '0.4'): [],
                               ('0.2', '0.4'): [],
                               ('0.3', '0.4'): [],
                               ('0.4', '0.4'): [],
                               ('0.5', '0.4'): [],
                               ('0.6', '0.4'): [],
                               ('0.7', '0.4'): [],
                               ('0.8', '0.4'): [],
                               ('0.9', '0.4'): [],
                               ('1.0', '0.4'): [],

                               ('0.0', '0.5'): [],
                               ('0.1', '0.5'): [],
                               ('0.2', '0.5'): [],
                               ('0.3', '0.5'): [],
                               ('0.4', '0.5'): [],
                               ('0.5', '0.5'): [],
                               ('0.6', '0.5'): [],
                               ('0.7', '0.5'): [],
                               ('0.8', '0.5'): [],
                               ('0.9', '0.5'): [],
                               ('1.0', '0.5'): [],

                               ('0.0', '0.6'): [],
                               ('0.1', '0.6'): [],
                               ('0.2', '0.6'): [],
                               ('0.3', '0.6'): [],
                               ('0.4', '0.6'): [],
                               ('0.5', '0.6'): [],
                               ('0.6', '0.6'): [],
                               ('0.7', '0.6'): [],
                               ('0.8', '0.6'): [],
                               ('0.9', '0.6'): [],
                               ('1.0', '0.6'): [],

                               ('0.0', '0.7'): [],
                               ('0.1', '0.7'): [],
                               ('0.2', '0.7'): [],
                               ('0.3', '0.7'): [],
                               ('0.4', '0.7'): [],
                               ('0.5', '0.7'): [],
                               ('0.6', '0.7'): [],
                               ('0.7', '0.7'): [],
                               ('0.8', '0.7'): [],
                               ('0.9', '0.7'): [],
                               ('1.0', '0.7'): [],

                               ('0.0', '0.8'): [],
                               ('0.1', '0.8'): [],
                               ('0.2', '0.8'): [],
                               ('0.3', '0.8'): [],
                               ('0.4', '0.8'): [],
                               ('0.5', '0.8'): [],
                               ('0.6', '0.8'): [],
                               ('0.7', '0.8'): [],
                               ('0.8', '0.8'): [],
                               ('0.9', '0.8'): [],
                               ('1.0', '0.8'): [],

                               ('0.0', '0.9'): [],
                               ('0.1', '0.9'): [],
                               ('0.2', '0.9'): [],
                               ('0.3', '0.9'): [],
                               ('0.4', '0.9'): [],
                               ('0.5', '0.9'): [],
                               ('0.6', '0.9'): [],
                               ('0.7', '0.9'): [],
                               ('0.8', '0.9'): [],
                               ('0.9', '0.9'): [],
                               ('1.0', '0.9'): [],

                               ('0.0', '1.0'): [],
                               ('0.1', '1.0'): [],
                               ('0.2', '1.0'): [],
                               ('0.3', '1.0'): [],
                               ('0.4', '1.0'): [],
                               ('0.5', '1.0'): [],
                               ('0.6', '1.0'): [],
                               ('0.7', '1.0'): [],
                               ('0.8', '1.0'): [],
                               ('0.9', '1.0'): [],
                               ('1.0', '1.0'): [],
                                }  # edit distance = 5 }

        if i in indices:
            # if i == 516:
            #     print("pause")
            subgraph = k_hop_subgraph(i, 3, graph_tensor)[1]
            matrix_sub0_graph = subgraph[0].cpu().numpy().tolist()
            matrix_sub1_graph = subgraph[1].cpu().numpy().tolist()

            gt_graph_matrix = coo_matrix(
                (weights,
                 (matrix_0, matrix_1)), shape=(features.shape[0], features.shape[0])).tocsr()
            weight_subgraph = gt_graph_matrix[matrix_sub0_graph, matrix_sub1_graph].A[0]

            maps, explain_list, explain_nodes, non_explain_list = explain_mapping(matrix_sub0_graph, matrix_sub1_graph,
                                                                                  weight_subgraph, True)
            explain_np = np.arange(0, len(explain_list), 1)
            non_explain_np = np.arange(0, len(non_explain_list), 1)

            for key in edit_distance_rt.keys():
                remove_c = float(key[0])
                adding_c = float(key[1])
                if remove_c == 0 and adding_c == 0:
                    weight_t = remake_weight(nodes_num, matrix_0, matrix_1,
                                             matrix_sub0_graph, matrix_sub1_graph,
                                             weight_subgraph)
                    edit_distance_rt[(key[0], key[1])].append(weight_t)
                    continue

                # removing edges
                sample_ratio_explain = remove_c * len(explain_list) * np.ones_like(explain_np)/len(explain_list)
                # adding edges
                sample_ratio_nonexplain = adding_c * len(explain_list) * np.ones_like(non_explain_np)/len(non_explain_list)
                sample_ratio_nonexplain = np.where(sample_ratio_nonexplain>1,np.ones_like(sample_ratio_nonexplain),sample_ratio_nonexplain)

                # sample
                samples_explain_removing = np.random.binomial(1,sample_ratio_explain,size=(10,sample_ratio_explain.shape[0]))
                samples_nonexplain_removing = np.random.binomial(1,sample_ratio_nonexplain,size=(10,sample_ratio_nonexplain.shape[0]))


                for ii in range(samples_explain_removing.shape[0]):
                    weight_n = weight_subgraph.copy()
                    # remove the edges
                    for idx, edge in enumerate(explain_list):
                        if samples_explain_removing[ii,idx]==1:
                            id_lists = maps[edge]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 0
                    # adding the edges
                    for idx, edge in enumerate(non_explain_list):
                        if samples_nonexplain_removing[ii, idx] == 1:
                            id_lists = maps[edge]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 1

                    weight_t = remake_weight(nodes_num, matrix_0, matrix_1,
                                             matrix_sub0_graph, matrix_sub1_graph,
                                             weight_n)
                    edit_distance_rt[(key[0], key[1])].append(weight_t)

        for key in edit_distance_rt.keys():
            all_edit_distance_remap[key].append(edit_distance_rt[key])
       
    np.save('./data/%s_random_sample_maps_undirected_ratio' % dataset_name, all_edit_distance_remap)


if __name__=="__main__":
    remap_ed_k_ratio()



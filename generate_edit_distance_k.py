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

    def get_adding_edge_list(graph,explain_nodes,explain_list_t):
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
    for i in tqdm(range(len(graphs))): # len(graphs) indices

        # if i > 10 :
        #     continue

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
                pos = visulaized(graph,weights,name='%d_k_%d_exp'%(i,0))
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
                # temp_remove_one_edge.append(weight_n)

            # by adding edge
            adding_k_1_edge_list = get_adding_edge_list(graph,explain_nodes,explain_list)
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
                # temp_adding_one_edge.append(weight_n)
            edit_distance_lists[k] = set(edit_distance_lists[k])
            for idx,weight_n in enumerate(list(edit_distance_lists[k])):
                if i < 10 and visualization:
                    visulaized(graph, weight_n, name='%d_k_%d_%d' % (i, k,idx), posx=pos)


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

                    # option two ,adding one edge from non_explanation subgraphs
                    adding_k_1_edge_list_t = get_adding_edge_list(graph,explain_nodes_t,explain_list_t)
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
                edit_distance_lists[k] = set(edit_distance_lists[k])

                for idx, weight_n in enumerate(list(edit_distance_lists[k])):
                    if i < 10 and visualization:
                        visulaized(graph, weight_n, name='%d_k_%d_%d' % (i, k, idx), posx=pos)

        all_edit_distance_lists.append(edit_distance_lists)
    np.save('./data/mutag_sample_weights',all_edit_distance_lists)


def remap_ed_k():
    undirect = True
    visualization = False
    dataset_name = 'ba2'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)

    all_edit_distance_lists = []
                               # remove, adding
    all_edit_distance_remap = {(0, 0): [],
                               (0, 1): [],  # edit distance = 1
                               (0, 2): [],
                               (0, 3): [],  # edit distance = 2
                               (0, 4): [],
                               (0, 5): [],

                               (1, 0): [],
                               (1, 1): [],  # edit distance = 1
                               (1, 2): [],
                               (1, 3): [],  # edit distance = 2
                               (1, 4): [],
                               (1, 5): [],

                               (2, 0): [],
                               (2, 1): [],  # edit distance = 1
                               (2, 2): [],
                               (2, 3): [],  # edit distance = 2
                               (2, 4): [],
                               (2, 5): [],

                               (3, 0): [],
                               (3, 1): [],  # edit distance = 1
                               (3, 2): [],
                               (3, 3): [],  # edit distance = 2
                               (3, 4): [],
                               (3, 5): [],

                               (4, 0): [],
                               (4, 1): [],  # edit distance = 1
                               (4, 2): [],
                               (4, 3): [],  # edit distance = 2
                               (4, 4): [],
                               (4, 5): [],

                               (5, 0): [],
                               (5, 1): [],  # edit distance = 1
                               (5, 2): [],
                               (5, 3): [],  # edit distance = 2
                               (5, 4): [],
                               (5, 5): [],
                               }

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

        # if i > 10 :
        #     continue

        edit_distance_lists = [[], [], [], [], [], [], []]

        if dataset_name == "mutag":
            graph = explanation_labels[0][i]
        else:
            graph = graphs[i]
        weights = explanation_labels[1][i]

        matrix_0 = graph[0]
        matrix_1 = graph[1]
        # weights = label[1]
        edit_distance_lists[0].append(tuple(weights.tolist()))
        if undirect:
            maps, explain_list, explain_nodes, non_explain_list = explain_mapping(matrix_0, matrix_1, weights, True)
            # visualize the gt
            # visulaized_three(graph,weights,name='%d'%i)
            # exit(0)
            # visulaized(graph,np.ones_like(weights),name='%d_graph'%i)
            if i < 10 and visualization:
                pos = visulaized(graph, weights, name='%d_k_%d_exp' % (i, 0))
            # visulaized(graph,1-weights,name='%d_nonexp'%i)
            # generate sample
            # by deleting edge
            k = 1
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
                # temp_remove_one_edge.append(weight_n)

            # by adding edge
            adding_k_1_edge_list = get_adding_edge_list(graph, explain_nodes, explain_list)
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
                # temp_adding_one_edge.append(weight_n)
            edit_distance_lists[k] = list(set(edit_distance_lists[k]))
            for idx, weight_n in enumerate(list(edit_distance_lists[k])):
                if i < 10 and visualization:
                    visulaized(graph, weight_n, name='%d_k_%d_%d' % (i, k, idx), posx=pos)

            ############################################################################################################
            #  k = 2
            #  deleting one edge adding one edge
            for k in range(2, 7):
                if k == 5:
                    pass
                # k = 2
                # 1. remove one edge from remove one edge weights
                for weight_t in edit_distance_lists[k - 1]:
                    # relist the explanation nodes,
                    # maps remain the same
                    explain_list_t, explain_nodes_t, non_explain_list_t = explain_mapping(matrix_0, matrix_1, weight_t)

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

                    # option two ,adding one edge from non_explanation subgraphs
                    adding_k_1_edge_list_t = get_adding_edge_list(graph, explain_nodes_t, explain_list_t)
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
                edit_distance_lists[k] = list(set(edit_distance_lists[k]))

                for idx, weight_n in enumerate(list(edit_distance_lists[k])):
                    if i < 10 and visualization:
                        visulaized(graph, weight_n, name='%d_k_%d_%d' % (i, k, idx), posx=pos)

        all_edit_distance_lists.append(edit_distance_lists)

        edit_distance_rt =        {(0, 0): [],
                                   (1, 0): [],  # edit distance = 1
                                   (0, 1): [],
                                   (2, 0): [],  # edit distance = 2
                                   (0, 2): [],
                                   (1, 1): [],
                                   (3, 0): [],  # edit distance = 3
                                   (2, 1): [],
                                   (1, 2): [],
                                   (0, 3): [],
                                   (4, 0): [],  # edit distance = 4
                                   (3, 1): [],
                                   (2, 2): [],
                                   (1, 3): [],
                                   (0, 4): [],
                                   (5, 0): [],  # edit distance = 5
                                   (4, 1): [],
                                   (3, 2): [],
                                   (2, 3): [],
                                   (1, 4): [],
                                   (0, 5): [],
                                   (6, 0): [],  # edit distance = 6
                                   (5, 1): [],
                                   (4, 2): [],
                                   (3, 3): [],
                                   (2, 4): [],
                                   (1, 5): [],
                                   (0, 6): [],
                                   }

        original_weights = weights
        for list_t in edit_distance_lists:
            for weight in list_t:
                ori_weight_np = np.array(original_weights)
                weight_np = np.array(weight)
                mask_explaination = np.where(ori_weight_np>0,np.ones_like(original_weights),np.zeros_like(original_weights))
                explanation_difference = mask_explaination * np.abs(ori_weight_np-weight_np)
                non_explanation_difference = (1-mask_explaination) * np.abs(ori_weight_np - weight_np)

                remove_number =  int(np.sum(explanation_difference)//2)
                add_number =  int(np.sum(non_explanation_difference)//2)

                edit_distance_rt[remove_number,add_number].append(list(weight))

        for key in edit_distance_rt.keys():
            all_edit_distance_remap[key].append(edit_distance_rt[key])


    # np.save('./data/%s_sample_weights_k'%dataset_name, all_edit_distance_lists)

    np.save('./data/%s_sample_weights_remap_k'%dataset_name, all_edit_distance_remap)




if __name__=="__main__":
    remap_ed_k()
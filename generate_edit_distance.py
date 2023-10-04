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
    if store:
        plt.savefig("./graph_%s.png"%(name))
        plt.savefig("./graph_%s.pdf"%(name))

def visulaized_three(graph,weights,store=True,name='graph'):
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

    # labels = nx.get_edge_attributes(G, 'weight')
    edges, weights_c = zip(*nx.get_edge_attributes(G, 'color').items())

    pos = nx.spring_layout(G)
    #nx.draw_networkx(G,pos,edge_color=weights_c,node_size=node_size) # node_size=node_size,font_size=font_size,
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size)
    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k')

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.savefig("./data/gt_show/graph_%s.png"%(name))
        plt.savefig("./data/gt_show/graph_%s.pdf"%(name))

    plt.cla()

    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size,edge_color='w',node_color='w')
    nx.draw_networkx_nodes(G, pos, nodelist=explain_nodes, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list, edge_color='r')
    # nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list, edge_color='k')

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.savefig("./data/gt_show/graph_%s_explain.png"%(name))
        plt.savefig("./data/gt_show/graph_%s_explain.pdf"%(name))

    plt.cla()
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size, edge_color='w', node_color='w')
    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    # nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k')

    # nx.draw_networkx_nodes(G, pos, nodelist=explain_nodes, node_size=node_size, edge_color='w', node_color='w')
    # nx.draw_networkx_nodes(G, pos, nodelist=non_explain_nodes, node_size=node_size)
    # nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list, edge_color='r')
    # nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list, edge_color='k')
    # nx.draw_networkx(G,pos,nodelist=non_explain_nodes,edge_color='k', node_size=node_size,
    #                  edgelist=non_explain_edge_list)  #  node_size=node_size,font_size=font_size,

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.savefig("./data/gt_show/graph_%s_nonexplain.png"%(name))
        plt.savefig("./data/gt_show/graph_%s_nonexplain.pdf"%(name))


def visulaized_graph_three(graph,weights,dirpath,pos=None,store=True,name='graph'):
    plt.cla()
    dpi = 300
    figure_size = (10,5)
    node_size = 300
    # font_size = 10
    weight_font_size = 5

    fig = plt.figure( figsize=figure_size,dpi=dpi)  # , dpi=60

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

    # labels = nx.get_edge_attributes(G, 'weight')
    edges, weights_c = zip(*nx.get_edge_attributes(G, 'color').items())

    if pos == None:
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)
        # pos = nx.fruchterman_reingold_layout(G)
    else:
        pass
    #nx.draw_networkx(G,pos,edge_color=weights_c,node_size=node_size) # node_size=node_size,font_size=font_size,
    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size,node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k')
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r')

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        save_name = dirpath+'/graph_%s.png'%(name)
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.pdf'))

    plt.cla()

    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size,edge_color='w',node_color='w')
    nx.draw_networkx_nodes(G, pos, nodelist=explain_nodes, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list, edge_color='r')
    # nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list, edge_color='k')

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')

        save_name = dirpath+'/graph_%s_explain.png'%(name)
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.pdf'))

    plt.cla()
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size, edge_color='w', node_color='w')
    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    # nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k')

    # nx.draw_networkx_nodes(G, pos, nodelist=explain_nodes, node_size=node_size, edge_color='w', node_color='w')
    # nx.draw_networkx_nodes(G, pos, nodelist=non_explain_nodes, node_size=node_size)
    # nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list, edge_color='r')
    # nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list, edge_color='k')
    # nx.draw_networkx(G,pos,nodelist=non_explain_nodes,edge_color='k', node_size=node_size,
    #                  edgelist=non_explain_edge_list)  #  node_size=node_size,font_size=font_size,

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        save_name = dirpath+'/graph_%s_nonexplain.png'%(name)
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.pdf'))
    plt.close(fig)
    return pos

def visulaized_nodes_three(graph,weights,dirpath,pos=None,store=True,name='nodes'):
    plt.cla()
    dpi = 300
    figure_size = (10,5)
    node_size = 300
    # font_size = 10
    weight_font_size = 5

    fig = plt.figure( figsize=figure_size,dpi=dpi)  # , dpi=60

    G = nx.DiGraph()
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

    # labels = nx.get_edge_attributes(G, 'weight')
    edges, weights_c = zip(*nx.get_edge_attributes(G, 'color').items())

    if pos == None:
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)
        # pos = nx.fruchterman_reingold_layout(G)
    else:
        pass
    #nx.draw_networkx(G,pos,edge_color=weights_c,node_size=node_size) # node_size=node_size,font_size=font_size,
    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size,node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k',arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r',arrows=False)

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        save_name = dirpath+'/graph_%s.png'%(name)
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.pdf'))

    plt.cla()

    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size,edge_color='w',node_color='w')
    nx.draw_networkx_nodes(G, pos, nodelist=explain_nodes, node_size=node_size,node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list, edge_color='r',arrows=False)
    # nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list, edge_color='k')

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')

        save_name = dirpath+'/graph_%s_explain.png'%(name)
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.pdf'))

    plt.cla()
    nx.draw_networkx_nodes(G, pos,nodelist=explain_nodes,node_size=node_size, edge_color='w', node_color='w')
    nx.draw_networkx_nodes(G, pos,nodelist=non_explain_nodes,node_size=node_size)
    # nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list,edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list,edge_color='k',arrows=False)

    # nx.draw_networkx_nodes(G, pos, nodelist=explain_nodes, node_size=node_size, edge_color='w', node_color='w')
    # nx.draw_networkx_nodes(G, pos, nodelist=non_explain_nodes, node_size=node_size)
    # nx.draw_networkx_edges(G, pos, edgelist=explain_edge_list, edge_color='r')
    # nx.draw_networkx_edges(G, pos, edgelist=non_explain_edge_list, edge_color='k')
    # nx.draw_networkx(G,pos,nodelist=non_explain_nodes,edge_color='k', node_size=node_size,
    #                  edgelist=non_explain_edge_list)  #  node_size=node_size,font_size=font_size,

    if store:
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        save_name = dirpath+'/graph_%s_nonexplain.png'%(name)
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.pdf'))
    plt.close(fig)
    return pos

def draw_figures():
    dataset_name = 'ba2'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)


    edit_distance_lists = [[], [], [], [], []]

    for i in range(len(graphs)):
        graph = graphs[i]
        weights = explanation_labels[1][i]

        matrix_0 = graph[0]
        matrix_1 = graph[1]
        #weights = label[1]
        edit_distance_lists[0].append(weights)

        visulaized_three(graph,weights,name='%d'%i)

def draw_graphs_figures_editdistance():
    dataset_name = 'ba2'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)
    path = './data/%s_random_sample_maps.npy'%dataset_name
    edit_distance_map = np.load(path, allow_pickle=True).item()

    if dataset_name == "mutag":
        graphs = explanation_labels[0]

    positions_path = r'./for_show/%s/pos.npy'%dataset_name
    if os.path.exists(positions_path):
        positions = np.load(positions_path,allow_pickle=True).item()
        positions_save = positions
    else:
        positions = None

        positions_save = {}

    keys_list = [(0,0)] #,(0,2),(0,3),(0,4),(1,1),(2,2),(3,3),(1,0),(2,0),(3,0),(4,0),(0,1)] # (0,0),(0,2),(0,3),(0,4),(1,1),(2,2),(3,3),(1,0),(2,0),(3,0),(4,0),(0,1)
    for key in keys_list:
        save_dir = r'./for_show/%s/%d_%d'%(dataset_name,key[0],key[1])
        if os.path.exists(save_dir):
            pass
        else:
            os.mkdir(save_dir)

        edit_distance_lists = edit_distance_map[key]
        count = 0
        for i in tqdm(range(len(graphs))):
            if i in indices:
                graph = graphs[i]
                for ii,weight in enumerate(edit_distance_lists[i]):

                    if positions == None and i not in positions_save.keys():
                        pos = visulaized_graph_three(graph, weight,save_dir, name='%d_sample_%d' % (i,ii))
                        positions_save[i] = pos
                    else:
                        visulaized_graph_three(graph, weight, save_dir,
                            pos = positions_save[i],name='%d_sample_%d' % (i, ii))

                    if ii>9:
                        break

                count +=1
                if count>10:
                    break


    if positions== None:
        np.save(positions_path.replace('.npy',''),positions_save)


def draw_nodes_figures_editdistance():
    dataset_name = 'syn3'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    matrix_0 = graphs[0]
    matrix_1 = graphs[1]

    features = torch.tensor(features)
    labels = torch.tensor(labels)
    graphs_tensor = torch.tensor(graphs)


    explanation_labels, indices = load_dataset_ground_truth(dataset_name)
    path = './data/%s_random_sample_maps.npy'%dataset_name
    edit_distance_map = np.load(path, allow_pickle=True).item()

    if dataset_name == "mutag":
        graphs = explanation_labels[0]

    positions_path = r'./for_show/%s/pos.npy'%dataset_name
    if os.path.exists(positions_path):  #  False: #
        positions = np.load(positions_path,allow_pickle=True).item()
        positions_save = positions
    else:
        positions = None
        positions_save = {}
    # ,(0,2),(0,3),(0,4),(1,1),(2,2),(3,3),(1,0),(2,0),(3,0),(4,0),(0,1)
    keys_list = [(0,0)] # (0,0),(0,2),(0,3),(0,4),(1,1),(2,2),(3,3),(1,0),(2,0),(3,0),(4,0),(0,1)
    for key in keys_list:
        save_dir = r'./for_show/%s/%d_%d'%(dataset_name,key[0],key[1])
        if os.path.exists(save_dir):
            pass
        else:
            os.mkdir(save_dir)

        edit_distance_lists = edit_distance_map[key]
        count = 0
        for i in tqdm(range(len(edit_distance_lists))):
            if i in indices:
                graph = graphs
                # sample
                subset, edge_index, inv, edge_mask = k_hop_subgraph(i, num_hops=3,
                                                                    edge_index=graphs_tensor)
                subgrpah_nodes = len(subset)
                subgraph_edges = edge_index.shape[1]
                # max_nodes = int((1 - node_sp) * subgrpah_nodes)

                for ii,weight in enumerate(edit_distance_lists[i]):

                    new_weight = weight[edge_mask]
                    new_subgraph = edge_index.cpu().numpy()
                    if positions == None and i not in positions_save.keys():
                        pos = visulaized_nodes_three(new_subgraph, new_weight,save_dir, name='%d_sample_%d' % (i,ii))
                        positions_save[i] = pos
                    else:
                        visulaized_nodes_three(new_subgraph, new_weight, save_dir,
                            pos = positions_save[i],name='%d_sample_%d' % (i, ii))

                    if ii>1:
                        break

                count +=1
                # if count>10:
                #     break


    if positions== None:
        np.save(positions_path.replace('.npy',''),positions_save)


def edit_distance_k_gen(dir='./data',undirect=True):

    dataset_name = 'ba2'

    graphs, features, labels, _, _, test_mask = load_dataset(dataset_name)
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)

    edit_distance_lists=[[],[],[],[],[]]

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
    for i in range(len(graphs)):
        graph = graphs[i]
        weights = explanation_labels[1][i]

        matrix_0 = graph[0]
        matrix_1 = graph[1]
        #weights = label[1]
        edit_distance_lists[0].append(weights)
        if undirect:
            maps, explain_list, explain_nodes, non_explain_list = explain_mapping(matrix_0, matrix_1, weights,True)
            # visualize the gt
            # visulaized_three(graph,weights,name='%d'%i)
            # exit(0)
            # visulaized(graph,np.ones_like(weights),name='%d_graph'%i)
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
            for i, com in enumerate(explain_indexs_combin):
                weight_n = weights.copy()
                for c in com:  # (min_node_id, max_node_id)
                    id_lists = maps[c]  # get two edges id
                    for id in id_lists:
                        weight_n[id] = 0
                edit_distance_lists[k].append(weight_n)
                # temp_remove_one_edge.append(weight_n)

            # by adding edge
            adding_k_1_edge_list = get_adding_edge_list(graph,explain_nodes)
            adding_k_1_edge_list = list(set(adding_k_1_edge_list))
            explain_indexs_combin = combinations(adding_k_1_edge_list, k)
            explain_indexs_combin = list(explain_indexs_combin)
            # temp_adding_one_edge = []
            for i, com in enumerate(explain_indexs_combin):
                weight_n = weights.copy()
                for c in com:  # (min_node_id, max_node_id)
                    id_lists = maps[c]  # get two edges id
                    for id in id_lists:
                        weight_n[id] = 1
                edit_distance_lists[k].append(weight_n)
                # temp_adding_one_edge.append(weight_n)
            edit_distance_lists[k] = set(edit_distance_lists[k])

            ############################################################################################################
            #  k = 2
            #  deleting one edge adding one edge
            for k in range(2,6):
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
                    for i, com in enumerate(explain_indexs_combin):
                        weight_n = weight_t.copy()
                        for c in com:  # (min_node_id, max_node_id)
                            id_lists = maps[c]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 0
                        edit_distance_lists[k].append(weight_n)

                    # option two , remove one edge again, only from motifs
                    adding_k_1_edge_list_t = get_adding_edge_list(graph,explain_nodes_t)
                    adding_k_1_edge_list_t = list(set(adding_k_1_edge_list_t) - set(explain_list))
                    explain_indexs_combin = combinations(adding_k_1_edge_list_t, 1)
                    explain_indexs_combin = list(explain_indexs_combin)
                    for i, com in enumerate(explain_indexs_combin):
                        weight_n = weight_t.copy()
                        for c in com:  # (min_node_id, max_node_id)
                            id_lists = maps[c]  # get two edges id
                            for id in id_lists:
                                weight_n[id] = 1
                        edit_distance_lists[k].append(weight_n)
                edit_distance_lists[k] = set(edit_distance_lists[k])

    np.save('./data/sample_weights',edit_distance_lists)

if __name__=="__main__":
    draw_nodes_figures_editdistance()
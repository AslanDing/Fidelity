import os
import torch
import random
import numpy as np
# from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset,Data, DataLoader
# from torch.utils.data import Dataset
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset,load_dataset_contrastive
from ExplanationEvaluation.models.model_selector_xgnn import model_selector
from ExplanationEvaluation.datasets.utils import preprocess_features, preprocess_adj, adj_to_edge_index, load_real_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

import scipy.sparse as sp

class GraphDataset(InMemoryDataset):
    def __init__(self,graphs):
        super().__init__()
        self.graphs = graphs

    def get(self, idx: int):
        pass

def create_data_list(graphs, features, labels, mask):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in indices:
        x = torch.tensor(features[i]).cuda()
        edge_index = torch.tensor(graphs[i]).cuda()
        y = torch.tensor(labels[i].argmax()).cuda()
        data = Data(x=x, edge_index=edge_index, y=y).cuda()
        data_list.append(data)
    return data_list

def create_data_list_extend_ex(filtered, graphs, features, labels, edge_label,only_gt = False,graph = True):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    # indices = np.argwhere(mask).squeeze()
    data_list = []
    if graph:
        length = len(edge_label[0])
    else:
        length = len(edge_label)

    for i in range(length):
    # for i in filtered:
        x = torch.tensor(features[i]).cuda()
        # here
        if graph:
            src_graph = sp.coo_matrix((edge_label[1][i],(graphs[i][0],graphs[i][1]))).tocsr()
            index = src_graph.nonzero()

            new_list = index[0].tolist()
            new_list.extend(index[1].tolist())
            set_index = list(set(new_list))

            new_src_graph =src_graph[set_index,:]
            new_src_graph =new_src_graph[:,set_index]
            new_index = new_src_graph.nonzero()
            x = x[set_index]

            edge_index = torch.tensor(new_index, dtype=torch.long).cuda()
            y = torch.tensor(labels[i].argmax()).cuda()
            data = Data(x=x, edge_index=edge_index, y=y).cuda()
            data_list.append(data)

        else:
            src_graph = sp.coo_matrix((edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()
            index = src_graph.nonzero()
            edge_index = torch.tensor(index,dtype=torch.long).cuda()
            y = torch.tensor(labels[i].argmax()).cuda()
            data = Data(x=x, edge_index=edge_index, y=y).cuda()
            data_list.append(data)

        if not only_gt:
            x = torch.tensor(features[i]).cuda()
            # here
            if graph:
                src_graph = sp.coo_matrix(( 1- edge_label[1][i], (graphs[i][0], graphs[i][1]))).tocsr()
            else:
                src_graph = sp.coo_matrix(( 1- edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()

            index = src_graph.nonzero()
            edge_index = torch.tensor(index,dtype=torch.long).cuda()
            y = torch.tensor(labels[i].argmin()).cuda()
            data = Data(x=x, edge_index=edge_index, y=y).cuda()

            data_list.append(data)

    return data_list

def create_data_list_extend(filtered, graphs, features, labels, edge_label,only_gt = False,graph = True):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    # indices = np.argwhere(mask).squeeze()
    data_list = []
    if graph:
        length = len(edge_label[0])
    else:
        length = len(edge_label)

    # for i in range(length):
    for i in filtered:
        x = torch.tensor(features[i]).cuda()
        # here
        if graph:
            # src_graph = sp.coo_matrix((edge_label[1][i],(graphs[i][0],graphs[i][1]))).tocsr()
            src_graph = sp.coo_matrix((1-edge_label[1][i],(edge_label[0][i][0],edge_label[0][i][1]))).tocsr()
        else:
            src_graph = sp.coo_matrix((edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()
        index = src_graph.nonzero()
        edge_index = torch.tensor(index,dtype=torch.long).cuda()
        y = torch.tensor(labels[i].argmax()).cuda()
        data = Data(x=x, edge_index=edge_index, y=y).cuda()
        data_list.append(data)

        if not only_gt:
            x = torch.tensor(features[i]).cuda()
            # here
            if graph:
                # src_graph = sp.coo_matrix((edge_label[1][i],(graphs[i][0],graphs[i][1]))).tocsr()
                src_graph = sp.coo_matrix(( 1- edge_label[1][i], (edge_label[i][0], edge_label[i][1]))).tocsr()
            else:
                src_graph = sp.coo_matrix(( 1- edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()

            index = src_graph.nonzero()
            edge_index = torch.tensor(index,dtype=torch.long).cuda()
            y = torch.tensor(labels[i].argmin()).cuda()
            data = Data(x=x, edge_index=edge_index, y=y).cuda()

            data_list.append(data)

    return data_list
def create_data_list_extend_backup(filtered, graphs, features, labels, edge_label,only_gt = False):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    # indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in range(len(edge_label)):
    # for i in filtered:
        x = torch.tensor(features[i]).cuda()
        # here
        src_graph = sp.coo_matrix((edge_label[i],(graphs[i][0],graphs[i][1]))).tocsr()
        index = src_graph.nonzero()
        edge_index = torch.tensor(index,dtype=torch.long).cuda()
        y = torch.tensor(labels[i].argmax()).cuda()
        data = Data(x=x, edge_index=edge_index, y=y).cuda()
        data_list.append(data)

        if not only_gt:
            x = torch.tensor(features[i]).cuda()
            # here
            src_graph = sp.coo_matrix(( 1- edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()
            index = src_graph.nonzero()
            edge_index = torch.tensor(index,dtype=torch.long).cuda()
            y = torch.tensor(labels[i].argmin()).cuda()
            data = Data(x=x, edge_index=edge_index, y=y).cuda()

            data_list.append(data)

    return data_list

def create_data_list_contrastive(filtered, graphs, features, labels, edge_label):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    # indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in range(len(edge_label)):
    # for i in filtered:
        rate = random.random()

        src_graph = sp.coo_matrix((edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()
        index = src_graph.nonzero()

        op_src_graph = sp.coo_matrix((1 - edge_label[i], (graphs[i][0], graphs[i][1]))).tocsr()
        op_index = op_src_graph.nonzero()
        length = op_index[0].shape[0]
        a = np.arange(0,length,1)
        select_list = np.random.choice(a,int(length*rate))

        x = torch.tensor(features[i]).cuda()
        # here
        new_list = [op_index[0][select_list],op_index[1][select_list]]
        new_list[0] = np.concatenate([new_list[0],index[0]],axis=0)
        new_list[1] = np.concatenate([new_list[1],index[1]],axis=0)
        edge_index = torch.tensor(new_list,dtype=torch.long).cuda()
        y = torch.tensor(labels[i].argmax()).cuda()
        data = Data(x=x, edge_index=edge_index, y=y).cuda()
        data_list.append(data)



        x = torch.tensor(features[i]).cuda()
        # here
        new_list = [op_index[0][select_list],op_index[1][select_list]]
        edge_index = torch.tensor(new_list,dtype=torch.long).cuda()
        y = torch.tensor(labels[i].argmin()).cuda()
        data = Data(x=x, edge_index=edge_index, y=y).cuda()
        data_list.append(data)

    return data_list

def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc


def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper
    :param dataset: str, the dataset
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :retunrs: None
    """
    save_dir = f"./checkpoints/{paper}/{dataset}"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def load_best_model(best_epoch, paper, dataset, model, eval_enabled):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper
    :param dataset: str, the dataset
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :return: model with pramaters taken from the checkpoint
    """
    print(best_epoch)
    if best_epoch == -1:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_model")
    else:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model

def train_node_contrastives(_dataset, _paper, args):
    """
        Train a explainer to explain node classifications
        :param _dataset: the dataset we wish to use for training
        :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
        :param args: a dict containing the relevant model arguements
        """
    # for node classification
    adj, features, label, train_mask, val_mask, test_mask, adj_expl_label = load_dataset_contrastive(_dataset)
    # graph, features, labels, train_mask, val_mask, test_mask, graph_extend = load_dataset(_dataset)
    model = model_selector(_paper, _dataset, False)
    model.cuda()

    x = torch.tensor(features).cuda()
    labels = torch.tensor(label).cuda()
    graph = preprocess_adj(adj)[0].astype('int64').T
    edge_index = torch.tensor(graph).cuda()

    # Define graph
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    # source
    label_mask = adj_expl_label.nonzero()
    no_label_mask = (1 - adj_expl_label).nonzero()
    length = no_label_mask[0].shape[0]

    for epoch in range(0, args.epochs):
        model.train()
        optimizer.zero_grad()

        if epoch % 3 == 0:
            out = model(x, edge_index)
        elif epoch % 3 == 1:
            # for postive label + random mask of no_lael_mask
            rate = random.random()
            # label random select
            indexes = np.arange(0, length, 1).tolist()
            select = np.random.choice(indexes, int(rate * (length)))

            label__1_list = [no_label_mask[0][select], no_label_mask[1][select]]
            label__1_list[0] = np.concatenate([label__1_list[0],label_mask[0]],axis=0)
            label__1_list[1] = np.concatenate([label__1_list[1],label_mask[1]],axis=0)

            edge_index_extend = torch.tensor(label__1_list).cuda()
            out = model(x, edge_index_extend)

        else:
            # for postive label + random mask of no_lael_mask
            rate = random.random()
            # label random select
            indexes = np.arange(0, length, 1).tolist()
            select = np.random.choice(indexes, int(rate * (length)))
            label__0_list = [no_label_mask[0][select], no_label_mask[1][select]]
            edge_index_extend = torch.tensor(label__0_list).cuda()
            out = model(x, edge_index_extend)


        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper + "_contrastive", _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper + "_contrastive", _dataset, model, args.eval_enabled)
    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper + "_contrastive", _dataset, model, train_acc, val_acc, test_acc)
def train_node_contrastive(_dataset, _paper, args, test_gt = True):
    """
        Train a explainer to explain node classifications
        :param _dataset: the dataset we wish to use for training
        :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
        :param args: a dict containing the relevant model arguements
        """
    # for node classification
    adj, features, label, train_mask, val_mask, test_mask, adj_expl_label = load_dataset_contrastive(_dataset)
    # graph, features, labels, train_mask, val_mask, test_mask, graph_extend = load_dataset(_dataset)
    model = model_selector(_paper, _dataset, True)

    explanation_labels, indices = load_dataset_ground_truth(_dataset)
    # if extension: indices = np.argwhere(test_mask).squeeze()

    # path = './checkpoints/contrastive_pretrain/syn1/bashapes_150_inter_explain1t_sp2_4_70_maxsub.pth'
    # dict = torch.load(path)
    # model.load_state_dict(dict)

    model.cuda()

    x = torch.tensor(features).cuda()
    labels = torch.tensor(label).cuda()
    graph = preprocess_adj(adj)[0].astype('int64').T
    edge_index = torch.tensor(graph).cuda()

    # Define graph
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    # source
    label_mask = adj_expl_label.nonzero()
    no_label_mask = (1 - adj_expl_label).nonzero()
    length = no_label_mask[0].shape[0]

    epoch = 0
    loss = 0
    with torch.no_grad():
        out = model(x, edge_index)
    # Evaluate train
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])

    print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

    if val_acc > best_val_acc:  # New best results
        print("Val improved")
        best_val_acc = val_acc
        best_epoch = epoch
        store_checkpoint(_paper + "_contrastive", _dataset, model, train_acc, val_acc, test_acc, best_epoch)


    # for epoch in range(0, args.epochs):
    for epoch in range(0, 0):
        model.train()
        optimizer.zero_grad()

        if epoch % 3 == 0:
            out = model(x, edge_index)
            loss = criterion(out[train_mask], labels[train_mask])
        elif epoch % 3 == 1:
            # for postive label + random mask of no_lael_mask
            rate = random.random()
            # label random select
            indexes = np.arange(0, length, 1).tolist()
            select = np.random.choice(indexes, int(rate * (length)))

            label__1_list = [no_label_mask[0][select], no_label_mask[1][select]]
            label__1_list[0] = np.concatenate([label__1_list[0], label_mask[0]], axis=0)
            label__1_list[1] = np.concatenate([label__1_list[1], label_mask[1]], axis=0)

            edge_index_extend = torch.tensor(label__1_list).cuda()
            out = model(x, edge_index_extend)

            # label only
            train_masks = np.where(label > 1, train_mask, np.zeros_like(train_mask)).astype(dtype=bool)

            loss = criterion(out[train_masks], labels[train_masks])
        else:
            # for postive label + random mask of no_lael_mask
            rate = random.random()
            # label random select
            indexes = np.arange(0, length, 1).tolist()
            select = np.random.choice(indexes, int(rate * (length)))
            label__0_list = [no_label_mask[0][select], no_label_mask[1][select]]
            edge_index_extend = torch.tensor(label__0_list).cuda()
            out = model(x, edge_index_extend)

            # no label only
            train_masks = np.where(label < 1, train_mask, np.zeros_like(train_mask)).astype(dtype=bool)
            loss = criterion(out[train_masks], labels[train_masks])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper + "_contrastive", _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper + "_contrastive", _dataset, model, args.eval_enabled)

    # test all gt
    if test_gt:
        with torch.no_grad():
            gt_mask=np.zeros_like(train_mask).astype(bool)
            for idx in  indices: #range(adj_expl_label.shape[0]):
                gt_mask[idx]=True

            out = model(x, edge_index,
                        edge_weights= 1- torch.from_numpy(explanation_labels[1]).float().to(x.device) )
            out_temp = torch.softmax(out[gt_mask],dim=1)
            label_temp = labels[gt_mask]
            print(label_temp.min(),label_temp.max())
            #indx = torch.argwhere(label_temp==1)
            temp = out_temp.mean(dim=0)
            print(temp.cpu().detach().numpy())
            gt_acc = evaluate(out[gt_mask], labels[gt_mask])
            print(f"final gt_acc:{gt_acc}")

    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper + "_contrastive", _dataset, model, train_acc, val_acc, test_acc)

def train_node_extend(_dataset, _paper, args):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graph, features, labels, train_mask, val_mask, test_mask,graph_extend = load_dataset(_dataset,extend=True)
    model = model_selector(_paper, _dataset, False)

    x = torch.tensor(features).cuda()
    edge_index = torch.tensor(graph).cuda()
    edge_index_extend = torch.tensor(graph_extend).cuda()
    edge_index_extendx = edge_index_extend+ x.shape[0]
    new_edge_index = torch.concat([edge_index,edge_index_extendx],dim=1)
    new_x = torch.concat([x,x],dim=0)

    labels = torch.tensor(labels).cuda()
    # new_train_mask = torch.where(labels>0,train_mask,torch.zeros_like(train_mask))
    # new_val_mask = torch.where(labels>0,val_mask,torch.zeros_like(val_mask))
    # new_test_mask = torch.where(labels>0,test_mask,torch.zeros_like(test_mask))

    new_labels = torch.concat([labels,labels],dim=-1)

    another_mask = np.where(labels.cpu().numpy()>=1,train_mask,np.zeros_like(train_mask))
    new_train_mask = np.concatenate([train_mask,another_mask],axis=-1)
    another_mask = np.where(labels.cpu().numpy()>=1,val_mask,np.zeros_like(val_mask))
    new_val_mask = np.concatenate([val_mask,another_mask],axis=-1)
    another_mask = np.where(labels.cpu().numpy()>=1,test_mask,np.zeros_like(test_mask))
    new_test_mask = np.concatenate([test_mask,another_mask],axis=-1)

    # Define graph
    print(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        model.train()
        optimizer.zero_grad()
        # if epoch%2 == 0:
        #     out = model(x, edge_index)
        # else:
        #     out = model(x, edge_index_extend)
        out = model(new_x, new_edge_index)
        loss = criterion(out[new_train_mask], new_labels[new_train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(new_x, new_edge_index)

        # Evaluate train
        train_acc = evaluate(out[new_train_mask], new_labels[new_train_mask])
        test_acc = evaluate(out[new_test_mask], new_labels[new_test_mask])
        val_acc = evaluate(out[new_val_mask], new_labels[new_val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper+"_extend", _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper+"_extend", _dataset, model, args.eval_enabled)

    out = model(x, edge_index_extend)
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"finalextend train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")


    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")



    store_checkpoint(_paper+"_extend", _dataset, model, train_acc, val_acc, test_acc)

def train_node(_dataset, _paper, args):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    model = model_selector(_paper, _dataset, False)
    model.cuda()

    x = torch.tensor(features).cuda()
    edge_index = torch.tensor(graph).cuda()
    labels = torch.tensor(labels).cuda()

    # Define graph
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc: # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled)
    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)


def train_graph(_dataset, _paper, args):
    """
    Train a explainer to explain graph classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    if _dataset == 'ba2':
        # _dataset = "ba2_2"
        model = model_selector(_paper, _dataset, False)
    else:
        model = model_selector(_paper, _dataset, False)
    model.cuda()

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            optimizer.step()

        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss += criterion(out, data.y)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))
            train_loss = float(loss) / int(len(train_loader))

            eval_data = next(iter(test_loader)) # Loads all test samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader)) # Loads all eval samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            val_acc = evaluate(out, eval_data.y)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled)

    with torch.no_grad():
        train_sum = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))

        eval_data = next(iter(test_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        val_acc = evaluate(out, eval_data.y)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)


def train_graph_extend(_dataset, _paper, args):
    """
    Train a explainer to explain graph classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    (np_edge_list, np_edge_labels), filtered = load_dataset_ground_truth(_dataset)
    train_set = create_data_list(graphs, features, labels, train_mask)
    train_set_extend = create_data_list_extend(filtered, graphs, features, labels, np_edge_labels)  # may be all the graoundtruth
    train_set.extend(train_set_extend)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    model = model_selector(_paper, _dataset, False)
    model.cuda()
    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            optimizer.step()

        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss += criterion(out, data.y)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))
            train_loss = float(loss) / int(len(train_loader))

            eval_data = next(iter(test_loader))  # Loads all test samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader))  # Loads all eval samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            val_acc = evaluate(out, eval_data.y)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper+"_extend", _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break

    model = load_best_model(best_epoch, _paper+"_extend", _dataset, model, args.eval_enabled)

    with torch.no_grad():
        train_sum = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))

        eval_data = next(iter(test_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        val_acc = evaluate(out, eval_data.y)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper+"_extend", _dataset, model, train_acc, val_acc, test_acc)

def train_graph_contrastive(_dataset, _paper, args,pre_train_path= None):
    """
    Train a explainer to explain graph classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)

    # explanation_labels, indices = load_dataset_ground_truth(_dataset)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    explanation_labels, indices = load_dataset_ground_truth(_dataset)
    gt_val_set = create_data_list_extend(indices,graphs, features, labels, explanation_labels,True)
    gt_val_loader = DataLoader(gt_val_set, batch_size=len(gt_val_set), shuffle=False)

    if _dataset == 'ba2':
        _dataset = "ba2"

    model = model_selector(_paper, _dataset, True)
    # path = './ExplanationEvaluation/models/pretrained/GNN/ba2/best_model'
    # path = './checkpoints/contrastive_pretrain/ba2motifs/ba2motifs_100_reg_CE_loop_inter_3_spectral_pooling.pth'
    # path = './checkpoints/contrastive_pretrain/ba2motifs/ba2motifs_80_inter_explain1t_sp2_4_70_maxsub.pth'
    # path = './checkpoints/contrastive_pretrain/mutag/mutag_400_reg_ce_loop_inter_explain_posneg.pth'
    # path = './checkpoints/GNN/mutag/best_model'
    # path = './checkpoints/contrastive_pretrain/ba2motifs/ba2motifs_95_reg_CE_loop_inter_3_onlymax.pth'
    #path = './checkpoints/contrastive_pretrain/mutag/mutag_30_inter_explain1t_sp2_4_100_maxsub.pth'
    # path = './checkpoints/GNN/ba2/best_model'
    #pdict = torch.load(path)
    # #
    # new_dict = {}
    # for key in dict['model_state_dict'].keys():
    #     new_key = key
    #     if 'weight' in key and 'conv' in key:
    #         str_list = key.split('.')
    #         new_key = str_list[0]+'.lin.'+str_list[1]
    #         new_dict[new_key] = dict['model_state_dict'][key].T
    #     else:
    #         new_dict[new_key] = dict['model_state_dict'][key]
    #
    # model.load_state_dict(new_dict)

    # model.load_state_dict(dict['model_state_dict'])
    #model.load_state_dict(pdict)
    model.cuda()

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()



    best_val_acc = 0.0
    best_epoch = 0
    model.eval()

    epoch = 0
    # Evaluate train
    with torch.no_grad():
        train_sum = 0
        loss = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch, embedding=False)
            loss += criterion(out, data.y)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))
        train_loss = float(loss) / int(len(train_loader))

        eval_data = next(iter(test_loader))  # Loads all test samples
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch, embedding=False)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader))  # Loads all eval samples
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch, embedding=False)
        val_acc = evaluate(out, eval_data.y)

    print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

    if val_acc > best_val_acc:  # New best results
        print("Val improved")
        best_val_acc = val_acc
        best_epoch = epoch
        store_checkpoint(_paper + "_contrastive", _dataset, model, train_acc, val_acc, test_acc, best_epoch)


    # for epoch in range(0, args.epochs):
    for epoch in range(0, 0):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch,embedding= False)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            optimizer.step()

        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch,embedding= False)
                loss += criterion(out, data.y)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))
            train_loss = float(loss) / int(len(train_loader))

            eval_data = next(iter(test_loader))  # Loads all test samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch,embedding= False)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader))  # Loads all eval samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch,embedding= False)
            val_acc = evaluate(out, eval_data.y)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper+"_contrastive", _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break

    model = load_best_model(best_epoch, _paper+"_contrastive", _dataset, model, args.eval_enabled)

    # eval the gt
    with torch.no_grad():
        gt_val_data = next(iter(gt_val_loader))
        out = model(gt_val_data.x, gt_val_data.edge_index, gt_val_data.batch,embedding= False)
        out_temp = torch.softmax(out,dim=1)
        y_temp = gt_val_data.y
        out_mean = out_temp.mean(dim=0)
        indx0 = torch.argwhere(y_temp==0)
        out0_mean = out_temp[indx0,:].mean(dim=0)
        indx1 = torch.argwhere(y_temp==1)
        out1_mean = out_temp[indx1,:].mean(dim=0)
        print(out_mean.cpu().detach().numpy(),out0_mean,out1_mean)
        gt_val_acc = evaluate(out, gt_val_data.y)

    print(f"final  gt_val acc:{gt_val_acc}")

    with torch.no_grad():
        train_sum = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch,embedding= False)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))

        eval_data = next(iter(test_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch,embedding= False)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch,embedding= False)
        val_acc = evaluate(out, eval_data.y)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper+"_contrastive", _dataset, model, train_acc, val_acc, test_acc)

import random
import time
import json
import os

import torch
import numpy as np
from tqdm import tqdm

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.EfficiencyEvaluation import EfficiencyEvluation
from ExplanationEvaluation.explainers.GNNExplainer import GNNExplainer
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.models.model_selector_xgnn import model_selector
from ExplanationEvaluation.utils.plotting import plot


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_classification_task(graphs):
    """
    Given the original data, determines if the task as hand is a node or graph classification task
    :return: str either 'graph' or 'node'
    """
    if isinstance(graphs, list): # We're working with a model for graph classification
        return "graph"
    else:
        return "node"

def to_torch_graph(graphs, task):
    """
    Transforms the numpy graphs to torch tensors depending on the task of the model that we want to explain
    :param graphs: list of single numpy graph
    :param task: either 'node' or 'graph'
    :return: torch tensor
    """
    if task == 'graph':
        return [torch.tensor(g).cuda() for g in graphs]
    else:
        return torch.tensor(graphs).cuda()

def select_explainer(explainer, model, graphs, features, task, epochs, lr, reg_coefs, temp=None, sample_bias=None,model_eval=None):
    """
    Select the explainer we which to use.
    :param explainer: str, "PG" or "GNN"
    :param model: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :params sample_bias: the bias we add when sampling random graphs. 
    """
    if explainer == "PG":
        return PGExplainer(model, graphs, features, task, epochs=epochs, lr=lr, reg_coefs=reg_coefs,
                           temp=temp, sample_bias=sample_bias,model_eval = model_eval)
    elif explainer == "GNN":
        return GNNExplainer(model, graphs, features, task, epochs=epochs, lr=lr, reg_coefs=reg_coefs,model_eval = model_eval)
    else:
        raise NotImplementedError("Unknown explainer type")

def run_experiment(inference_eval, auc_eval, explainer, indices, labels):
    """
    Runs an experiment.
    We generate explanations for given indices and calculate the AUC score.
    :param inference_eval: object for measure the inference speed
    :param auc_eval: a metric object, which calculate the AUC score
    :param explainer: the explainer we wish to obtain predictions from
    :param indices: indices over which to evaluate the auc
    :returns: AUC score, inference speed
    """
    inference_eval.start_prepate()
    explainer.prepare(indices)

    inference_eval.start_explaining()
    explanations = []
    fid_plus = []
    fid_minus = []
    fid_plus_label = []
    fid_minus_label = []
    delta_fid = []
    delta_fid_label = []
    for idx in tqdm(indices):
        # if idx == 695:
        #     print("")s
        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]
        f_p, f_m, f_p_label, f_m_label = explainer.cal_fid(idx,graph,expl,label_idx)
        fid_plus.append(f_p)
        fid_minus.append(f_m)
        fid_plus_label.append(f_p_label)
        fid_minus_label.append(f_m_label)
        delta_fid.append(f_p-f_m)
        delta_fid_label.append(f_p_label - f_m_label)

        explanations.append((graph, expl))
    inference_eval.done_explaining()

    auc_score = auc_eval.get_score(explanations)
    time_score = inference_eval.get_score(explanations)

    fid_plus_mean = torch.stack(fid_plus).mean()
    fid_minus_mean = torch.stack(fid_minus).mean()
    fid_plus_label_mean = np.array(fid_plus_label).mean()
    fid_minus_label_mean = np.array(fid_minus_label).mean()
    delta_fid_mean = np.array(delta_fid).mean()
    delta_fid_label_mean = np.array(delta_fid_label).mean()
    print("fid_plus_mean", fid_plus_mean)
    print("fid_minus_mean", fid_minus_mean)
    print("fid_plus_label_mean", fid_plus_label_mean)
    print("fid_minus_label_mean", fid_minus_label_mean)
    print("delta_fid_mean", delta_fid_mean)
    print("delta_fid_label_mean", delta_fid_label_mean)

    # return auc_score, time_score
    return auc_score, time_score, fid_plus_mean, fid_minus_mean, fid_plus_label_mean,  fid_minus_label_mean


def cal_gt_ori_fids(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels,
                      new = False,save=False,reverse=False,data_name='xx'):
    """
    Runs an experiment.
    We generate explanations for given indices and calculate the AUC score.
    :param inference_eval: object for measure the inference speed
    :param auc_eval: a metric object, which calculate the AUC score
    :param explainer: the explainer we wish to obtain predictions from
    :param indices: indices over which to evaluate the auc
    :returns: AUC score, inference speed
    """
    inference_eval.start_prepate()
    explainer.prepare(indices,train=False)

    inference_eval.start_explaining()
    explanations = []
    fid_plus = []  # the last column is label
    fid_minus = []
    fid_plus_label = []
    fid_minus_label = []
    delta_fid = []
    delta_fid_label = []

    sparsity_list = []

    for idx in tqdm(indices):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]


        f_p_list = []
        f_m_list = []
        f_p_label_list = []
        f_m_label_list = []
        sp_list = []
        delta_f = []
        delta_f_label = []

        if new:
            f_p, f_m, f_p_label, f_m_label, sp = explainer.cal_fid_gt_new(idx, graph, explanation_labels, label_idx,reverse=reverse)
        else:
            f_p, f_m, f_p_label, f_m_label, sp = explainer.cal_fid_gt(idx, graph, explanation_labels, label_idx)

        f_p_list.append(f_p.item())
        f_m_list.append(f_m.item())
        f_p_label_list.append(f_p_label)
        f_m_label_list.append(f_m_label)
        delta_f.append(f_p.item() - f_m.item())
        delta_f_label.append(f_p_label - f_m_label)
        sp_list.append(sp.item())

        fid_plus.append(f_p_list)
        fid_minus.append(f_m_list)
        fid_plus_label.append(f_p_label_list)
        fid_minus_label.append(f_m_label_list)
        sparsity_list.append(sp_list)
        explanations.append((graph, expl))

        delta_fid.append(delta_f)
        delta_fid_label.append(delta_f_label)

        # explanations_alone.append(expl)
    inference_eval.done_explaining()

    # auc_score = auc_eval.get_score(explanations)
    time_score = inference_eval.get_score(explanations)

    fid_plus_mean = np.mean(np.array(fid_plus),axis=0)
    fid_minus_mean = np.mean(np.array(fid_minus),axis=0)
    fid_plus_label_mean = np.mean(np.array(fid_plus_label),axis=0)
    fid_minus_label_mean = np.mean(np.array(fid_minus_label),axis=0)
    sparsity_mean  = np.mean(np.array(sparsity_list),axis=0)
    delta_fid_mean = np.mean(np.array(delta_fid),axis=0) # np.array(delta_fid).mean(axis=0)
    delta_fid_label_mean = np.mean(np.array(delta_fid_label),axis=0) # np.array(delta_fid_label).mean(axis=0)
    # reverse = True fid-
    print("fid_plus_mean", fid_plus_mean)
    print("fid_minus_mean", fid_minus_mean)
    print("fid_plus_label_mean", fid_plus_label_mean)
    print("fid_minus_label_mean", fid_minus_label_mean)
    print("sparsity_mean", sparsity_mean)
    print("delta_fid_mean", delta_fid_mean)
    print("delta_fid_label_mean", delta_fid_label_mean)

    return 1.0, time_score, fid_plus_mean, fid_minus_mean, fid_plus_label_mean,  fid_minus_label_mean,\
        sparsity_mean, explanations

def cal_weight_ori_fids(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels, weight_list,count,
                      new = False,save=False,reverse=False,data_name='xx'):
    """
    Runs an experiment.
    We generate explanations for given indices and calculate the AUC score.
    :param inference_eval: object for measure the inference speed
    :param auc_eval: a metric object, which calculate the AUC score
    :param explainer: the explainer we wish to obtain predictions from
    :param indices: indices over which to evaluate the auc
    :returns: AUC score, inference speed
    """
    inference_eval.start_prepate()
    explainer.prepare(indices,train=False)

    inference_eval.start_explaining()
    explanations = []
    fid_plus = []  # the last column is label
    fid_minus = []
    fid_plus_label = []
    fid_minus_label = []
    delta_fid = []
    delta_fid_label = []

    sparsity_list = []

    for idx in tqdm(indices):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]


        f_p_list = []
        f_m_list = []
        f_p_label_list = []
        f_m_label_list = []
        sp_list = []
        delta_f = []
        delta_f_label = []

        # if new:
        #     f_p, f_m, f_p_label, f_m_label, sp = explainer.cal_fid_gt_new(idx, graph, explanation_labels, label_idx,reverse=reverse)
        # else:
        if len(weight_list[idx])<count+1:
            if explainer.type == "node":
                f_p, f_m, f_p_label, f_m_label, sp,weights_n = explainer.cal_fid_edit_distance(idx, graph, explanation_labels,
                                                                                       np.ones_like(explanation_labels[1]),label_idx)
            else:
                f_p, f_m, f_p_label, f_m_label, sp, weights_n = explainer.cal_fid_edit_distance(idx, graph,
                                                            explanation_labels,np.ones_like(explanation_labels[1][idx]), label_idx)
        else:
            f_p, f_m, f_p_label, f_m_label, sp,weights_n = explainer.cal_fid_edit_distance(idx, graph, explanation_labels,
                                                                                       np.array(weight_list[idx][count]),label_idx)

        f_p_list.append(f_p.item())
        f_m_list.append(f_m.item())
        f_p_label_list.append(f_p_label)
        f_m_label_list.append(f_m_label)
        delta_f.append(f_p.item() - f_m.item())
        delta_f_label.append(f_p_label - f_m_label)
        sp_list.append(sp.item())

        fid_plus.append(f_p_list)
        fid_minus.append(f_m_list)
        fid_plus_label.append(f_p_label_list)
        fid_minus_label.append(f_m_label_list)
        sparsity_list.append(sp_list)
        explanations.append((graph, torch.from_numpy(weights_n)))

        delta_fid.append(delta_f)
        delta_fid_label.append(delta_f_label)

        # explanations_alone.append(expl)
    inference_eval.done_explaining()

    # auc_score = auc_eval.get_score(explanations)
    time_score = inference_eval.get_score(explanations)

    fid_plus_mean = np.mean(np.array(fid_plus),axis=0)
    fid_minus_mean = np.mean(np.array(fid_minus),axis=0)
    fid_plus_label_mean = np.mean(np.array(fid_plus_label),axis=0)
    fid_minus_label_mean = np.mean(np.array(fid_minus_label),axis=0)
    sparsity_mean  = np.mean(np.array(sparsity_list),axis=0)
    delta_fid_mean = np.mean(np.array(delta_fid),axis=0) # np.array(delta_fid).mean(axis=0)
    delta_fid_label_mean = np.mean(np.array(delta_fid_label),axis=0) # np.array(delta_fid_label).mean(axis=0)
    # reverse = True fid-
    if False:
        print("fid_plus_mean", fid_plus_mean)
        print("fid_minus_mean", fid_minus_mean)
        print("fid_plus_label_mean", fid_plus_label_mean)
        print("fid_minus_label_mean", fid_minus_label_mean)
        print("sparsity_mean", sparsity_mean)
        print("delta_fid_mean", delta_fid_mean)
        print("delta_fid_label_mean", delta_fid_label_mean)

    return 1.0, time_score, fid_plus_mean, fid_minus_mean, fid_plus_label_mean,  fid_minus_label_mean,\
        delta_fid_mean,delta_fid_label_mean,sparsity_mean, explanations

def run_edit_distance_deltafid_ratio(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels,edit_distance_weights,
                      edit_sample_count=0, k_plus=None,k_minus=None):
    explainer.prepare(indices, False)
    fid_minus = []
    fid_minus_label = []
    embedding_minus_distance_src_k = []

    fid_plus = []
    fid_plus_label = []
    embedding_plus_distance_src_k = []

    delta_fid_prob = []
    delta_fid_acc = []

    explanations = []

    for idx in tqdm(indices):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape) > 0:
            label_idx = torch.argwhere(labels[idx] > 0)[0]
        else:
            label_idx = labels[idx]

        if len(edit_distance_weights[idx])<edit_sample_count+1:
            if explainer.type == "node":
                ed_weight = np.ones_like(explanation_labels[1])
            else:
                ed_weight = np.ones_like(explanation_labels[1][idx])
        else:
            ed_weight = np.array(edit_distance_weights[idx][edit_sample_count])

        fid_plus_mean, fid_plus_label_mean, fid_plus_embedding_distance_list,expl_for_auc = explainer.edit_distance_gt_ratio_plus(idx,
                                                                                                                    graph,
                                                                                                                    explanation_labels,
                                                                                                                    label_idx,ed_weight,
                                                                                                                    k=k_plus)

        fid_minus_mean, fid_minus_label_mean, fid_minus_embedding_distance_list, _ = explainer.edit_distance_gt_ratio_minus(
                idx,
                graph,
                explanation_labels,
                label_idx, ed_weight,
                k=k_minus)
        fid_minus.append(fid_minus_mean)
        fid_minus_label.append(fid_minus_label_mean)
        embedding_minus_distance_src_k.append(fid_minus_embedding_distance_list)

        fid_plus.append(fid_plus_mean)
        fid_plus_label.append(fid_plus_label_mean)
        embedding_plus_distance_src_k.append(fid_plus_embedding_distance_list)

        delta_fid_prob.append(fid_plus_mean - fid_minus_mean)
        delta_fid_acc.append(fid_plus_label_mean - fid_minus_label_mean)

        explanations.append((graph, torch.from_numpy(expl_for_auc)))

    fid_minus_mean = np.mean(np.array(fid_minus), axis=0)
    fid_minus_label_mean = np.mean(np.array(fid_minus_label), axis=0)
    distance_mean_minus = np.mean(np.array(embedding_minus_distance_src_k), axis=0)

    fid_plus_mean = np.mean(np.array(fid_plus), axis=0)
    fid_plus_label_mean = np.mean(np.array(fid_plus_label), axis=0)
    distance_mean_plus = np.mean(np.array(embedding_plus_distance_src_k), axis=0)

    delta_fid_prob = np.mean(np.array(delta_fid_prob), axis=0)
    delta_fid_acc = np.mean(np.array(delta_fid_acc), axis=0)

    iou_score,auc_score = auc_eval.get_auc_IOU_score(explanations)


    # reverse = True fid-
    # print()
    if False:
        print("k==", k_plus,k_minus)
        print("fid_plus_mean ", fid_plus_mean)
        print("fid_plus_label_mean ", fid_plus_label_mean)
        print("distance_mean_plus ", distance_mean_plus)
        print("fid_minus_mean ", fid_minus_mean)
        print("fid_minus_label_mean ", fid_minus_label_mean)
        print("distance_mean_minus ", distance_mean_minus)
        print("delta_fid_prob ", delta_fid_prob)
        print("delta_fid_acc ", delta_fid_acc)
        print("auc score ", auc_score)

    return fid_plus_mean, fid_plus_label_mean, distance_mean_plus, \
        fid_minus_mean, fid_minus_label_mean, distance_mean_minus, \
        delta_fid_prob, delta_fid_acc,auc_score,iou_score

def run_qualitative_experiment(explainer, indices, labels, config, explanation_labels):
    """
    Plot the explaination generated by the explainer
    :param explainer: the explainer object
    :param indices: indices on which we validate
    :param labels: predictions of the explainer
    :param config: dict holding which subgraphs to plot
    :param explanation_labels: the ground truth labels 
    """
    for idx in indices:
        graph, expl = explainer.explain(idx)
        plot(graph, expl, labels, idx, config.thres_min, config.thres_snip, config.dataset, config, explanation_labels)

def store_results(auc, auc_std, inf_time, checkpoint, config):
    """
    Save the replication results into a json file
    :param auc: the obtained AUC score
    :param auc_std: the obtained AUC standard deviation
    :param inf_time: time it takes to make a single prediction
    :param checkpoint: the checkpoint of the explained model
    :param config: dict config
    """
    results = {"AUC": auc,
               "AUC std": auc_std,
               "Inference time (ms)": inf_time}

    model_res = {"Training Accuracy": checkpoint["train_acc"],
                 "Validation Accuracy": checkpoint["val_acc"],
                 "Test Accuracy": checkpoint["test_acc"], }

    explainer_params = {"Explainer": config.explainer,
                        "Model": config.model,
                        "Dataset": config.dataset}

    json_dict = {"Explainer parameters": explainer_params,
                 "Results": results,
                 "Trained model stats": model_res}

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"./results/P_{config.explainer}_M_{config.model}_D_{config.dataset}_results.json", "w") as fp:
        json.dump(json_dict, fp, indent=4)

def replication(config, extension=False, run_qual=True, results_store=True):
    """
    Perform the replication study.
    First load a pre-trained model.
    Then we train our expainer.
    Followed by obtaining the generated explanations.
    And saving the obtained AUC score in a json file.
    :param config: a dict containing the config file values
    :param extension: bool, wheter to use all indices 
    """
    # Load complete dataset
    graphs, features, labels, _, _, test_mask = load_dataset(config.dataset)
    task = get_classification_task(graphs)

    features = torch.tensor(features).cuda()
    labels = torch.tensor(labels).cuda()
    graphs = to_torch_graph(graphs, task)

    print(config.dataset,config.model,config.explainer)
    print(config.lr,config.epochs,config.sample_bias,config.reg_size,config.reg_ent,config.temps)

    # Load pretrained models
    model, checkpoint = model_selector(config.model,
                                        config.dataset,
                                        pretrained=True,
                                        return_checkpoint=True)
    model.cuda()
    if config.eval_enabled:
        model.eval()

    # Get ground_truth for every node
    explanation_labels, indices = load_dataset_ground_truth(config.dataset)
    if extension: indices = np.argwhere(test_mask).squeeze()

    # Get explainer
    explainer = select_explainer(config.explainer,
                                 model=model,
                                 graphs=graphs,
                                 features=features,
                                 task=task,
                                 epochs=config.epochs,
                                 lr=config.lr,
                                 reg_coefs=[config.reg_size,
                                            config.reg_ent],
                                 temp=config.temps,
                                 sample_bias=config.sample_bias)

    # Get evaluation methods
    auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
    inference_eval = EfficiencyEvluation()

    # Perform the evaluation 10 times
    auc_scores = []
    times = []

    fid_plus = []
    fid_minus = []
    fid_plus_label = []
    fid_minus_label = []


    for idx, s in enumerate(config.seeds):
        print(f"Run {idx} with seed {s}")
        # Set all seeds needed for reproducibility
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        inference_eval.reset()
        # auc_score, time_score = run_experiment(inference_eval, auc_evaluation, explainer, indices)
        auc_score, time_score, \
        f_p, f_m, f_p_label, f_m_label = run_experiment(inference_eval, auc_evaluation, explainer, indices,labels)

        if idx == 0 and run_qual: # We only run the qualitative experiment once
            run_qualitative_experiment(explainer, indices, labels, config, explanation_labels)

        auc_scores.append(auc_score)
        print("score:",auc_score)
        times.append(time_score)
        print("time_elased:",time_score)
        fid_plus.append(f_p)
        fid_minus.append(f_m)
        fid_plus_label.append(f_p_label)
        fid_minus_label.append(f_m_label)

    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    inf_time = np.mean(times) / 10

    fid_plus_mean = torch.stack(fid_plus).mean()
    fid_minus_mean = torch.stack(fid_minus).mean()
    fid_plus_label_mean = np.array(fid_plus_label).mean()
    fid_minus_label_mean = np.array(fid_minus_label).mean()
    fid_plus_std = torch.stack(fid_plus).std()
    fid_minus_std = torch.stack(fid_minus).std()
    fid_plus_label_std = np.array(fid_plus_label).std()
    fid_minus_label_std = np.array(fid_minus_label).std()

    print()
    print("fid plus ",fid_plus_mean,fid_plus_std)
    print("fid minus ",fid_minus_mean,fid_minus_std)
    print("fid label plus ",fid_plus_label_mean,fid_plus_label_std)
    print("fid label minus ",fid_minus_label_mean,fid_minus_label_std)
    if results_store: store_results(auc, auc_std, inf_time, checkpoint, config)
        
    return (auc, auc_std), inf_time

def experiments_editdistance_ori_fids(config, extension=False,seeds_num = 1):

    # Load complete dataset
    graphs, features, labels, _, _, test_mask = load_dataset(config.dataset)
    task = get_classification_task(graphs)

    features = torch.tensor(features).cuda()
    labels = torch.tensor(labels).cuda()
    graphs = to_torch_graph(graphs, task)

    print(config.dataset, config.model, config.explainer)
    print(config.lr, config.epochs, config.sample_bias, config.reg_size, config.reg_ent, config.temps)

    # Load pretrained models
    model, checkpoint = model_selector(config.model,
                                       config.dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    model_extend = model
    model.cuda()
    model_extend.cuda()
    if config.eval_enabled:
        model.eval()
        model_extend.eval()

    # Get ground_truth for every node
    explanation_labels, indices = load_dataset_ground_truth(config.dataset)
    if extension: indices = np.argwhere(test_mask).squeeze()

    if task=='graph':
        path = './data/%s_random_sample_maps_ratio.npy'%config.dataset
    else:
        path = './data/%s_random_sample_maps_undirected_ratio.npy' % config.dataset

    edit_distance_map = np.load(path, allow_pickle=True).item()
    dict_for_save = {}

    for key in edit_distance_map.keys():

        edit_distance_lists = edit_distance_map[key]
        sample_t = []
        for i in indices:
            listx = edit_distance_lists[i]
            if len(listx) > 0:
                sample_t.append(len(listx))
        min_sample = min(sample_t)
        max_sample = max(sample_t)

        # Get a explainer, it doesn't matter which explainer
        explainer = select_explainer(config.explainer,
                                     model=model,
                                     graphs=graphs,
                                     features=features,
                                     task=task,
                                     epochs=config.epochs,
                                     lr=config.lr,
                                     reg_coefs=[config.reg_size,
                                                config.reg_ent],
                                     temp=config.temps,
                                     sample_bias=config.sample_bias,
                                     model_eval = model_extend)

        auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
        inference_eval = EfficiencyEvluation()

        fid_plus_mean_list_std = []
        fid_plus_mean_label_list_std = []
        fid_minus_mean_list_std = []
        fid_minus_mean_label_list_std = []
        delta_fid_prob_list_std = []
        delta_fid_acc_list_std = []
        auc_score_list_std = []

        time_consuming = []
        for seed in range(seeds_num):
            set_seed(seed)
            for i in indices:
                # listx = edit_distance_lists[i]
                random.shuffle(edit_distance_lists[i])

            if True:
                edit_distance_weight = edit_distance_lists  
                fid_plus_mean_list = []
                fid_plus_mean_label_list = []
                fid_minus_mean_list = []
                fid_minus_mean_label_list = []
                delta_fid_prob_mean_list = []
                delta_fid_acc_mean_list = []
                auc_score_list = []
                print("================================================",seed)

                start_time = time.time()
                for sample_count in range(min(min_sample, 10)):
                    auc_score, time_score, \
                    f_p, f_m, f_p_label, f_m_label,\
                        f_delta,f_delta_acc, sp, explanations = cal_weight_ori_fids(inference_eval, auc_evaluation,
                                                                 explainer, indices, labels,explanation_labels,
                                                                 edit_distance_weight, sample_count,data_name=task)
                    fid_plus_mean_list.append(f_p)
                    fid_plus_mean_label_list.append(f_p_label)
                    fid_minus_mean_list.append(f_m)
                    fid_minus_mean_label_list.append(f_m_label)
                    delta_fid_prob_mean_list.append(f_delta)
                    delta_fid_acc_mean_list.append(f_delta_acc)
                    auc_score_list.append(auc_score)
                end_time = time.time()
                time_consuming.append(end_time-start_time)
                fid_plus_mean_list_std.append(np.array(fid_plus_mean_list).mean())
                fid_plus_mean_label_list_std.append(np.array(fid_plus_mean_label_list).mean())
                fid_minus_mean_list_std.append(np.array(fid_minus_mean_list).mean())
                fid_minus_mean_label_list_std.append(np.array(fid_minus_mean_label_list).mean())
                delta_fid_prob_list_std.append(np.array(delta_fid_prob_mean_list).mean())
                delta_fid_acc_list_std.append(np.array(delta_fid_acc_mean_list).mean())
                auc_score_list_std.append(np.array(auc_score_list).mean())
        dict_for_save[key] = [np.array(fid_plus_mean_list_std).mean(), np.array(fid_plus_mean_list_std).std(),
                              np.array(fid_plus_mean_label_list_std).mean(),np.array(fid_plus_mean_label_list_std).std(),
                              np.array(fid_minus_mean_list_std).mean(), np.array(fid_minus_mean_list_std).std(),
                              np.array(fid_minus_mean_label_list_std).mean(),np.array(fid_minus_mean_label_list_std).std(),
                              np.array(delta_fid_prob_list_std).mean(), np.array(delta_fid_prob_list_std).std(),
                              np.array(delta_fid_acc_list_std).mean(), np.array(delta_fid_acc_list_std).std(),
                              ]

        print("#####################################################################")
        print("ed_k (remove,add ) ori fid , ", key)
        print("time      ", np.array(time_consuming).mean(),np.array(time_consuming).std())
        print("fid_plus_mean_list ", np.array(fid_plus_mean_list_std).mean(), np.array(fid_plus_mean_list_std).std())
        print("fid_plus_mean_label_list ", np.array(fid_plus_mean_label_list_std).mean(),
              np.array(fid_plus_mean_label_list_std).std())
        print("fid_minus_mean_list ", np.array(fid_minus_mean_list_std).mean(), np.array(fid_minus_mean_list_std).std())
        print("fid_minus_mean_label_list ", np.array(fid_minus_mean_label_list_std).mean(),
              np.array(fid_minus_mean_label_list_std).std())
        print("delta_fid_prob_list ", np.array(delta_fid_prob_list_std).mean(),np.array(delta_fid_prob_list_std).std())
        print("delta_fid_acc_list ", np.array(delta_fid_acc_list_std).mean(),np.array(delta_fid_acc_list_std).std())
    np.save('./data/%s_%s_results_ori_fid_%d_k.npy'%(config.model,config.dataset,seeds_num),dict_for_save)
    return (1.0, 0.0), time_score

def experiment_new_fid_ratio_editdistance(config, k_p=0,k_m=0,seeds_num=1):
    
    graphs, features, labels, _, _, test_mask = load_dataset(config.dataset)
    task = get_classification_task(graphs)
    if task == "node":
        direction = True
    else:
        direction = False

    features = torch.tensor(features).cuda()
    labels = torch.tensor(labels).cuda()
    graphs = to_torch_graph(graphs, task)

    print(config.dataset, config.model, config.explainer)
    print(config.lr, config.epochs, config.sample_bias, config.reg_size, config.reg_ent, config.temps)

    # Load pretrained models
    model, checkpoint = model_selector(config.model,
                                       config.dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    model.cuda()
    if config.eval_enabled:
        model.eval()

    # Get ground_truth for every node
    explanation_labels, indices = load_dataset_ground_truth(config.dataset)

    if task=='graph':
        path = './data/%s_random_sample_maps_ratio.npy'%config.dataset
    else:
        path = './data/%s_random_sample_maps_undirected_ratio.npy' % config.dataset


    edit_distance_map = np.load(path, allow_pickle=True).item()

    # Get explainer
    explainer = select_explainer(config.explainer,
                                 model=model,
                                 graphs=graphs,
                                 features=features,
                                 task=task,
                                 epochs=config.epochs,
                                 lr=config.lr,
                                 reg_coefs=[config.reg_size,
                                            config.reg_ent],
                                 temp=config.temps,
                                 sample_bias=config.sample_bias,
                                 model_eval=model)
    explainer.set_undirect(not direction)
    

    auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
    
    inference_eval = EfficiencyEvluation()

    select_keys = ['0.0','0.1','0.3','0.5','0.7','0.9']  

    dict_for_save = {}
    for key in edit_distance_map.keys():
        if key ==('0.0','0.0'): 
            pass
        else:
            continue
        edit_distance_lists = edit_distance_map[key]
        sample_t = []
        for i in indices:
            listx = edit_distance_lists[i]
            if len(listx)>0 :
                sample_t.append(len(listx))
        min_sample = min(sample_t)
        max_sample = max(sample_t)

        if k_p==0:
            k_p = 1
        if k_m==0:
            k_m = 1

        fid_plus_mean_list_std = []
        fid_plus_mean_label_list_std = []
        distance_mean_plus_mean_list_std = []
        fid_minus_mean_list_std = []
        fid_minus_mean_label_list_std = []
        distance_minus_plus_mean_list_std = []
        delta_fid_prob_list_std = []
        delta_fid_acc_list_std = []
        auc_score_list_std = []
        iou_score_list_std = []

        time_consuming = []
        for seed in range(seeds_num):
            set_seed(seed)
            for i in indices:
                random.shuffle(edit_distance_lists[i])

            if True:
                edit_distance_weight = edit_distance_lists
                fid_plus_mean_list = []
                fid_plus_mean_label_list = []
                distance_mean_plus_mean_list = []
                fid_minus_mean_list = []
                fid_minus_mean_label_list = []
                distance_minus_plus_mean_list = []

                delta_fid_prob_list = []
                delta_fid_acc_list = []

                auc_score_list = []
                iou_score_list = []
                # print()
                print("================================================", seed)
                # print("ed_k (remove,add ) ", key, seed)
                # print("sample count ", min(min_sample, 20))
                start_time = time.time()
                for sample_count in range(min(min_sample, 10)):

                    fid_plus_mean, fid_plus_label_mean, distance_mean_plus, \
                        fid_minus_mean, fid_minus_label_mean, distance_mean_minus, \
                        delta_fid_prob, delta_fid_acc, auc_score,iou_score = run_edit_distance_deltafid_ratio(inference_eval,
                                                 auc_evaluation,
                                                 explainer,
                                                 indices,
                                                 labels,
                                                 explanation_labels,
                                                 edit_distance_weights=edit_distance_weight,
                                                 edit_sample_count=sample_count,k_plus=k_p,k_minus=k_m)
                    fid_plus_mean_list.append(fid_plus_mean)
                    fid_plus_mean_label_list.append(fid_plus_label_mean)
                    distance_mean_plus_mean_list.append(distance_mean_plus)

                    fid_minus_mean_list.append(fid_minus_mean)
                    fid_minus_mean_label_list.append(fid_minus_label_mean)
                    distance_minus_plus_mean_list.append(distance_mean_minus)

                    delta_fid_prob_list.append(delta_fid_prob)
                    delta_fid_acc_list.append(delta_fid_acc)

                    auc_score_list.append(auc_score)
                    iou_score_list.append(iou_score)
                end_time = time.time()
                time_consuming.append(end_time-start_time)
                fid_plus_mean_list_std.append(np.array(fid_plus_mean_list).mean())
                fid_plus_mean_label_list_std.append(np.array(fid_plus_mean_label_list).mean())
                distance_mean_plus_mean_list_std.append(np.array(distance_mean_plus_mean_list).mean())
                fid_minus_mean_list_std.append(np.array(fid_minus_mean_list).mean())
                fid_minus_mean_label_list_std.append(np.array(fid_minus_mean_label_list).mean())
                distance_minus_plus_mean_list_std.append(np.array(distance_minus_plus_mean_list).mean())
                delta_fid_prob_list_std.append(np.array(delta_fid_prob_list).mean())
                delta_fid_acc_list_std.append(np.array(delta_fid_acc_list).mean())
                auc_score_list_std.append(np.array(auc_score_list).mean())
                iou_score_list_std.append(np.array(iou_score_list).mean())

        dict_for_save[key] = [np.array(fid_plus_mean_list_std).mean(), np.array(fid_plus_mean_list_std).std(),
                              np.array(fid_plus_mean_label_list_std).mean(),np.array(fid_plus_mean_label_list_std).std(),
                              np.array(fid_minus_mean_list_std).mean(), np.array(fid_minus_mean_list_std).std(),
                              np.array(fid_minus_mean_label_list_std).mean(),np.array(fid_minus_mean_label_list_std).std(),
                              np.array(delta_fid_prob_list_std).mean(), np.array(delta_fid_prob_list_std).std(),
                              np.array(delta_fid_acc_list_std).mean(), np.array(delta_fid_acc_list_std).std(),
                              np.array(auc_score_list_std).mean(), np.array(auc_score_list_std).std(),
                              np.array(iou_score_list_std).mean(), np.array(iou_score_list_std).std(),
                              np.array(distance_mean_plus_mean_list_std).mean(),np.array(distance_mean_plus_mean_list_std).std(),
                              np.array(distance_minus_plus_mean_list_std).mean(),np.array(distance_minus_plus_mean_list_std).std()
                              ]
        print("#####################################################################")
        print("time ", np.array(time_consuming).mean(),np.array(time_consuming).std())
        print("ed_k (remove,add ), k_plus, k_minus, ", key,k_p, k_m )
        print("fid_plus_mean_list ", np.array(fid_plus_mean_list_std).mean(),np.array(fid_plus_mean_list_std).std())
        print("fid_plus_mean_label_list ", np.array(fid_plus_mean_label_list_std).mean(),np.array(fid_plus_mean_label_list_std).std())
        print("distance_mean_plus_mean_list ", np.array(distance_mean_plus_mean_list_std).mean(), np.array(distance_mean_plus_mean_list_std).std())
        print("fid_minus_mean_list ", np.array(fid_minus_mean_list_std).mean(), np.array(fid_minus_mean_list_std).std())
        print("fid_minus_mean_label_list ", np.array(fid_minus_mean_label_list_std).mean(), np.array(fid_minus_mean_label_list_std).std())
        print("distance_minus_plus_mean_list ", np.array(distance_minus_plus_mean_list_std).mean(), np.array(distance_minus_plus_mean_list_std).std())
        print("delta_fid_prob_list ", np.array(delta_fid_prob_list_std).mean(), np.array(delta_fid_prob_list_std).std())
        print("delta_fid_acc_list ", np.array(delta_fid_acc_list_std).mean(), np.array(delta_fid_acc_list_std).std())
        print("auc_score_list ", np.array(auc_score_list_std).mean(), np.array(auc_score_list_std).std())
        print("iou_score_list ", np.array(iou_score_list_std).mean(), np.array(iou_score_list_std).std())

    np.save('./data/%s_%s_results_new_fid_%.2f_%.2f_seeds_%d.npy'%(config.model,config.dataset,k_p,k_m,seeds_num),dict_for_save)

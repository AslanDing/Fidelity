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
from ExplanationEvaluation.models.model_selector_xgnn import model_selector,model_selector_extend,model_selector_contrastive
from ExplanationEvaluation.utils.plotting import plot


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

def run_experiment_sp(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels, sparsitys,
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

    embedding_src_list = []
    embedding_src_another_list = []

    embedding_G_E_list_plus = []
    embedding_G_E_list_minus = []
    # embedding_G_E_GT_list = []

    embedding_expl_minus_src_list = []
    embedding_expl_minus_another_list = []
    embedding_expl_plus_src_list = []
    embedding_expl_plus_another_list = []
    label_lists = []

    # explanations_alone = []

    for idx in tqdm(indices):
    # for idx in tqdm(range(labels.shape[0])):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]

        if save:
            # embedding_src,embedding_src_another,\
            #     embedding_expl_src_minus,embedding_expl_another_minus = explainer.getembedding_gt(idx, graph,
            #                                                 explanation_labels, label_idx,reverse=reverse)

            embedding_src, embedding_src_another, embedding_expl_src_minus, embedding_expl_another_minus, \
                embedding_expl_src_plus, embedding_expl_another_plus = explainer.getembedding_gt(idx, graph,
                                                            explanation_labels, label_idx,reverse=reverse)

            embedding_src,embedding_g_e_list_plus,embedding_g_e_list_minus = explainer.get_embedding_gt(idx, graph, explanation_labels,reverse=reverse)

            embedding_G_E_list_plus.append(embedding_g_e_list_plus[0].cpu().detach().numpy())
            embedding_G_E_list_minus.append(embedding_g_e_list_minus[0].cpu().detach().numpy())

            embedding_src_list.append(embedding_src.cpu().detach().numpy())
            embedding_src_another_list.append(embedding_src_another.cpu().detach().numpy())

            embedding_expl_minus_src_list.append(embedding_expl_src_minus.cpu().detach().numpy())
            embedding_expl_minus_another_list.append(embedding_expl_another_minus.cpu().detach().numpy())

            embedding_expl_plus_src_list.append(embedding_expl_src_plus.cpu().detach().numpy())
            embedding_expl_plus_another_list.append(embedding_expl_another_plus.cpu().detach().numpy())


            label_lists.append(label_idx.cpu().detach().numpy())
        # continue

        f_p_list = []
        f_m_list = []
        f_p_label_list = []
        f_m_label_list = []
        sp_list = []
        delta_f = []
        delta_f_label = []
        for sp in sparsitys:
            if new : # reverse = False ,fid+ , reverse= True fid-
                f_p, f_m, f_p_label, f_m_label, sp = explainer.cal_fid_sparsity_new(idx, graph, expl, label_idx, sp,reverse=reverse)
            else:
                f_p, f_m, f_p_label, f_m_label,sp = explainer.cal_fid_sparsity(idx,graph,expl,label_idx,sp)
            f_p_list.append(f_p.item())
            f_m_list.append(f_m.item())
            f_p_label_list.append(f_p_label)
            f_m_label_list.append(f_m_label)
            delta_f.append(f_p.item() - f_m.item())
            delta_f_label.append(f_p_label - f_m_label)
            sp_list.append(sp.item())
        # here beed a gt results
        if new:
            f_p, f_m, f_p_label, f_m_label, sp = explainer.cal_fid_gt_new(idx, graph, explanation_labels, label_idx,reverse=reverse)
        else:
            f_p, f_m, f_p_label, f_m_label, sp = explainer.cal_fid_gt(idx, graph, explanation_labels, label_idx)
        #print(f_p,f_m)
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

    if save:
        # save embeddings
        embedding_src_list = np.concatenate(embedding_src_list,axis=0)
        # embedding_src_another_list = np.concatenate(embedding_src_another_list,axis=0)
        embedding_expl_minus_src_list = np.concatenate(embedding_expl_minus_src_list,axis=0)
        # embedding_expl_minus_another_list = np.concatenate(embedding_expl_minus_another_list,axis=0)
        embedding_expl_plus_src_list = np.concatenate(embedding_expl_plus_src_list,axis=0)
        # embedding_expl_plus_another_list = np.concatenate(embedding_expl_plus_another_list,axis=0)

        embedding_G_E_list_plus = np.concatenate(embedding_G_E_list_plus,axis=0)
        embedding_G_E_list_minus = np.concatenate(embedding_G_E_list_minus,axis=0)
        label_lists = np.array(label_lists)
        np.save('./%s_embedding_src1'%data_name,embedding_src_list)
        # np.save('./embedding_expl_src1', embedding_expl_src_list)
        np.save('./%s_embedding_expl_minus_src_list'%data_name, embedding_expl_minus_src_list)
        # np.save('./embedding_expl_minus_another_list', embedding_expl_minus_another_list)
        np.save('./%s_embedding_expl_plus_src_list'%data_name, embedding_expl_plus_src_list)
        # np.save('./embedding_expl_plus_another_list', embedding_expl_plus_another_list)
        np.save('./%s_embedding_G_E_list_plus'%data_name, embedding_G_E_list_plus)
        np.save('./%s_embedding_G_E_list_minus'%data_name, embedding_G_E_list_minus)

        np.save('./%s_label_list'%data_name, label_lists)
        # exit(0)

    # explanations_alone = torch.stack(explanations_alone)
    # print(explanations_alone.max(),explanations_alone.min())
    auc_score = auc_eval.get_score(explanations)
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

    return auc_score, time_score, fid_plus_mean, fid_minus_mean, fid_plus_label_mean,  fid_minus_label_mean, sparsity_mean, explanations


def run_gt_plus_k(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels, sparsitys,
                      new = False,save=False,reverse=False,data_name='xx',k=1,save_name=''):
    explainer.prepare(indices,False)
    fid_plus = []
    fid_plus_label = []
    embedding_distance_src_k = []
    for idx in tqdm(indices):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]

        fid_plus_mean,fid_plus_label_mean,fid_plus_embedding_distance_list = explainer.getembedding_gt_k_plus(idx, graph, explanation_labels, label_idx,k=k)

        fid_plus.append(fid_plus_mean)
        fid_plus_label.append(fid_plus_label_mean)
        embedding_distance_src_k.append(fid_plus_embedding_distance_list)


    fid_plus_mean = np.mean(np.array(fid_plus),axis=0)
    fid_plus_label_mean = np.mean(np.array(fid_plus_label),axis=0)
    distance_mean  = np.mean(np.array(embedding_distance_src_k),axis=0)
    if save :
        path = r'./fid_save/%s'%save_name
        np.save(path+'_fid_plus',np.array(fid_plus))
        np.save(path+'_fid_plus_acc',np.array(fid_plus_label))
        np.save(path+'_fid_plus_dis',np.array(embedding_distance_src_k))
    print("k ",k)
    print("fid_plus_mean", fid_plus_mean)
    print("fid_plus_label_mean", fid_plus_label_mean)
    print("distance_mean", distance_mean)

    return fid_plus_mean, fid_plus_label_mean, distance_mean

def run_gt_minus_k(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels, sparsitys,
                      new = False,save=False,reverse=False,data_name='xx',k=1,save_name=''):
    explainer.prepare(indices,False)
    fid_minus = []
    fid_minus_label = []
    embedding_distance_src_k = []


    for idx in tqdm(indices):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]

        fid_minus_mean,fid_minus_label_mean,fid_minus_embedding_distance_list = explainer.getembedding_gt_k_minus(idx, graph, explanation_labels, label_idx,k=k)

        fid_minus.append(fid_minus_mean)
        fid_minus_label.append(fid_minus_label_mean)
        embedding_distance_src_k.append(fid_minus_embedding_distance_list)


    fid_minus_mean = np.mean(np.array(fid_minus),axis=0)
    fid_minus_label_mean = np.mean(np.array(fid_minus_label),axis=0)
    distance_mean  = np.mean(np.array(embedding_distance_src_k),axis=0)
    if save :
        path = r'./fid_save/%s'%save_name
        np.save(path+'_fid_minus',np.array(fid_minus))
        np.save(path+'_fid_minus_acc',np.array(fid_minus_label))
        np.save(path+'_fid_minus_dis',np.array(embedding_distance_src_k))
    # reverse = True fid-
    print("k ", k)
    print("fid_minus_mean", fid_minus_mean)
    print("fid_minus_label_mean", fid_minus_label_mean)
    print("distance_mean", distance_mean)

    return  fid_minus_mean,  fid_minus_label_mean, distance_mean


def run_gt_deltafid_k(inference_eval, auc_eval, explainer,
                      indices, labels, explanation_labels, sparsitys,
                      new = False,save=False,reverse=False,data_name='xx',k=1):
    explainer.prepare(indices,False)
    fid_minus = []
    fid_minus_label = []
    embedding_minus_distance_src_k = []
    
    fid_plus = []
    fid_plus_label = []
    embedding_plus_distance_src_k = []

    delta_fid_prob = []
    delta_fid_acc = []

    for idx in tqdm(indices):

        graph, expl = explainer.explain(idx)
        if len(labels[idx].shape)>0:
            label_idx = torch.argwhere(labels[idx]>0)[0]
        else:
            label_idx = labels[idx]
        fid_plus_mean,fid_plus_label_mean,fid_plus_embedding_distance_list = explainer.getembedding_gt_k_plus(idx, graph, explanation_labels, label_idx,k=k)

        fid_minus_mean,fid_minus_label_mean,fid_minus_embedding_distance_list = explainer.getembedding_gt_k_minus(idx, graph, explanation_labels, label_idx,k=k)

        fid_minus.append(fid_minus_mean)
        fid_minus_label.append(fid_minus_label_mean)
        embedding_minus_distance_src_k.append(fid_minus_embedding_distance_list)

        fid_plus.append(fid_plus_mean)
        fid_plus_label.append(fid_plus_label_mean)
        embedding_plus_distance_src_k.append(fid_plus_embedding_distance_list)

        delta_fid_prob.append(fid_plus_mean-fid_minus_mean)
        delta_fid_acc.append(fid_plus_label_mean - fid_minus_label_mean)

    fid_minus_mean = np.mean(np.array(fid_minus),axis=0)
    fid_minus_label_mean = np.mean(np.array(fid_minus_label),axis=0)
    distance_mean_minus  = np.mean(np.array(embedding_minus_distance_src_k),axis=0)

    fid_plus_mean = np.mean(np.array(fid_plus),axis=0)
    fid_plus_label_mean = np.mean(np.array(fid_plus_label),axis=0)
    distance_mean_plus  = np.mean(np.array(embedding_plus_distance_src_k),axis=0)

    delta_fid_prob = np.mean(np.array(delta_fid_prob),axis=0)
    delta_fid_acc = np.mean(np.array(delta_fid_acc),axis=0)

    # reverse = True fid-
    print()
    print("k==", k)
    print("fid_plus_mean ", fid_plus_mean)
    print("fid_plus_label_mean ", fid_plus_label_mean)
    print("distance_mean_plus ", distance_mean_plus)
    print("fid_minus_mean ", fid_minus_mean)
    print("fid_minus_label_mean ", fid_minus_label_mean)
    print("distance_mean_minus ", distance_mean_minus)
    print("delta_fid_prob ",delta_fid_prob)
    print("delta_fid_acc ",delta_fid_acc)

    return  fid_plus_mean,  fid_plus_label_mean, distance_mean_plus,\
        fid_minus_mean,fid_minus_label_mean,distance_mean_minus, \
            delta_fid_prob,delta_fid_acc



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

def replication_sp(config, extension=False, run_qual=True, results_store=True,
                         sparsitys= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                   extend = False):
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

    print(config.dataset, config.model, config.explainer)
    print(config.lr, config.epochs, config.sample_bias, config.reg_size, config.reg_ent, config.temps)

    # Load pretrained models
    model, checkpoint = model_selector(config.model,
                                       config.dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    if extend:
        model_extend, checkpoint_extend = model_selector_extend(config.model,
                                           config.dataset,
                                           pretrained=True,
                                           return_checkpoint=True)
    else:
        model_extend = model
    model.cuda()
    model_extend.cuda()
    if config.eval_enabled:
        model.eval()
        model_extend.eval()

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
                                 sample_bias=config.sample_bias,
                                 model_eval = model_extend)

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

    sp_list = []

    for idx, s in enumerate(config.seeds):
        print(f"Run {idx} with seed {s}")
        # Set all seeds needed for reproducibility
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        inference_eval.reset()
        # auc_score, time_score = run_experiment(inference_eval, auc_evaluation, explainer, indices)
        auc_score, time_score, \
            f_p, f_m, f_p_label, f_m_label,sp,explanations = run_experiment_sp(inference_eval, auc_evaluation,
                                            explainer, indices, labels,explanation_labels,sparsitys)

        # for iidx ,(graph,expl) in enumerate(explanations):
        #     plot(graph, expl, labels, iidx, config.thres_min, config.thres_snip, config.dataset, config, explanation_labels)
        # if idx == 0 and run_qual:  # We only run the qualitative experiment once
        #     run_qualitative_experiment(explainer, indices, labels, config, explanation_labels)

        auc_scores.append(auc_score)
        print("score:", auc_score)
        times.append(time_score)
        print("time_elased:", time_score)
        fid_plus.append(f_p)
        fid_minus.append(f_m)
        fid_plus_label.append(f_p_label)
        fid_minus_label.append(f_m_label)
        sp_list.append(sp)
    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    inf_time = np.mean(times) / 10

    fid_plus_mean = np.stack(fid_plus).mean(axis=0)
    fid_minus_mean = np.stack(fid_minus).mean(axis=0)
    fid_plus_label_mean = np.array(fid_plus_label).mean(axis=0)
    fid_minus_label_mean = np.array(fid_minus_label).mean(axis=0)
    sp_list_mean = np.array(sp_list).mean(axis=0)
    fid_plus_std = np.stack(fid_plus).std(axis=0)
    fid_minus_std = np.stack(fid_minus).std(axis=0)
    fid_plus_label_std = np.array(fid_plus_label).std(axis=0)
    fid_minus_label_std = np.array(fid_minus_label).std(axis=0)
    sp_list_std = np.array(sp_list).std(axis=0)
    print()
    print("fid plus ", fid_plus_mean, fid_plus_std)
    print("fid minus ", fid_minus_mean, fid_minus_std)
    print("fid label plus ", fid_plus_label_mean, fid_plus_label_std)
    print("fid label minus ", fid_minus_label_mean, fid_minus_label_std)
    print("sparse ", sp_list_mean, sp_list_std)
    if results_store: store_results(auc, auc_std, inf_time, checkpoint, config)

    return (auc, auc_std), inf_time

def replication_sp_contrastive(config, extension=False, run_qual=True, results_store=True,
                         sparsitys= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                   extend = False):
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

    if config.dataset == 'ba2':
        config.dataset = "ba2"

    print(config.dataset, config.model, config.explainer)
    print(config.lr, config.epochs, config.sample_bias, config.reg_size, config.reg_ent, config.temps)


    # Load pretrained models

    model, checkpoint = model_selector(config.model,
                                       config.dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    if extend:
        model_extend, checkpoint_extend = model_selector_contrastive(config.model,
                                           config.dataset,
                                           pretrained=True,
                                           return_checkpoint=True)
        # model = model_extend
    else:
        model_extend = model

    model.cuda()
    model_extend.cuda()
    if config.eval_enabled:
        model.eval()
        model_extend.eval()

    # Get ground_truth for every node
    if config.dataset == "ba2_1":
        config.dataset = "ba2"
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
                                 sample_bias=config.sample_bias,
                                 model_eval = model_extend)

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

    sp_list = []

    for idx, s in enumerate(config.seeds):
        print(f"Run {idx} with seed {s}")
        # Set all seeds needed for reproducibility
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        inference_eval.reset()
        # auc_score, time_score = run_experiment(inference_eval, auc_evaluation, explainer, indices)
        auc_score, time_score, \
            f_p, f_m, f_p_label, f_m_label,sp,explanations = run_experiment_sp(inference_eval, auc_evaluation,
                                            explainer, indices, labels,explanation_labels,sparsitys)

        # for iidx ,(graph,expl) in enumerate(explanations):
        #     plot(graph, expl, labels, iidx, config.thres_min, config.thres_snip, config.dataset, config, explanation_labels)
        # if idx == 0 and run_qual:  # We only run the qualitative experiment once
        #     run_qualitative_experiment(explainer, indices, labels, config, explanation_labels)

        auc_scores.append(auc_score)
        print("score:", auc_score)
        times.append(time_score)
        print("time_elased:", time_score)
        fid_plus.append(f_p)
        fid_minus.append(f_m)
        fid_plus_label.append(f_p_label)
        fid_minus_label.append(f_m_label)
        sp_list.append(sp)
        break
    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    inf_time = np.mean(times) / 10

    fid_plus_mean = np.stack(fid_plus).mean(axis=0)
    fid_minus_mean = np.stack(fid_minus).mean(axis=0)
    fid_plus_label_mean = np.array(fid_plus_label).mean(axis=0)
    fid_minus_label_mean = np.array(fid_minus_label).mean(axis=0)
    sp_list_mean = np.array(sp_list).mean(axis=0)
    fid_plus_std = np.stack(fid_plus).std(axis=0)
    fid_minus_std = np.stack(fid_minus).std(axis=0)
    fid_plus_label_std = np.array(fid_plus_label).std(axis=0)
    fid_minus_label_std = np.array(fid_minus_label).std(axis=0)
    sp_list_std = np.array(sp_list).std(axis=0)
    print()
    print("fid plus ", fid_plus_mean, fid_plus_std)
    print("fid minus ", fid_minus_mean, fid_minus_std)
    print("fid label plus ", fid_plus_label_mean, fid_plus_label_std)
    print("fid label minus ", fid_minus_label_mean, fid_minus_label_std)
    print("sparse ", sp_list_mean, sp_list_std)
    if results_store: store_results(auc, auc_std, inf_time, checkpoint, config)

    return (auc, auc_std), inf_time

def replication_sp_new(config, extension=False, run_qual=True, results_store=True,
                         sparsitys= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                   extend = False,reverse=True):
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

    print(config.dataset, config.model, config.explainer)
    print(config.lr, config.epochs, config.sample_bias, config.reg_size, config.reg_ent, config.temps)

    # Load pretrained models
    model, checkpoint = model_selector(config.model,
                                       config.dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    if extend:
        model_extend, checkpoint_extend = model_selector_contrastive(config.model,
                                           config.dataset,
                                           pretrained=True,
                                           return_checkpoint=True)
        model = model_extend
    else:
        model_extend = model


    model.cuda()
    model_extend.cuda()
    if config.eval_enabled:
        model.eval()
        model_extend.eval()

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
                                 sample_bias=config.sample_bias,
                                 model_eval=model_extend)

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

    sp_list = []

    for idx, s in enumerate(config.seeds):
        print(f"Run {idx} with seed {s}")
        # Set all seeds needed for reproducibility
        # s = 1
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        if idx < 1 and results_store:
            # reverse = True , fid-     reverse=False  fid+
            inference_eval.reset()
            auc_score, time_score, \
                f_p, f_m, f_p_label, f_m_label, sp, explanations = run_experiment_sp(inference_eval, auc_evaluation,
                                                                                     explainer, indices, labels,
                                                                                     explanation_labels, sparsitys,
                                                                                     new = True,save=True,reverse=reverse,
                                                                                     data_name=config.dataset)
        else:
            inference_eval.reset()
            auc_score, time_score, \
                f_p, f_m, f_p_label, f_m_label, sp, explanations = run_experiment_sp(inference_eval, auc_evaluation,
                                                                                     explainer, indices, labels,
                                                                                     explanation_labels, sparsitys,
                                                                                     new=True, save=False,reverse=reverse,
                                                                                     data_name=config.dataset)

        auc_scores.append(auc_score)
        print("score:", auc_score)
        times.append(time_score)
        print("time_elased:", time_score)
        fid_plus.append(f_p)
        fid_minus.append(f_m)
        fid_plus_label.append(f_p_label)
        fid_minus_label.append(f_m_label)
        sp_list.append(sp)
        break

    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    inf_time = np.mean(times) / 10

    fid_plus_mean = np.stack(fid_plus).mean(axis=0)
    fid_minus_mean = np.stack(fid_minus).mean(axis=0)
    fid_plus_label_mean = np.array(fid_plus_label).mean(axis=0)
    fid_minus_label_mean = np.array(fid_minus_label).mean(axis=0)
    sp_list_mean = np.array(sp_list).mean(axis=0)
    fid_plus_std = np.stack(fid_plus).std(axis=0)
    fid_minus_std = np.stack(fid_minus).std(axis=0)
    fid_plus_label_std = np.array(fid_plus_label).std(axis=0)
    fid_minus_label_std = np.array(fid_minus_label).std(axis=0)
    sp_list_std = np.array(sp_list).std(axis=0)
    print()
    print("fid plus ", fid_plus_mean, fid_plus_std)
    print("fid minus ", fid_minus_mean, fid_minus_std)
    print("fid label plus ", fid_plus_label_mean, fid_plus_label_std)
    print("fid label minus ", fid_minus_label_mean, fid_minus_label_std)
    print("sparse ", sp_list_mean, sp_list_std)
    if results_store: store_results(auc, auc_std, inf_time, checkpoint, config)

    return (auc, auc_std), inf_time

def replication_sp_new_k(config, extension=False, run_qual=True, results_store=True,
                         sparsitys= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                   extend = False,reverse=True):
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
    direction = False

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
    if extend:
        model_extend, checkpoint_extend = model_selector_contrastive(config.model,
                                           config.dataset,
                                           pretrained=True,
                                           return_checkpoint=True)
        model = model_extend
    else:
        model_extend = model


    model.cuda()
    model_extend.cuda()
    if config.eval_enabled:
        model.eval()
        model_extend.eval()

    # Get ground_truth for every node
    explanation_labels, indices = load_dataset_ground_truth(config.dataset)
    # find the k of the label
    explanation_edge_numbers=[]
    nonexplanation_edge_numbers = []
    for idx in indices:
        explanation_edge_numbers.append(np.sum(explanation_labels[1][idx]))
        nonexplanation_edge_numbers.append(explanation_labels[1][idx].shape[0]-np.sum(explanation_labels[1][idx]))
    max_explanation_length = max(explanation_edge_numbers)
    min_explanation_length = min(explanation_edge_numbers)
    non_max_explanation_length = max(nonexplanation_edge_numbers)
    non_min_explanation_length = min(nonexplanation_edge_numbers)
    if not direction:
        max_explanation_length = max_explanation_length // 2
        min_explanation_length = min_explanation_length // 2
        non_max_explanation_length = non_max_explanation_length // 2
        non_min_explanation_length = non_min_explanation_length // 2

    print("explanation ",min_explanation_length,max_explanation_length)
    print("nonexplanation ",non_min_explanation_length,non_max_explanation_length)


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
                                 sample_bias=config.sample_bias,
                                 model_eval=model_extend)
    explainer.set_undirect(not direction)
    # Get evaluation methods
    auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
    inference_eval = EfficiencyEvluation()

    # Perform the evaluation 10 times
    # auc_scores = []
    # times = []
    #
    # fid_plus = []
    # fid_minus = []
    # fid_plus_label = []
    # fid_minus_label = []
    #
    # sp_list = []
    #


    for k in range(0,min_explanation_length,1):
        run_gt_plus_k(inference_eval, auc_evaluation,
                     explainer, indices, labels,
                     explanation_labels, sparsitys,
                     new = True,save=True,reverse=reverse,
                     data_name=config.dataset,k=k+1,save_name="random_k_%d"%(k+1))
    k=0
    run_gt_minus_k(inference_eval, auc_evaluation,
                 explainer, indices, labels,
                 explanation_labels, sparsitys,
                 new = True,save=True,reverse=reverse,
                 data_name=config.dataset,k=k+1,save_name="random_k_%d"%(k+1))
    for k in range(4,non_min_explanation_length,5):
        run_gt_minus_k(inference_eval, auc_evaluation,
                     explainer, indices, labels,
                     explanation_labels, sparsitys,
                     new = True,save=True,reverse=reverse,
                     data_name=config.dataset,k=k+1,save_name="random_k_%d"%(k+1))

    # for k in range(min(min_explanation_length,non_min_explanation_length)):
    #     run_gt_deltafid_k(inference_eval, auc_evaluation,
    #                  explainer, indices, labels,
    #                  explanation_labels, sparsitys,
    #                  new = True,save=True,reverse=reverse,
    #                  data_name=config.dataset,k=k+1)


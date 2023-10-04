import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiment_new_fid_explainers,replication_sp


_dataset = 'treecycles'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
_explainer = 'pgexplainer' # One of: pgexplainer, gnnexplainer

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, extension

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"
print(config_path)
config = Selector(config_path)

# config.args.explainer = merge_parameter(config.args.explainer, nni_params)

extension = (_folder == 'extension')

# config.args.explainer.seeds = [0]

# this py is to calculate new fid+ fid- and fid\Delta of different edit distances
# key -> (remove, add)
explainers = ['gnnexplainer','pgexplainer','subgraphx','pgmexplainer']
for index in range(2):
    explainer_name = explainers[index]
    explainer_dir = './data/%s'%explainer_name
    dataset_name = config.args.explainer.dataset

    if explainer_name in ['gnnexplainer','pgexplainer']:
        path = explainer_dir+"/%s_weight_dict.npy"%dataset_name
        weights_dict = np.load(path,allow_pickle=True).item()
        for key in weights_dict.keys():
            weight = weights_dict[key]
            print(explainer_name,dataset_name,key,path)
            experiment_new_fid_explainers(config.args.explainer, explainer_weights=weight, k_p=2, k_m=5)
    else:
        # find all weight
        weight_list = os.listdir(explainer_dir)
        for weight_name in weight_list:
            if dataset_name in weight_name:
                weight = np.load(weight_name,allow_pickle=True)
                print(explainer_name,dataset_name,weight_name)
                experiment_new_fid_explainers(config.args.explainer, explainer_weights=weight, k_p=1, k_m=1)


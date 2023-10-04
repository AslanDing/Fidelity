import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiment_new_fid_ab_editdistance,replication_sp


_dataset = 'bacommunity'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
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
keys = [(2,2),(3,3),(1,0),(2,0),(3,0),(4,0)]
for key in keys:
    (auc, auc_std), inf_time = experiment_new_fid_ab_editdistance(config.args.explainer, key=(0,5),k_p=1,k_m=1)

    print((auc, auc_std), inf_time)


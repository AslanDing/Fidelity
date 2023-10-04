import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiments_gt_new_fids,replication_sp


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

# this py is to calculate new fid+ fid- and fid\Delta of ground truth by given k_plus and k_minus

(auc, auc_std), inf_time = experiments_gt_new_fids(config.args.explainer,k_plus=4,k_minus=15)

print((auc, auc_std), inf_time)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiments_editdistance_ori_fids,replication_sp


_dataset = 'treegrids'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
_explainer = 'pgexplainer'  # One of: pgexplainer, gnnexplainer

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

(auc, auc_std), inf_time = experiments_editdistance_ori_fids(config.args.explainer, key=(4,0)) #,k_p=1,k_m=1

print((auc, auc_std), inf_time)


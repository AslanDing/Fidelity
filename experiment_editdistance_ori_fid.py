import os
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiments_editdistance_ori_fids


_dataset = 'mutag'      # One of: treecycles, treegrids, ba2motifs, mutag
_explainer = 'pgexplainer'  # One of: pgexplainer, gnnexplainer

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, extension

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"
print(config_path)
config = Selector(config_path)

extension = (_folder == 'extension')

config.args.explainer.model = 'PGIN' # PGIN for gin , PG for gcn
(auc, auc_std), inf_time = experiments_editdistance_ori_fids(config.args.explainer,seeds_num=1) 

print((auc, auc_std), inf_time)


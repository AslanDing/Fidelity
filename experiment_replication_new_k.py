import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import replication_sp_new_k

import nni
from nni.utils import merge_parameter

_dataset = 'ba2motifs'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
_explainer = 'pgexplainer' # One of: pgexplainer, gnnexplainer

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, extension

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"
print(config_path)
config = Selector(config_path)

nni_params = nni.get_next_parameter()
config.args.explainer = merge_parameter(config.args.explainer, nni_params)


extension = (_folder == 'extension')

# config.args.explainer.seeds = [0]

replication_sp_new_k(config.args.explainer, extension,
                            run_qual=False,results_store=False,extend=False)

# (auc, auc_std), inf_time = replication_sp_new_k(config.args.explainer, extension,
#                             run_qual=False,results_store=False,extend=False)

# print((auc, auc_std), inf_time)
# nni.report_final_result(auc)

"""
  0%|          | 0/200 [00:00<?, ?it/s]k  1
fid_minus_mean 0.11479389
fid_minus_label_mean 0.3671578947368421
distance_mean 0.024643548
100%|██████████| 200/200 [01:36<00:00,  2.06it/s]
  0%|          | 0/200 [00:00<?, ?it/s]k  5
fid_minus_mean 0.15132421
fid_minus_label_mean 0.47972499999999996
distance_mean 0.04694878
100%|██████████| 200/200 [01:42<00:00,  1.95it/s]
  0%|          | 0/200 [00:00<?, ?it/s]k  10
fid_minus_mean 0.16041397
fid_minus_label_mean 0.48
distance_mean 0.07634379
100%|██████████| 200/200 [01:38<00:00,  2.04it/s]
k  15
fid_minus_mean 0.1708197
fid_minus_label_mean 0.48
distance_mean 0.10989431

"""




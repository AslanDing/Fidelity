import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import fid_edit_distance_ab,new_fid_edit_distance_ab

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

new_fid_edit_distance_ab(config.args.explainer, extension,
                            run_qual=False,results_store=False,extend=False,key=(3,0),k_p=0,k_m=10)

# (auc, auc_std), inf_time = replication_sp_new_k(config.args.explainer, extension,
#                             run_qual=False,results_store=False,extend=False)

# print((auc, auc_std), inf_time)
# nni.report_final_result(auc)





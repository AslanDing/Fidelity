import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiments_editdistance_ori_fids,replication_sp

datasets = [ 'treecycles', 'treegrids','bashapes','bacommunity', 'ba2motifs', 'mutag']  # 'treecycles', 'treegrids','bashapes','bacommunity',
for _dataset in datasets:
    #_dataset = 'bacommunity'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
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
    config.args.explainer.model='PGIN'
    experiments_editdistance_ori_fids(config.args.explainer, seeds_num=1)  # ,k_p=1,k_m=1

    # print((auc, auc_std), inf_time)
    # exit(0)

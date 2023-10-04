import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiment_new_fid_ratio_editdistance_time,replication_sp

# , 'treegrids', 'ba2motifs', 'mutag'
datasets = [  'ba2motifs']  # 'treecycles', 'ba2motifs', 'mutag' 'bashapes','bacommunity', , 'treegrids', 'ba2motifs', 'mutag'
for _dataset in datasets:
    for  max_length in [10,20,30,40,50,60,70,80,90,100]:
        #_dataset = 'treecycles'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
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
        # keys = [(1,0),(2,0),(3,0),(4,0)]
        # for key in keys:
        #     experiment_new_fid_ab_editdistance(config.args.explainer, key=key,k_p=1,k_m=1)

        # config.args.explainer.model = 'PGIN'
        experiment_new_fid_ratio_editdistance_time(config.args.explainer, k_p=0.1, k_m=0.1, seeds_num=10,max_length=max_length)

        # # keys = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,1),(2,2),(3,3),(1,0),(2,0),(3,0),(4,0)]
        # keys = [(4,0),(3,0),(2,0),(1,0),(3,3),(2,2),(1,1),(0,5),(0,4),(0,3),(0,2),(0,1)]
        # for key in keys:
        #     experiment_new_fid_ab_editdistance(config.args.explainer, key=key,k_p=1,k_m=1)

        # print((auc, auc_std), inf_time)

"""
time consuming:


GCN:
syn3
time  13.343389558792115 0.31583134668409796
syn4
time  64.43943998813629 0.3470766226138392
ba2
time  48.97495810985565 0.03530905822380189
mutag
time  302.3786715507507 3.285786064615021

GIN:

syn3
time  7.744099640846253 0.23193616148673832

syn4
time  37.6479957818985 0.17555010127835508

ba2
time  30.530636620521545 0.0692979810380135


mutag
time  198.93237578868866 0.6500255243299172

"""

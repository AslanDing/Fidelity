import os
import glob
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiment_new_fid_explainers,experiment_new_fid_explainers_ex

datasets = [ 'treecycles','treegrids','ba2motifs','mutag'] # 'bashapes','bacommunity', 'treecycles','treegrids',
for _dataset in datasets:
    # _dataset = 'bashapes'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
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

    # config.args.explainer.model = 'PGIN'

    # this py is to calculate new fid+ fid- and fid\Delta of different edit distances
    # key -> (remove, add)
    model_select = 'gcn' #''PGIN'
    name_dict= {'treecycles':'syn3', 'treegrids':'syn4','bashapes':'syn1','bacommunity':'syn2', 'ba2motifs':'ba2', 'mutag':'mutag'}
    explainers = ['gnnexplainer','pgexplainer','subgraphx','pgmexplainer']
    for index in range(2,4):
        explainer_name = explainers[index]
        explainer_dir = './data/%s'%explainer_name
        dataset_name = config.args.explainer.dataset

        dataset_nikename = name_dict[_dataset]
        if explainer_name == 'gnnexplainer' :
            npy_name_list = glob.glob(explainer_dir+"/%s/*%s*dict.npy"%(model_select,dataset_nikename))
            for npy_name in npy_name_list:
                if dataset_name in npy_name:
                    if _dataset =="ba2motifs":
                        config.args.explainer.explainer= "GNN"
                    experiment_new_fid_explainers(config.args.explainer,k_p=0.1,k_m=0.1, path=npy_name,
                            expl_name=explainer_name,save_name=npy_name.replace('.npy','_ours.npy'))
        elif explainer_name == 'pgexplainer':
            npy_name_list = glob.glob(explainer_dir + "/%s/*%s*dict.npy" % (model_select, dataset_nikename))
            for npy_name in npy_name_list:
                if dataset_name in npy_name:
                    if _dataset == "ba2motifs":
                        config.args.explainer.explainer = "GNN"
                    experiment_new_fid_explainers(config.args.explainer, k_p=0.1, k_m=0.1, path=npy_name,
                                                  expl_name=explainer_name,
                                                  save_name=npy_name.replace('.npy', '_ours.npy'))
        else:
            npy_name_list = glob.glob(explainer_dir + "/%s/*%s*dict.npy" % (model_select, dataset_nikename))
            for npy_name in npy_name_list:
                if dataset_name in npy_name:
                    experiment_new_fid_explainers_ex(config.args.explainer, k_p=0.1,k_m=0.1,path=npy_name,
                                                  expl_name=explainer_name,
                                                  save_name=npy_name.replace('.npy', '_ours.npy'))


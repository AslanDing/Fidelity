import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication_table import experiments_explainers_ori_fids,replication_sp

datasets = [ 'treecycles','treegrids','bashapes','bacommunity','ba2motifs','mutag'] #
for _dataset in datasets:
    #_dataset = 'bashapes'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
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

    config.args.explainer.model = 'PGIN'

    # this py is to calculate new fid+ fid- and fid\Delta of different edit distances
    # key -> (remove, add)
    model_select = 'PGIN'
    name_dict= {'treecycles':'syn3', 'treegrids':'syn4','bashapes':'syn1','bacommunity':'syn2', 'ba2motifs':'ba2', 'mutag':'mutag'}
    explainers = ['gnnexplainer','pgexplainer'] # ,'subgraphx','pgmexplainer'
    for index in range(2):
        explainer_name = explainers[index]
        explainer_dir = './data/%s'%explainer_name
        dataset_name = config.args.explainer.dataset

        if explainer_name == 'gnnexplainer':

            npy_name_list = glob.glob(explainer_dir+"/*_dict.npy")  # /%s
            for npy_name in npy_name_list:
                if dataset_name in npy_name:
                    if _dataset =="ba2motifs":
                        config.args.explainer.explainer= "GNN"
                    experiments_explainers_ori_fids(config.args.explainer, path=npy_name, expl_name=explainer_name)
                    # path = npy_name #explainer_dir+"/gcn/GNN_%s_weight_dict.npy"%dataset_name
                    # weights_dict = np.load(path,allow_pickle=True).item()
                    # for key in weights_dict.keys():
                    #     weight = weights_dict[key]
                    #     print(explainer_name,dataset_name,key,path)

        elif explainer_name == 'pgexplainer':
            npy_name_list = glob.glob(explainer_dir+"/*_dict.npy")
            for npy_name in npy_name_list:
                if dataset_name in npy_name:
                    experiments_explainers_ori_fids(config.args.explainer, path=npy_name, expl_name=explainer_name)
                    # path = npy_name #explainer_dir+"/gcn/GNN_%s_weight_dict.npy"%dataset_name
                    # weights_dict = np.load(path,allow_pickle=True).item()
                    # for key in weights_dict.keys():
                    #     weight = weights_dict[key]
                    #     print(explainer_name,dataset_name,key,path)
                    #     experiments_explainers_ori_fids(config.args.explainer, weights=weight,expl_name = explainer_name)
        else:
            # find all weight
            weight_list = os.listdir(explainer_dir)
            for weight_name in weight_list:
                if dataset_name in weight_name:
                    weight = np.load(weight_name,allow_pickle=True)
                    print(explainer_name,dataset_name,weight_name)
                    experiments_explainers_ori_fids(config.args.explainer, weights=weight)



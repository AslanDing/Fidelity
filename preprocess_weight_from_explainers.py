import os
import glob
import numpy as np

def process_gnn_weights_graphs(weights):
    sparsity = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    weight_dict = {}
    for sp in sparsity:
        weight_list_sp = []
        for weight in weights:
            indices = np.argsort(weight)
            index = int(weight.shape[0]*sp)
            index_value = weight[indices[index]]
            bin_weight = np.where(weight>=index_value,np.ones_like(weight),np.zeros_like(weight))
            #spar = 1 - bin_weight.sum() / bin_weight.shape[0]
            weight_list_sp.append(bin_weight)

        weight_dict[int(sp*100)] = weight_list_sp
    return weight_dict


def process_gnn_weights_nodes(weights):
    sparsity = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    weight_dict = {}
    for sp in sparsity:
        weight_list_sp = []
        for weight in weights:
            indices = np.argsort(weight)
            index = int(weight.shape[0] * sp)
            index_value = weight[indices[index]]
            bin_weight = np.where(weight >= index_value, np.ones_like(weight), np.zeros_like(weight))
            # spar = 1 - bin_weight.sum() / bin_weight.shape[0]
            weight_list_sp.append(bin_weight)

        weight_dict[int(sp * 100)] = weight_list_sp
    return weight_dict

def process_nodes_weights(weights):
    sparsity = []
    for weight in weights:
        non_zeros = weight.sum()
        if non_zeros < 0.5:
            continue
        sp = non_zeros/weight.shape[0]
        sparsity.append(sp)
    print(np.mean(sparsity))
    # pass


def process_explainer_weights_graphs(weights):
    weight_dict = {}
    for key in weights.keys():
        weight_list = weights[key]
        # weight_list_ = []
        sp_list = []
        for weight in weight_list:
            spar = 1 - weight.sum() / weight.shape[0]
            sp_list.append(spar)
            # print(sp)
        sp_mean = np.array(sp_list).mean()
        print("mean :",sp_mean)
        weight_dict[int(sp_mean*100)]=weight_list
    return weight_dict
def process_explainer_weights_nodes(weights):
    weight_dict = {}
    for key in weights.keys():
        weight_list = weights[key]

        sp_list = []
        for weight in weight_list:
            spar = 1 - weight.sum() / weight.shape[0]
            sp_list.append(spar)
            # print(sp)
        sp_mean = np.array(sp_list).mean()
        print("mean :", sp_mean)
        weight_dict[int(sp_mean * 100)] = weight_list
    return weight_dict






if __name__=="__main__":

    explainer_lists = ['pgmexplainer','subgraphx'] # 'gnnexplainer','pgexplainer',
    dataset_lists = ['ba2','mutag','syn3','syn4'] #'syn1','syn2',
    #index = 1
    #dataset_idx = 2


    for index in range(0,2):
        for dataset_idx in range(0,4):
            data_dict={}
            explainer_name = explainer_lists[index]
            dataset_name = dataset_lists[dataset_idx]
            #dataset_name = dataset_lists[dataset_idx]
            name_path_list = glob.glob('./data/%s/gcn/*%s_*.npy'%(explainer_name,dataset_name))
            for name_path in name_path_list:
                path = name_path
                if dataset_lists[dataset_idx] in path:
                    pass
                else:
                    continue
                name_list = path.split('_')
                fake_sparsity = name_list[-2]
                #path = r'./data/%s/gcn/%s_weight.npy'%(explainer_name,dataset_name)
                weights = np.load(path,allow_pickle=True)
                data_dict[fake_sparsity] = weights
                # if 'syn' in path:
                #     weight_dict = process_gnn_weights_nodes(weights)
                # else:
                #     weight_dict = process_gnn_weights_graphs(weights)
            if len(name_path_list)>1:
                np.save(name_path_list[0].replace('.npy','_dict'),data_dict)


    exit(0)

    for index in range(0,2):
        #for dataset_idx in range(0,1):

            explainer_name = explainer_lists[index]
            #dataset_name = dataset_lists[dataset_idx]
            name_path_list = glob.glob('./data/%s/gcn/*.npy'%explainer_name)
            for name_path in name_path_list:
                path = name_path
                #path = r'./data/%s/gcn/%s_weight.npy'%(explainer_name,dataset_name)
                weights = np.load(path,allow_pickle=True)

                if 'syn' in path:
                    weight_dict = process_gnn_weights_nodes(weights)
                else:
                    weight_dict = process_gnn_weights_graphs(weights)

                np.save(path.replace('.npy','_dict'),weight_dict)

    exit(0)



    dir_path = ['../GraphXAI-main/examples','../subgraphX-dig/examples/xgraph']
    for index in range(3,4):
        for dataset_idx in range(2,6):

            explainer_name = explainer_lists[index]
            dataset_name = dataset_lists[dataset_idx]
            dir = dir_path[index-2]

            name_list = glob.glob(dir+'/'+f'*{dataset_name}*')
            # lists = os.listdir(dir)
            weight_dict = {}
            for path in name_list:
                weight = np.load(path, allow_pickle=True)
                name = path.split('/')[-1]
                name_split = name.split('_')
                if len(name_split) < 3:
                    continue
                sp = int(name_split[-2])
                weight_dict[sp] = weight

            # weights = np.load(path,allow_pickle=True)

            if 'syn' in dataset_name:
                process_explainer_weights_nodes(weight_dict)
            else:
                weight_dict = process_explainer_weights_graphs(weight_dict)
            save_name = dir + '/../%s_weight_dict'%dataset_name
            np.save(save_name,weight_dict)

"""


"""



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame
from scipy.sparse import coo_matrix
import math

def visualization(data_dict):

    for key in data_dict.keys():
        data = data_dict[key]
        remove_axis = data[0]
        add_axis = data[1]
        value = data[2]
        sp_matrix = coo_matrix((value,(remove_axis,add_axis)))
        data = DataFrame(sp_matrix.todense())

        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        plt.title('%s'%key)
        plt.show()
        # plt.savefig('./for_show/heatmap/%s.png'%key)
        plt.close()

def visualization_fid_new(data_dict,dataset_name='treecycle',k_p = 1, k_m = 1 ):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([length,length])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            value = data[value_count]
            matrix[remove_axis,add_axis] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        plt.title('%s'%name)
        if 'plus' in name:
            plt.savefig('./for_show/heatmap/%s_%s_new_fid_%d.png'%(dataset_name,name,k_p))
            plt.savefig('./for_show/heatmap/%s_%s_new_fid_%d.pdf'%(dataset_name,name,k_p))
        elif 'minus' in name:
            plt.savefig('./for_show/heatmap/%s_%s_new_fid_%d.png' % (dataset_name, name, k_m))
            plt.savefig('./for_show/heatmap/%s_%s_new_fid_%d.pdf' % (dataset_name, name, k_m))
        else:
            plt.savefig('./for_show/heatmap/%s_%s_new_fid_%d_%d.png' % (dataset_name, name, k_p,k_m))
            plt.savefig('./for_show/heatmap/%s_%s_new_fid_%d_%d.pdf' % (dataset_name, name, k_p,k_m))
        plt.close()

def visualization_fid_ori(data_dict,dataset_name='treecycle'):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([length,length])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            value = data[value_count]
            matrix[remove_axis,add_axis] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        plt.title('%s'%name)
        # plt.show()
        plt.savefig('./for_show/heatmap/%s_%s_ori_fid.png'%(dataset_name,name))
        plt.savefig('./for_show/heatmap/%s_%s_ori_fid.pdf'%(dataset_name,name))
        plt.close()

def data_gcn_gen():
    #
    syn1_prob_data = {
        'fid+':[[0, 0,0,0,0,0, 1,2,3, 1,2,3,4], # remove
                [0, 1,2,3,4,5, 1,2,3, 0,0,0,0], # add
                [0.51,0.51,0.510,0.509,0.510,0.511,0.514,0.499,0.457,0.516,0.497,0.459,0.384]],
        'fid+k1': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                 [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                 [0.215	,0.186,0.164,0.148,0.133,0.122,0.182,0.147,0.115,0.215,0.214,0.215,0.214]],
        'fid+k2': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                 [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                 [0.353,0.316,0.285,0.260,0.239,0.221,0.311,0.263,0.211,0.353,0.353,0.353,0.0]],

        'fid-': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                 [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                 [0.598	,0.598,0.598,0.598,0.598,0.599,0.598,0.578,0.597,0.598,0.578,0.598,0.598]],
        'fid-k1': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                 [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                 [0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.006,0.006,0.005,0.006,0.006,0.006]],
        'fid-k5': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                 [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                 [0.023, 0.024,0.024,0.024,0.024,0.023,0.024,0.026,0.028,0.026,0.027,0.028,0.0]],

    
        'fid-delta': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                 [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                 [-0.088,-0.088	,-0.088,-0.089,-0.087,-0.088,-0.084,-0.099,-0.140,-0.082,-0.101,-0.139,-0.214]],
        'fid-delta-(1,1)': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                   [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                   [0.21,0.181,0.159,0.143,0.128,0.117,0.177,0.142,0.109,0.210,0.209,0.209,0.208]],
        'fid-delta-(2,5)': [[0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4],  # remove
                   [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0],  # add
                   [0.330,0.292,0.262,0.236,0.215,0.197,0.286,0.237,0.183,0.327,0.326,0.325,0.321]],
    }
    return syn1_prob_data




if __name__ == "__main__":
    path = r'redata/syn3_results_ori_fid_1.npy'
    data = np.load(path,allow_pickle=True).item()
    visualization_fid_ori(data,'syn3')
    path = r'redata/syn3_results_new_fid_1_1_seeds_1.npy'
    data = np.load(path,allow_pickle=True).item()
    visualization_fid_new(data,'syn3')

import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame
from scipy.sparse import coo_matrix
import math

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

from scipy.stats import spearmanr



def visualization_fid_new_ratio(data_dict,save_path=None,k_p = 1, k_m = 1 ):
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

    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True,xticklabels=x_ticks,yticklabels=x_ticks,fmt='.2f',annot=True)
        plt.tick_params(labeltop=True)
        plt.tick_params(labelbottom=False)
        # plt.title('%s'%name)
        if 'plus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        elif 'minus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        else:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        plt.close()

def visualization_fid_ori(data_dict,save_path=None):
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
    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        # plt.title('%s'%name)
        # plt.show()
        plt.savefig(save_path+'%s.png'%name)
        plt.savefig(save_path+'%s.pdf'%name)
        plt.close()

def visualization_fid_ori_ratio(data_dict,save_path=None):
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

    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True,xticklabels=x_ticks,yticklabels=x_ticks,fmt='.2f',annot=True)
        plt.tick_params(labeltop=True)
        plt.tick_params(labelbottom=False)
        # plt.title('%s'%name)
        # plt.show()
        plt.savefig(save_path+'%s.png'%name)
        plt.savefig(save_path+'%s.pdf'%name)
        plt.close()


def test_correction(name = 'syn1',dir = './redata/gin'):
    # gcn
    # dir = './redata/gin'

    # name = 'ba2'

    exp_edges_dict = {'syn1':6.66666666666,
                      'syn2':13.3166666666,
                      'syn3':6.233333333333,
                      'syn4':11.30103806,
                      'ba2':5.49,
                    'mutag':2.811822660098}
    non_exp_edges_dict = {'syn1':362.85,
                          'syn2':493.7333333,
                          'syn3':7.416666666,
                          'syn4':2.37370242,
                          'ba2':19.975,
                     'mutag':25.8857142857}

    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']

    namelist_orifid = glob.glob(dir+'/*ori_fid*.npy')
    namelist_newfid = glob.glob(dir + '/*new_fid*.npy')

    orifid_path = None
    for oripath in namelist_orifid:
        if name in oripath:
            orifid_path = oripath


    newfid_path = None
    for newpath in namelist_newfid:
        if name in newpath:
            newfid_path = newpath

    ori_matrix = np.load(orifid_path, allow_pickle=True).item()
    new_matrix = np.load(newfid_path, allow_pickle=True).item()

    auc_matrix = np.zeros([len(x_ticks), len(x_ticks)])   # 12
    iou_matrix = np.zeros([len(x_ticks), len(x_ticks)])   # 14

    # distance_plus = np.zeros([len(x_ticks), len(x_ticks)]) # 16
    # distance_mean = np.zeros([len(x_ticks), len(x_ticks)]) # 18

    editdistance_matrix = np.zeros([len(x_ticks), len(x_ticks)])

    ori_fidplus_matrix_prob = np.zeros([len(x_ticks), len(x_ticks)]) # 0
    new_fidplus_matrix_prob = np.zeros([len(x_ticks), len(x_ticks)])

    ori_fidplus_matrix_acc = np.zeros([len(x_ticks), len(x_ticks)]) # 2
    new_fidplus_matrix_acc = np.zeros([len(x_ticks), len(x_ticks)])

    ori_fidminus_matrix_prob = np.zeros([len(x_ticks), len(x_ticks)]) # 4
    new_fidminus_matrix_prob = np.zeros([len(x_ticks), len(x_ticks)])

    ori_fidminus_matrix_acc = np.zeros([len(x_ticks), len(x_ticks)]) # 6
    new_fidminus_matrix_acc = np.zeros([len(x_ticks), len(x_ticks)])

    ori_fiddelta_matrix_prob = np.zeros([len(x_ticks), len(x_ticks)]) # 8
    new_fiddelta_matrix_prob = np.zeros([len(x_ticks), len(x_ticks)])

    ori_fiddelta_matrix_acc = np.zeros([len(x_ticks), len(x_ticks)]) # 10
    new_fiddelta_matrix_acc = np.zeros([len(x_ticks), len(x_ticks)])

    for key in ori_matrix.keys():
        remove_axis = key[0]  # remove
        add_axis = key[1]  # add
        if remove_axis in x_ticks and add_axis in x_ticks:
            ori_data = ori_matrix[key]
            new_data = new_matrix[key]

            edit_distnce = exp_edges_dict[name]*float(remove_axis) + non_exp_edges_dict[name]*float(add_axis)
            editdistance_matrix[dict_ratio[remove_axis], dict_ratio[add_axis]] = edit_distnce

            # ori
            value = ori_data[0]
            ori_fidplus_matrix_prob[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = ori_data[2]
            ori_fidplus_matrix_acc[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = ori_data[4]
            ori_fidminus_matrix_prob[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = ori_data[6]
            ori_fidminus_matrix_acc[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = ori_data[8]
            ori_fiddelta_matrix_prob[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = ori_data[10]
            ori_fiddelta_matrix_acc[dict_ratio[remove_axis], dict_ratio[add_axis]] = value

            #new
            value = new_data[0]
            new_fidplus_matrix_prob[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = new_data[2]
            new_fidplus_matrix_acc[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = new_data[4]
            new_fidminus_matrix_prob[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = new_data[6]
            new_fidminus_matrix_acc[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = new_data[8]
            new_fiddelta_matrix_prob[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = new_data[10]
            new_fiddelta_matrix_acc[dict_ratio[remove_axis], dict_ratio[add_axis]] = value

            value = new_data[12]
            auc_matrix[dict_ratio[remove_axis], dict_ratio[add_axis]] = value
            value = new_data[14]
            iou_matrix[dict_ratio[remove_axis], dict_ratio[add_axis]] = value

    def cal(matrix):
        statisticfloat_list = []
        pvaluefloat_list = []
        for i in range(matrix.shape[1]):
            statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix[:, i], b=matrix[:, i])
            statisticfloat_list.append(statisticfloat)
            pvaluefloat_list.append(pvaluefloat)
        return statisticfloat_list,pvaluefloat_list
    # cal
    print("dataset name ",name)
    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidplus_matrix_prob[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fidplus_matrix_prob)
    print("edit_dis -> ori fid plus prob:",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidplus_matrix_acc[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)

    statisticfloat_list, pvaluefloat_list = cal(ori_fidplus_matrix_acc)
    #statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fidplus_matrix_acc.reshape(-1,) )
    print("edit_dis -> ori fid plus acc :",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidminus_matrix_prob[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fidminus_matrix_prob)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fidminus_matrix_prob.reshape(-1,) )
    print("edit_dis -> ori fid minus prob:",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidminus_matrix_acc[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fidminus_matrix_acc)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fidminus_matrix_acc.reshape(-1,) )
    print("edit_dis -> ori fid minus acc :",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fiddelta_matrix_prob[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fiddelta_matrix_prob)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fiddelta_matrix_prob.reshape(-1,) )
    print("edit_dis -> ori fid delta prob:",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())


    statisticfloat_list, pvaluefloat_list = cal(ori_fiddelta_matrix_acc)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fiddelta_matrix_acc.reshape(-1,) )
    print("edit_dis -> ori fid delta acc :",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    ##########################
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidplus_matrix_prob.reshape(-1, ))
    print()
    statisticfloat_list, pvaluefloat_list = cal(new_fidplus_matrix_prob)
    print("edit_dis -> new fid plus prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    statisticfloat_list, pvaluefloat_list = cal(new_fidplus_matrix_acc)
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ), b=new_fidplus_matrix_acc.reshape(-1, ))
    print("edit_dis -> new fid plus acc :",  np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())


    statisticfloat_list, pvaluefloat_list = cal(new_fidminus_matrix_prob)
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidminus_matrix_prob.reshape(-1, ))
    print("edit_dis -> new fid minus prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidminus_matrix_acc.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fidminus_matrix_acc)
    print("edit_dis -> new fid minus acc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fiddelta_matrix_prob.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fiddelta_matrix_prob)
    print("edit_dis -> new fid delta prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fiddelta_matrix_acc.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fiddelta_matrix_acc)
    print("edit_dis -> new fid delta acc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    #################
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=auc_matrix.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(auc_matrix)
    print("edit_dis -> auc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=iou_matrix.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(iou_matrix)
    print("edit_dis -> iou :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

def test_correction_sp(name = 'syn1',dir = './redata/gin'):
    # gcn
    # dir = './redata/gin'

    # name = 'ba2'

    exp_edges_dict = {'syn1':6.66666666666,
                      'syn2':13.3166666666,
                      'syn3':6.233333333333,
                      'syn4':11.30103806,
                      'ba2':5.49,
                    'mutag':2.811822660098}
    non_exp_edges_dict = {'syn1':362.85,
                          'syn2':493.7333333,
                          'syn3':7.416666666,
                          'syn4':2.37370242,
                          'ba2':19.975,
                     'mutag':25.8857142857}

    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']

    namelist_orifid = glob.glob(dir+'/*ori_fid*.npy')
    namelist_newfid = glob.glob(dir + '/*new_fid*.npy')

    orifid_path = None
    for oripath in namelist_orifid:
        if name in oripath:
            orifid_path = oripath


    newfid_path = None
    for newpath in namelist_newfid:
        if name in newpath:
            newfid_path = newpath

    ori_matrix = np.load(orifid_path, allow_pickle=True).item()
    new_matrix = np.load(newfid_path, allow_pickle=True).item()

    auc_matrix = np.zeros([len(x_ticks), 1])   # 12
    iou_matrix = np.zeros([len(x_ticks), 1])   # 14

    # distance_plus = np.zeros([len(x_ticks), len(x_ticks)]) # 16
    # distance_mean = np.zeros([len(x_ticks), len(x_ticks)]) # 18

    editdistance_matrix = np.zeros([len(x_ticks), 1])

    ori_fidplus_matrix_prob = np.zeros([len(x_ticks), 1]) # 0
    new_fidplus_matrix_prob = np.zeros([len(x_ticks), 1])

    ori_fidplus_matrix_acc = np.zeros([len(x_ticks), 1]) # 2
    new_fidplus_matrix_acc = np.zeros([len(x_ticks), 1])

    ori_fidminus_matrix_prob = np.zeros([len(x_ticks), 1]) # 4
    new_fidminus_matrix_prob = np.zeros([len(x_ticks), 1])

    ori_fidminus_matrix_acc = np.zeros([len(x_ticks), 1]) # 6
    new_fidminus_matrix_acc = np.zeros([len(x_ticks), 1])

    ori_fiddelta_matrix_prob = np.zeros([len(x_ticks), 1]) # 8
    new_fiddelta_matrix_prob = np.zeros([len(x_ticks), 1])

    ori_fiddelta_matrix_acc = np.zeros([len(x_ticks), 1]) # 10
    new_fiddelta_matrix_acc = np.zeros([len(x_ticks), 1])

    for key in ori_matrix.keys():
        remove_axis = key[0]  # remove
        add_axis = key[1]  # add
        if remove_axis in x_ticks and add_axis in x_ticks:
            ori_data = ori_matrix[key]
            new_data = new_matrix[key]

            edit_distnce = exp_edges_dict[name]*float(remove_axis)*2 #+ non_exp_edges_dict[name]*float(add_axis)
            editdistance_matrix[dict_ratio[remove_axis], 0] = edit_distnce

            # ori
            value = ori_data[0]
            ori_fidplus_matrix_prob[dict_ratio[remove_axis], 0] = value
            value = ori_data[2]
            ori_fidplus_matrix_acc[dict_ratio[remove_axis], 0] = value
            value = ori_data[4]
            ori_fidminus_matrix_prob[dict_ratio[remove_axis], 0] = value
            value = ori_data[6]
            ori_fidminus_matrix_acc[dict_ratio[remove_axis], 0] = value
            value = ori_data[8]
            ori_fiddelta_matrix_prob[dict_ratio[remove_axis], 0] = value
            value = ori_data[10]
            ori_fiddelta_matrix_acc[dict_ratio[remove_axis], 0] = value

            #new
            value = new_data[0]
            new_fidplus_matrix_prob[dict_ratio[remove_axis], 0] = value
            value = new_data[2]
            new_fidplus_matrix_acc[dict_ratio[remove_axis], 0] = value
            value = new_data[4]
            new_fidminus_matrix_prob[dict_ratio[remove_axis], 0] = value
            value = new_data[6]
            new_fidminus_matrix_acc[dict_ratio[remove_axis], 0] = value
            value = new_data[8]
            new_fiddelta_matrix_prob[dict_ratio[remove_axis], 0] = value
            value = new_data[10]
            new_fiddelta_matrix_acc[dict_ratio[remove_axis], 0] = value

            value = new_data[12]
            auc_matrix[dict_ratio[remove_axis], 0] = value
            value = new_data[14]
            iou_matrix[dict_ratio[remove_axis], 0] = value

    def cal(matrix):
        statisticfloat_list = []
        pvaluefloat_list = []
        # for i in range(matrix.shape[1]):
        statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix[:, 0], b=matrix[:, 0])
        statisticfloat_list.append(statisticfloat)
        pvaluefloat_list.append(pvaluefloat)
        return statisticfloat_list,pvaluefloat_list
    # cal
    print("dataset name ",name)
    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidplus_matrix_prob[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fidplus_matrix_prob)
    print("edit_dis -> ori fid plus prob:",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidplus_matrix_acc[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)

    statisticfloat_list, pvaluefloat_list = cal(ori_fidplus_matrix_acc)
    #statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fidplus_matrix_acc.reshape(-1,) )
    print("edit_dis -> ori fid plus acc :",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidminus_matrix_prob[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fidminus_matrix_prob)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fidminus_matrix_prob.reshape(-1,) )
    print("edit_dis -> ori fid minus prob:",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fidminus_matrix_acc[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fidminus_matrix_acc)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fidminus_matrix_acc.reshape(-1,) )
    print("edit_dis -> ori fid minus acc :",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat_list=[], pvaluefloat_list=[]
    # for i in range(editdistance_matrix.shape[1]):
    #     statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix[:,i], b=ori_fiddelta_matrix_prob[:,i] )
    #     statisticfloat_list.append(statisticfloat)
    #     pvaluefloat_list.append(pvaluefloat)
    statisticfloat_list, pvaluefloat_list = cal(ori_fiddelta_matrix_prob)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fiddelta_matrix_prob.reshape(-1,) )
    print("edit_dis -> ori fid delta prob:",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())


    statisticfloat_list, pvaluefloat_list = cal(ori_fiddelta_matrix_acc)
    # statisticfloat ,pvaluefloat =  spearmanr( a=editdistance_matrix.reshape(-1,), b=ori_fiddelta_matrix_acc.reshape(-1,) )
    print("edit_dis -> ori fid delta acc :",np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    ##########################
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidplus_matrix_prob.reshape(-1, ))
    print()
    statisticfloat_list, pvaluefloat_list = cal(new_fidplus_matrix_prob)
    print("edit_dis -> new fid plus prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    statisticfloat_list, pvaluefloat_list = cal(new_fidplus_matrix_acc)
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ), b=new_fidplus_matrix_acc.reshape(-1, ))
    print("edit_dis -> new fid plus acc :",  np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())


    statisticfloat_list, pvaluefloat_list = cal(new_fidminus_matrix_prob)
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidminus_matrix_prob.reshape(-1, ))
    print("edit_dis -> new fid minus prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidminus_matrix_acc.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fidminus_matrix_acc)
    print("edit_dis -> new fid minus acc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fiddelta_matrix_prob.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fiddelta_matrix_prob)
    print("edit_dis -> new fid delta prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fiddelta_matrix_acc.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fiddelta_matrix_acc)
    print("edit_dis -> new fid delta acc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    #################
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=auc_matrix.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(auc_matrix)
    print("edit_dis -> auc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=iou_matrix.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(iou_matrix)
    print("edit_dis -> iou :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())



if __name__ == "__main__":
    dir_list = ['./redata_0922/gcn','./redata_0922/gin']
    dataset_list = ['ba2'] #'syn1','syn2','syn3','syn4','ba2','mutag'
    for dir in dir_list:
        for dataname in dataset_list:
            print(dir)
            test_correction_sp(dataname,dir)
    exit(0)


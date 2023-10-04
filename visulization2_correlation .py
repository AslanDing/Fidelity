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

def visualization_fid_new(data_dict,save_path=None,k_p = 1, k_m = 1 ):
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


def testvalue():

    x_ticks = ['0', '1',
               '2',  '3']
    matrix = np.zeros([4, 4])
    for i in range(4):
        for j in range(4):
            matrix[i,j] = i+j*0.1

    data = DataFrame(matrix)
    plt.figure(dpi=120)
    sns.heatmap(data=data, square=True, xticklabels=x_ticks, yticklabels=x_ticks,fmt='.2f',annot=True)
    plt.tick_params(labeltop=True)
    plt.tick_params(labelbottom=False)
    plt.show()

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
    # gcn
    dir = './redata/gin'

    # graph
    namelist = glob.glob(dir+'/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori_ratio(data, path.replace('.npy',''))


    namelist = glob.glob(dir + '/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new_ratio(data, path.replace('.npy', ''))

    exit(0)
    # direct nodes
    namelist = glob.glob(dir + '/directed_sample/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori(data, path.replace('.npy',''))

    # undirect nodes
    namelist = glob.glob(dir + '/undirect_sample/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori(data, path.replace('.npy',''))

    # gin
    dir = './redata/gin'
    # graph
    namelist = glob.glob(dir+'/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori(data, path.replace('.npy',''))

    # direct nodes
    namelist = glob.glob(dir + '/directed_sample/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori(data, path.replace('.npy',''))

    # undirect nodes
    namelist = glob.glob(dir + '/undirect_sample/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori(data, path.replace('.npy',''))

    # new fid
    dir = './redata/gcn'
    # graph
    namelist = glob.glob(dir + '/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new(data, path.replace('.npy', ''))

    # direct nodes
    namelist = glob.glob(dir + '/directed_sample/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new(data, path.replace('.npy', ''))

    # undirect nodes
    namelist = glob.glob(dir + '/undirect_sample/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new(data, path.replace('.npy', ''))

    # gin
    dir = './redata/gin'
    # graph
    namelist = glob.glob(dir + '/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new(data, path.replace('.npy', ''))

    # direct nodes
    namelist = glob.glob(dir + '/directed_sample/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new(data, path.replace('.npy', ''))

    # undirect nodes
    namelist = glob.glob(dir + '/undirect_sample/*new_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new(data, path.replace('.npy', ''))

    exit(0)
    path = r'redata/syn3_results_ori_fid_1.npy'
    data = np.load(path,allow_pickle=True).item()
    visualization_fid_ori(data,'syn3')
    path = r'redata/syn3_results_new_fid_1_1_seeds_1.npy'
    data = np.load(path,allow_pickle=True).item()
    visualization_fid_new(data,'syn3')

"""
dataset name  syn1
edit_dis -> ori fid plus prob: -0.22316602316602321 0.1907674072690435
edit_dis -> ori fid plus acc : 0.006697148953034402 0.9690776339028445
edit_dis -> ori fid minus prob: -0.1462033462033462 0.39485435416559655
edit_dis -> ori fid minus acc : -0.4142103265102191 0.012021788811137479
edit_dis -> ori fid delta prob: -0.022393822393822396 0.8968535272610024
edit_dis -> ori fid delta acc : 0.20039899648936604 0.24123971071956787

edit_dis -> new fid plus prob: 0.6404118404118405 2.5910792187395473e-05
edit_dis -> new fid plus acc : 0.8191002989928194 1.0184469229365756e-09
edit_dis -> new fid minus prob: -0.7683397683397685 4.4722364144116904e-08
edit_dis -> new fid minus acc : -0.9097071871679302 1.563742087803471e-14
edit_dis -> new fid delta prob: 0.9577863577863578 5.538174444755113e-20
edit_dis -> new fid delta acc : 0.9567260750704669 8.374679418838987e-20

edit_dis -> auc : -0.8054054054054054 3.1453533724300317e-09
edit_dis -> iou : -0.8638352638352639 1.171083997793049e-11


dataset name  syn2
edit_dis -> ori fid plus prob: -0.18867438867438868 0.2704491978900149
edit_dis -> ori fid plus acc : -0.12895752895752896 0.4535097407148043
edit_dis -> ori fid minus prob: 0.050965250965250966 0.7678476369276918
edit_dis -> ori fid minus acc : -0.005662805662805663 0.9738516659471731
edit_dis -> ori fid delta prob: -0.010810810810810811 0.950103020518532
edit_dis -> ori fid delta acc : 0.012356007490673823 0.9429820921226926

edit_dis -> new fid plus prob: 0.952895752895753 3.438058714997171e-19
edit_dis -> new fid plus acc : 0.902864546281053 5.129145271314256e-14
edit_dis -> new fid minus prob: -0.9485199485199486 1.5047728162789892e-18
edit_dis -> new fid minus acc : -0.9516087516087516 5.382852859563301e-19
edit_dis -> new fid delta prob: 0.9480051480051481 1.775026545508608e-18
edit_dis -> new fid delta acc : 0.9490347490347492 1.2734980248736797e-18
edit_dis -> auc : -0.8118404118404119 1.8727587906400048e-09
edit_dis -> iou : -0.8530244530244531 3.9335274574123253e-11


dataset name  syn3
edit_dis -> ori fid plus prob: -0.7297297297297298 4.4259243628903754e-07
edit_dis -> ori fid plus acc : -0.7027027027027027 1.7709726241218843e-06
edit_dis -> ori fid minus prob: 0.6767052767052768 5.864559143253161e-06
edit_dis -> ori fid minus acc : 0.7166023166023167 8.852312084226757e-07
edit_dis -> ori fid delta prob: -0.7531531531531532 1.1575673858196473e-07
edit_dis -> ori fid delta acc : -0.7858429858429858 1.3622788046264227e-08

edit_dis -> new fid plus prob: -0.003346203346203347 0.9845469918813772
edit_dis -> new fid plus acc : 0.0007722007722007723 0.9964337195848452
edit_dis -> new fid minus prob: -0.10965250965250965 0.5243722215437996
edit_dis -> new fid minus acc : -0.06254826254826255 0.7170530389173337
edit_dis -> new fid delta prob: 0.10244530244530246 0.5521465174318699
edit_dis -> new fid delta acc : 0.06460746460746461 0.7081397328920174
edit_dis -> auc : -0.9930501930501932 3.477073262260595e-33
edit_dis -> iou : -0.8617760617760618 1.4866388665045857e-11

dataset name  syn4
edit_dis -> ori fid plus prob: -0.18790218790218793 0.2724508793641028
edit_dis -> ori fid plus acc : 0.7261261261261261 5.374564992513693e-07
edit_dis -> ori fid minus prob: 0.5621621621621622 0.0003594684520427888
edit_dis -> ori fid minus acc : -0.06152262063064674 0.7215064099198722
edit_dis -> ori fid delta prob: -0.8561132561132561 2.8106699395722605e-11
edit_dis -> ori fid delta acc : 0.6262548262548263 4.396954905129252e-05

edit_dis -> new fid plus prob: -0.9384813384813385 2.876917119901389e-17
edit_dis -> new fid plus acc : -0.9469755469755471 2.4576170616033157e-18
edit_dis -> new fid minus prob: 0.960875160875161 1.558426654774916e-20
edit_dis -> new fid minus acc : 0.9670527670527671 8.800746908888367e-22
edit_dis -> new fid delta prob: -0.9341055341055342 8.942543095819253e-17
edit_dis -> new fid delta acc : -0.9747747747747749 9.964780436821175e-24
edit_dis -> auc : -0.8131274131274132 1.6842939617324273e-09
edit_dis -> iou : -0.9989703989703991 2.900753709173796e-47


dataset name  ba2
edit_dis -> ori fid plus prob: -0.24890604890604892 0.14322600615965353
edit_dis -> ori fid plus acc : 0.5293301603593632 0.000901599115576647
edit_dis -> ori fid minus prob: 0.10785070785070787 0.5312511499012005
edit_dis -> ori fid minus acc : -0.3205877768298156 0.056611298822022746
edit_dis -> ori fid delta prob: 0.02831402831402832 0.8697930584031821
edit_dis -> ori fid delta acc : 0.5051310085249208 0.0016770224372405513

edit_dis -> new fid plus prob: 0.7685971685971686 4.398033831787747e-08
edit_dis -> new fid plus acc : 0.851994851994852 4.392427262717494e-11
edit_dis -> new fid minus prob: -0.7328185328185328 3.738097073438887e-07
edit_dis -> new fid minus acc : -0.8532818532818532 3.825999399247565e-11
edit_dis -> new fid delta prob: 0.7696267696267697 4.1124541917935225e-08
edit_dis -> new fid delta acc : 0.8532818532818532 3.825999399247565e-11
edit_dis -> auc : -0.8507078507078508 5.036202790050471e-11
edit_dis -> iou : -0.7680823680823681 4.54759423763886e-08


dataset name  mutag
edit_dis -> ori fid plus prob: -0.6674388674388675 8.735864278911905e-06
edit_dis -> ori fid plus acc : -0.6836550836550838 4.308843663236339e-06
edit_dis -> ori fid minus prob: 0.6805662805662807 4.946555131620319e-06
edit_dis -> ori fid minus acc : 0.5814671814671816 0.00019987172164509355
edit_dis -> ori fid delta prob: -0.9706563706563707 1.2627725654941156e-22
edit_dis -> ori fid delta acc : -0.8604890604890605 1.7223743790323912e-11

edit_dis -> new fid plus prob: 0.865894465894466 9.18914065035358e-12
edit_dis -> new fid plus acc : 0.8190476190476191 1.023057499471021e-09
edit_dis -> new fid minus prob: -0.84993564993565 5.4635771484489384e-11
edit_dis -> new fid minus acc : -0.7886743886743889 1.1124163959088622e-08
edit_dis -> new fid delta prob: 0.8615186615186615 1.5312318142002196e-11
edit_dis -> new fid delta acc : 0.8126126126126128 1.7574604883161942e-09
edit_dis -> auc : -0.8066924066924068 2.839887238372265e-09
edit_dis -> iou : -0.7732303732303732 3.2423867445044604e-08


"""

""""
gin
dataset name  syn1
edit_dis -> ori fid plus prob: -0.13616473616473618 0.4284458807177395
edit_dis -> ori fid plus acc : -0.15294331143787387 0.3731816710628061
edit_dis -> ori fid minus prob: -0.14388674388674388 0.4024679511297866
edit_dis -> ori fid minus acc : -0.16921208868048207 0.32384858420533236
edit_dis -> ori fid delta prob: 0.10990990990990991 0.5233930887934501
edit_dis -> ori fid delta acc : 0.07902187902187902 0.6468691697206936

edit_dis -> new fid plus prob: -0.8980694980694982 1.1202744793530555e-13
edit_dis -> new fid plus acc : -0.8336766496709368 2.751386386653195e-10
edit_dis -> new fid minus prob: 0.738996138996139 2.647935071173352e-07
edit_dis -> new fid minus acc : 0.6903474903474903 3.1767999561172227e-06
edit_dis -> new fid delta prob: -0.7971685971685972 5.9441199477484395e-09
edit_dis -> new fid delta acc : -0.7214929214929217 6.868635236476324e-07
edit_dis -> auc : -0.8054054054054054 3.1453533724300317e-09
edit_dis -> iou : -0.8638352638352639 1.171083997793049e-11

dataset name  syn2
edit_dis -> ori fid plus prob: 0.11171171171171172 0.5165643451595487
edit_dis -> ori fid plus acc : 0.09759412733456646 0.5712212928237935
edit_dis -> ori fid minus prob: -0.25817245817245815 0.1284426484936497
edit_dis -> ori fid minus acc : -0.27363042804607995 0.10634743661759176
edit_dis -> ori fid delta prob: 0.209009009009009 0.22119264566185382
edit_dis -> ori fid delta acc : 0.24287277223855733 0.15349891173275446

edit_dis -> new fid plus prob: 0.9644787644787646 3.0996759497844538e-21
edit_dis -> new fid plus acc : 0.9624195624195625 7.952442503789346e-21
edit_dis -> new fid minus prob: -0.9562419562419564 1.0080575518241447e-19
edit_dis -> new fid minus acc : -0.9567567567567569 8.276264770080746e-20
edit_dis -> new fid delta prob: 0.9590733590733591 3.304272319422919e-20
edit_dis -> new fid delta acc : 0.9562419562419564 1.0080575518241447e-19
edit_dis -> auc : -0.8118404118404119 1.8727587906400048e-09
edit_dis -> iou : -0.8530244530244531 3.9335274574123253e-11

dataset name  syn3
edit_dis -> ori fid plus prob: -0.06872586872586874 0.690428244569177
edit_dis -> ori fid plus acc : -0.09574827736369097 0.5785569382034508
edit_dis -> ori fid minus prob: -0.46409266409266414 0.004356141992479954
edit_dis -> ori fid minus acc : -0.3019562955837077 0.07347976461861677
edit_dis -> ori fid delta prob: 0.42419562419562423 0.009926095248905172
edit_dis -> ori fid delta acc : 0.1647671988565238 0.3369050140954122

edit_dis -> new fid plus prob: 0.36293436293436293 0.029589503999162898
edit_dis -> new fid plus acc : 0.3974259974259975 0.01638620026570573
edit_dis -> new fid minus prob: -0.37400257400257403 0.024635617478231585
edit_dis -> new fid minus acc : -0.4334620334620335 0.008267682455996692
edit_dis -> new fid delta prob: 0.36653796653796655 0.027893994517147424
edit_dis -> new fid delta acc : 0.40051480051480054 0.015495815340729696
edit_dis -> auc : -0.9930501930501932 3.477073262260595e-33
edit_dis -> iou : -0.8617760617760618 1.4866388665045857e-11


dataset name  syn4
edit_dis -> ori fid plus prob: 0.9343629343629345 8.38368629112928e-17
edit_dis -> ori fid plus acc : 0.2847473987257497 0.09232929018884624
edit_dis -> ori fid minus prob: -0.9564993564993566 9.13669556598797e-20
edit_dis -> ori fid minus acc : 0.2847473987257497 0.09232929018884624
edit_dis -> ori fid delta prob: 0.932818532818533 1.2300339653431265e-16
edit_dis -> ori fid delta acc : nan nan

edit_dis -> new fid plus prob: 0.9276241459469307 4.1896041084339006e-16
edit_dis -> new fid plus acc : 0.2847473987257497 0.09232929018884624
edit_dis -> new fid minus prob: -0.9455982548536215 3.7601560948363474e-18
edit_dis -> new fid minus acc : 0.2847473987257497 0.09232929018884624
edit_dis -> new fid delta prob: 0.9276705276705277 4.1456955851247874e-16
edit_dis -> new fid delta acc : nan nan
edit_dis -> auc : -0.8131274131274132 1.6842939617324273e-09
edit_dis -> iou : -0.9989703989703991 2.900753709173796e-47

dataset name  ba2
edit_dis -> ori fid plus prob: 0.2074646074646075 0.2247018229251721
edit_dis -> ori fid plus acc : 0.12094798713129185 0.48226564917160275
edit_dis -> ori fid minus prob: -0.25405405405405407 0.13486652402772442
edit_dis -> ori fid minus acc : -0.0801861683220457 0.6420121015376017
edit_dis -> ori fid delta prob: 0.10064350064350065 0.5591962132399981
edit_dis -> ori fid delta acc : 0.1505791505791506 0.38070226039431665

edit_dis -> new fid plus prob: -0.07207207207207207 0.6761554214137167
edit_dis -> new fid plus acc : -0.07207207207207207 0.6761554214137167
edit_dis -> new fid minus prob: 0.06743886743886744 0.6959462380344117
edit_dis -> new fid minus acc : 0.06023166023166024 0.7271246300663884
edit_dis -> new fid delta prob: -0.06743886743886744 0.6959462380344117
edit_dis -> new fid delta acc : -0.06023166023166024 0.7271246300663884
edit_dis -> auc : -0.8507078507078508 5.036202790050471e-11
edit_dis -> iou : -0.7680823680823681 4.54759423763886e-08

"""

"""
/aul/homes/xzhen019/anaconda3/envs/repeg/bin/python /aul/homes/xzhen019/data/code/XGNN/forICLR24/visulization2_correlation .py 
./redata/gcn
dataset name  syn1
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 0.8666666666666667 0.1885618083164127 0.06933333333333327 0.0980521403245345
edit_dis -> ori fid minus acc : 0.7047619047619048 0.42559815183817773 0.26332361516034986 0.38470176261679867
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -0.942857142857143 0.08081220356417679 0.013854227405247782 0.019592836292702424

edit_dis -> new fid plus prob: -0.5523809523809525 0.5996975806259492 0.06266277939747324 0.04381120078743788
edit_dis -> new fid plus acc : nan nan nan nan
edit_dis -> new fid minus prob: 0.7333333333333334 0.1735320681741771 0.12957045675413012 0.1291739793944295
edit_dis -> new fid minus acc : nan nan nan nan
edit_dis -> new fid delta prob: -0.561904761904762 0.6062411604891257 0.06186200194363456 0.04489198686398153
edit_dis -> new fid delta acc : nan nan nan nan
edit_dis -> auc : -1.0 0.0 0.0 0.0
/aul/homes/xzhen019/anaconda3/envs/repeg/lib/python3.7/site-packages/scipy/stats/stats.py:4484: SpearmanRConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(SpearmanRConstantInputWarning())
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  syn2
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -0.7428571428571429 0.32784599005825726 0.1824606413994169 0.3473087977874299
edit_dis -> new fid plus acc : -0.5361728405140985 0.6694512257053866 0.061399957888040424 0.0760595444836088
edit_dis -> new fid minus prob: 0.9047619047619048 0.042591770999995976 0.015611273080660805 0.013195606535599745
edit_dis -> new fid minus acc : 0.7904761904761907 0.11741740958036145 0.07779203109815346 0.06191431115733338
edit_dis -> new fid delta prob: -0.9238095238095237 0.07126966450997985 0.01526919339164236 0.02560827676986896
edit_dis -> new fid delta acc : -0.8857142857142858 0.08728715609439695 0.02887463556851311 0.03130857479334082
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  syn3
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : 0.19047619047619047 0.5231164245660747 0.34587366375121475 0.2700840242975926
edit_dis -> ori fid minus prob: -0.20952380952380956 0.6341026476844689 0.2740991253644314 0.32247826254422396
edit_dis -> ori fid minus acc : 0.12380952380952383 0.3124041803210805 0.5972478134110787 0.23594772967835378
edit_dis -> ori fid delta prob: 0.10476190476190476 0.5653646506535555 0.3217414965986393 0.29418189453738003
edit_dis -> ori fid delta acc : -0.20952380952380956 0.4808344975757844 0.39031292517006794 0.27375057991741103

edit_dis -> new fid plus prob: -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid plus acc : -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus prob: 0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  syn4
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : 0.8285714285714288 1.1102230246251565e-16 0.041562682215743336 6.938893903907228e-18
edit_dis -> ori fid minus prob: 0.45714285714285713 0.02857142857142858 0.3626122448979591 0.03388921282798826
edit_dis -> ori fid minus acc : -0.01904761904761905 0.10647942749999 0.9014266277939748 0.12461135286855973
edit_dis -> ori fid delta prob: -0.780952380952381 0.021295885499998047 0.06725753158406217 0.011491085971835688
edit_dis -> ori fid delta acc : 0.7523809523809524 0.07126966450997985 0.09004470359572388 0.048425791675959624

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  ba2
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : nan nan nan nan
edit_dis -> ori fid minus prob: 0.8190476190476191 0.17274625854492434 0.08220019436345961 0.09242213525095588
edit_dis -> ori fid minus acc : nan nan nan nan
edit_dis -> ori fid delta prob: -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> ori fid delta acc : -0.7602661323901128 0.29848803667093826 0.16221991579680564 0.2913268482061816

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  mutag
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -0.3904761904761905 0.7571278824514969 0.0877434402332361 0.12109461067104471
edit_dis -> ori fid minus prob: -0.27619047619047626 0.8428436910414319 0.06765403304178814 0.11962494229798554
edit_dis -> ori fid minus acc : -0.3428571428571429 0.7936825390476444 0.09025461613216712 0.1306844432321529
edit_dis -> ori fid delta prob: -0.10476190476190476 0.86577662303722 0.09921088435374148 0.1994803005272675
edit_dis -> ori fid delta acc : 0.20952380952380958 0.790705646559536 0.12519339164237123 0.2239276349412722

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0

./redata/gin
dataset name  syn1
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -0.9976017934348604 0.005362552903738458 5.1514277974944985e-05 0.00011518942736379717
edit_dis -> ori fid minus prob: 0.8285714285714286 0.23790468565327225 0.1049096209912536 0.17216720716734685
edit_dis -> ori fid minus acc : 0.833889497614925 0.276001676109597 0.11745681969995041 0.2423598129939501
edit_dis -> ori fid delta prob: -0.9238095238095237 0.09134917187262319 0.019793974732750224 0.027811826516374906
edit_dis -> ori fid delta acc : -0.9333333333333335 0.07678340712665278 0.014655004859086456 0.01910228129707461

edit_dis -> new fid plus prob: 0.5904761904761905 0.5395051062197541 0.07849173955296394 0.07585297055893961
edit_dis -> new fid plus acc : 0.6181629504356428 0.3009947666824669 0.265699400402234 0.31495729239541187
edit_dis -> new fid minus prob: -0.2571428571428572 0.8121525944886797 0.042480077745383864 0.030568836392670815
edit_dis -> new fid minus acc : -0.12380952380952386 0.7678340712665286 0.12806219630709423 0.12686312311200126
edit_dis -> new fid delta prob: 0.16190476190476194 0.7657048092282583 0.18634791059280853 0.2840158713619217
edit_dis -> new fid delta acc : 0.08571428571428573 0.7890405781793517 0.19399805636540327 0.34356624440580796
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn2
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -0.9877891181205806 0.021520555593251108 0.0009011043721185243 0.001880093934877174
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 0.9952035868697208 0.006783152499585279 0.00010302855594988997 0.00014570438113604964
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -0.05714285714285716 0.6125668087956572 0.3646569484936831 0.34465306955479075
edit_dis -> new fid plus acc : -0.3904761904761906 0.5367239163278237 0.3184761904761904 0.2905988078222534
edit_dis -> new fid minus prob: 0.4952380952380952 0.4112918644131273 0.40183479105928077 0.3535653449214335
edit_dis -> new fid minus acc : 0.6000000000000001 0.329914439536929 0.30232069970845477 0.3145100155007163
edit_dis -> new fid delta prob: -0.5142857142857143 0.6767268161329721 0.08153935860058303 0.14190518353520992
edit_dis -> new fid delta acc : -0.6190476190476191 0.4856209060564557 0.12054421768707475 0.14158845895261268
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn3
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : nan nan nan nan
edit_dis -> ori fid minus prob: -0.22857142857142865 0.803224792960981 0.20461030126336252 0.3501365186139897
edit_dis -> ori fid minus acc : nan nan nan nan
edit_dis -> ori fid delta prob: -0.28571428571428575 0.6213875191694729 0.2838872691933915 0.2928442932575558
edit_dis -> ori fid delta acc : -0.7139146538787001 0.19962383750711093 0.15357111758006484 0.14621960290121627

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn4
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : nan nan nan nan
edit_dis -> ori fid minus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : nan nan nan nan
edit_dis -> ori fid delta prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : nan nan nan nan

edit_dis -> new fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : nan nan nan nan
edit_dis -> new fid minus prob: -0.9928053803045812 0.007194619695418847 0.00015454283392483495 0.00015454283392483492
edit_dis -> new fid minus acc : nan nan nan nan
edit_dis -> new fid delta prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : nan nan nan nan
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  ba2
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -0.952457684384857 0.053780913595071034 0.007386785687192139 0.012146538525739258
edit_dis -> ori fid minus prob: 0.9047619047619048 0.09712418121129114 0.025733722060252668 0.03305381910962917
edit_dis -> ori fid minus acc : 0.9880089671743019 0.005362552903738458 0.0002575713898747249 0.00011518942736379718
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
Traceback (most recent call last):
  File "/aul/homes/xzhen019/data/code/XGNN/forICLR24/visulization2_correlation .py", line 478, in <module>
    test_correction(dataname,dir)
  File "/aul/homes/xzhen019/data/code/XGNN/forICLR24/visulization2_correlation .py", line 284, in test_correction
    new_matrix = np.load(newfid_path, allow_pickle=True).item()
  File "/aul/homes/xzhen019/anaconda3/envs/repeg/lib/python3.7/site-packages/numpy/lib/npyio.py", line 417, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
TypeError: expected str, bytes or os.PathLike object, not NoneType

Process finished with exit code 1


"""


"""
/aul/homes/xzhen019/anaconda3/envs/repeg/bin/python /aul/homes/xzhen019/data/code/XGNN/forICLR24/visulization2_correlation .py 
./redata/gcn
dataset name  syn1
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 0.8666666666666667 0.1885618083164127 0.06933333333333327 0.0980521403245345
edit_dis -> ori fid minus acc : 0.7047619047619048 0.42559815183817773 0.26332361516034986 0.38470176261679867
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -0.942857142857143 0.08081220356417679 0.013854227405247782 0.019592836292702424

edit_dis -> new fid plus prob: -0.5523809523809525 0.5996975806259492 0.06266277939747324 0.04381120078743788
edit_dis -> new fid plus acc : nan nan nan nan
edit_dis -> new fid minus prob: 0.7333333333333334 0.1735320681741771 0.12957045675413012 0.1291739793944295
edit_dis -> new fid minus acc : nan nan nan nan
/aul/homes/xzhen019/anaconda3/envs/repeg/lib/python3.7/site-packages/scipy/stats/stats.py:4484: SpearmanRConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(SpearmanRConstantInputWarning())
edit_dis -> new fid delta prob: -0.561904761904762 0.6062411604891257 0.06186200194363456 0.04489198686398153
edit_dis -> new fid delta acc : nan nan nan nan
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  syn2
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -0.7428571428571429 0.32784599005825726 0.1824606413994169 0.3473087977874299
edit_dis -> new fid plus acc : -0.5361728405140985 0.6694512257053866 0.061399957888040424 0.0760595444836088
edit_dis -> new fid minus prob: 0.9047619047619048 0.042591770999995976 0.015611273080660805 0.013195606535599745
edit_dis -> new fid minus acc : 0.7904761904761907 0.11741740958036145 0.07779203109815346 0.06191431115733338
edit_dis -> new fid delta prob: -0.9238095238095237 0.07126966450997985 0.01526919339164236 0.02560827676986896
edit_dis -> new fid delta acc : -0.8857142857142858 0.08728715609439695 0.02887463556851311 0.03130857479334082
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  syn3
edit_dis -> ori fid plus prob: 0.22857142857142865 0.6352459484058017 0.25173955296404266 0.2729078481657217
edit_dis -> ori fid plus acc : 0.19047619047619047 0.5231164245660747 0.34587366375121475 0.2700840242975926
edit_dis -> ori fid minus prob: -0.20952380952380956 0.6341026476844689 0.2740991253644314 0.32247826254422396
edit_dis -> ori fid minus acc : 0.12380952380952383 0.3124041803210805 0.5972478134110787 0.23594772967835378
edit_dis -> ori fid delta prob: 0.10476190476190476 0.5653646506535555 0.3217414965986393 0.29418189453738003
edit_dis -> ori fid delta acc : -0.20952380952380956 0.4808344975757844 0.39031292517006794 0.27375057991741103

edit_dis -> new fid plus prob: -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid plus acc : -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus prob: 0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  syn4
edit_dis -> ori fid plus prob: 0.09523809523809525 0.021295885499998 0.857648202137998 0.03151791053999706
edit_dis -> ori fid plus acc : 0.8285714285714288 1.1102230246251565e-16 0.041562682215743336 6.938893903907228e-18
edit_dis -> ori fid minus prob: 0.45714285714285713 0.02857142857142858 0.3626122448979591 0.03388921282798826
edit_dis -> ori fid minus acc : -0.01904761904761905 0.10647942749999 0.9014266277939748 0.12461135286855973
edit_dis -> ori fid delta prob: -0.780952380952381 0.021295885499998047 0.06725753158406217 0.011491085971835688
edit_dis -> ori fid delta acc : 0.7523809523809524 0.07126966450997985 0.09004470359572388 0.048425791675959624

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  ba2
edit_dis -> ori fid plus prob: -0.9238095238095237 0.063173805530579 0.014009718172983453 0.014624251823187004
edit_dis -> ori fid plus acc : nan nan nan nan
edit_dis -> ori fid minus prob: 0.8190476190476191 0.17274625854492434 0.08220019436345961 0.09242213525095588
edit_dis -> ori fid minus acc : nan nan nan nan
edit_dis -> ori fid delta prob: -0.9904761904761905 0.02129588549999796 0.0008007774538386759 0.0017905928216324792
edit_dis -> ori fid delta acc : -0.7602661323901128 0.29848803667093826 0.16221991579680564 0.2913268482061816

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gcn
dataset name  mutag
edit_dis -> ori fid plus prob: -0.19047619047619047 0.8413896290878013 0.16805442176870747 0.3531913238011196
edit_dis -> ori fid plus acc : -0.3904761904761905 0.7571278824514969 0.0877434402332361 0.12109461067104471
edit_dis -> ori fid minus prob: -0.27619047619047626 0.8428436910414319 0.06765403304178814 0.11962494229798554
edit_dis -> ori fid minus acc : -0.3428571428571429 0.7936825390476444 0.09025461613216712 0.1306844432321529
edit_dis -> ori fid delta prob: -0.10476190476190476 0.86577662303722 0.09921088435374148 0.1994803005272675
edit_dis -> ori fid delta acc : 0.20952380952380958 0.790705646559536 0.12519339164237123 0.2239276349412722

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn1
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -0.9976017934348604 0.005362552903738458 5.1514277974944985e-05 0.00011518942736379717
edit_dis -> ori fid minus prob: 0.8285714285714286 0.23790468565327225 0.1049096209912536 0.17216720716734685
edit_dis -> ori fid minus acc : 0.833889497614925 0.276001676109597 0.11745681969995041 0.2423598129939501
edit_dis -> ori fid delta prob: -0.9238095238095237 0.09134917187262319 0.019793974732750224 0.027811826516374906
edit_dis -> ori fid delta acc : -0.9333333333333335 0.07678340712665278 0.014655004859086456 0.01910228129707461

edit_dis -> new fid plus prob: 0.5904761904761905 0.5395051062197541 0.07849173955296394 0.07585297055893961
edit_dis -> new fid plus acc : 0.6181629504356428 0.3009947666824669 0.265699400402234 0.31495729239541187
edit_dis -> new fid minus prob: -0.2571428571428572 0.8121525944886797 0.042480077745383864 0.030568836392670815
edit_dis -> new fid minus acc : -0.12380952380952386 0.7678340712665286 0.12806219630709423 0.12686312311200126
edit_dis -> new fid delta prob: 0.16190476190476194 0.7657048092282583 0.18634791059280853 0.2840158713619217
edit_dis -> new fid delta acc : 0.08571428571428573 0.7890405781793517 0.19399805636540327 0.34356624440580796
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn2
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -0.9877891181205806 0.021520555593251108 0.0009011043721185243 0.001880093934877174
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 0.9952035868697208 0.006783152499585279 0.00010302855594988997 0.00014570438113604964
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -0.05714285714285716 0.6125668087956572 0.3646569484936831 0.34465306955479075
edit_dis -> new fid plus acc : -0.3904761904761906 0.5367239163278237 0.3184761904761904 0.2905988078222534
edit_dis -> new fid minus prob: 0.4952380952380952 0.4112918644131273 0.40183479105928077 0.3535653449214335
edit_dis -> new fid minus acc : 0.6000000000000001 0.329914439536929 0.30232069970845477 0.3145100155007163
edit_dis -> new fid delta prob: -0.5142857142857143 0.6767268161329721 0.08153935860058303 0.14190518353520992
edit_dis -> new fid delta acc : -0.6190476190476191 0.4856209060564557 0.12054421768707475 0.14158845895261268
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn3
edit_dis -> ori fid plus prob: 0.20000000000000004 0.8228042311030543 0.15018853255587947 0.2861435869315297
edit_dis -> ori fid plus acc : nan nan nan nan
edit_dis -> ori fid minus prob: -0.22857142857142865 0.803224792960981 0.20461030126336252 0.3501365186139897
edit_dis -> ori fid minus acc : nan nan nan nan
edit_dis -> ori fid delta prob: -0.28571428571428575 0.6213875191694729 0.2838872691933915 0.2928442932575558
edit_dis -> ori fid delta acc : -0.7139146538787001 0.19962383750711093 0.15357111758006484 0.14621960290121627

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  syn4
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : nan nan nan nan
edit_dis -> ori fid minus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : nan nan nan nan
edit_dis -> ori fid delta prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : nan nan nan nan

edit_dis -> new fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : nan nan nan nan
edit_dis -> new fid minus prob: -0.9928053803045812 0.007194619695418847 0.00015454283392483495 0.00015454283392483492
edit_dis -> new fid minus acc : nan nan nan nan
edit_dis -> new fid delta prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : nan nan nan nan
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  ba2
edit_dis -> ori fid plus prob: -0.8380952380952382 0.1963383631246505 0.08073858114674436 0.14234880456956145
edit_dis -> ori fid plus acc : -0.952457684384857 0.053780913595071034 0.007386785687192139 0.012146538525739258
edit_dis -> ori fid minus prob: 0.9047619047619048 0.09712418121129114 0.025733722060252668 0.03305381910962917
edit_dis -> ori fid minus acc : 0.9880089671743019 0.005362552903738458 0.0002575713898747249 0.00011518942736379718
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata/gin
dataset name  mutag
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 0.8857142857142857 0.15118578920369088 0.04753352769679297 0.07634540009567099
edit_dis -> ori fid minus acc : 0.8476190476190476 0.1766403522951563 0.07093488824101063 0.09693951669951505
edit_dis -> ori fid delta prob: -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792
edit_dis -> ori fid delta acc : -0.9904761904761905 0.021295885499997964 0.0008007774538386759 0.0017905928216324792

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0

Process finished with exit code 0

"""


"""
/aul/homes/xzhen019/anaconda3/envs/repeg/bin/python /aul/homes/xzhen019/data/code/XGNN/forICLR24/visulization2_correlation .py 
./redata_0922/gcn
dataset name  syn1
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 0.6 0.0 0.20799999999999982 0.0
edit_dis -> ori fid minus acc : -0.028571428571428574 0.0 0.9571545189504373 0.0
edit_dis -> ori fid delta prob: -0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> ori fid delta acc : -0.942857142857143 0.0 0.004804664723032055 0.0

edit_dis -> new fid plus prob: 0.7714285714285715 0.0 0.07239650145772594 0.0
edit_dis -> new fid plus acc : nan nan nan nan
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gcn
dataset name  syn2
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -0.3714285714285715 0.0 0.46847813411078715 0.0
edit_dis -> new fid plus acc : 0.3086066999241839 0.0 0.5517855072529712 0.0
edit_dis -> new fid minus prob: -0.028571428571428574 0.0 0.9571545189504373 0.0
edit_dis -> new fid minus acc : -0.3142857142857143 0.0 0.5440932944606414 0.0
edit_dis -> new fid delta prob: 0.028571428571428574 0.0 0.9571545189504373 0.0
edit_dis -> new fid delta acc : 0.3142857142857143 0.0 0.5440932944606414 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gcn
/aul/homes/xzhen019/anaconda3/envs/repeg/lib/python3.7/site-packages/scipy/stats/stats.py:4484: SpearmanRConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(SpearmanRConstantInputWarning())
dataset name  syn3
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: 0.48571428571428577 0.0 0.3287230320699709 0.0
edit_dis -> new fid plus acc : 0.48571428571428577 0.0 0.3287230320699709 0.0
edit_dis -> new fid minus prob: -0.3714285714285715 0.0 0.46847813411078715 0.0
edit_dis -> new fid minus acc : 0.3714285714285715 0.0 0.46847813411078715 0.0
edit_dis -> new fid delta prob: 0.7142857142857143 0.0 0.1107871720116617 0.0
edit_dis -> new fid delta acc : -0.08571428571428573 0.0 0.8717434402332361 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gcn
dataset name  syn4
edit_dis -> ori fid plus prob: -0.5428571428571429 0.0 0.26570262390670557 0.0
edit_dis -> ori fid plus acc : 0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> ori fid minus prob: 0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> ori fid minus acc : 0.028571428571428574 0.0 0.9571545189504373 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : 0.5428571428571429 0.0 0.26570262390670557 0.0

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gcn
dataset name  ba2
edit_dis -> ori fid plus prob: 0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> ori fid plus acc : 0.5408987230262506 0.0 0.26777767151474974 0.0
edit_dis -> ori fid minus prob: 0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> ori fid minus acc : 0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> ori fid delta prob: -0.48571428571428577 0.0 0.3287230320699709 0.0
edit_dis -> ori fid delta acc : 0.13093073414159545 0.0 0.8047261622231061 0.0

edit_dis -> new fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : 1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gin
dataset name  syn1
edit_dis -> ori fid plus prob: -0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> ori fid plus acc : -0.9856107606091623 0.0 0.0003090856678496699 0.0
edit_dis -> ori fid minus prob: 0.4285714285714286 0.0 0.3965014577259474 0.0
edit_dis -> ori fid minus acc : 0.14285714285714288 0.0 0.7871720116618075 0.0
edit_dis -> ori fid delta prob: -0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> ori fid delta acc : -0.8285714285714287 0.0 0.04156268221574334 0.0

edit_dis -> new fid plus prob: -0.8857142857142858 0.0 0.01884548104956266 0.0
edit_dis -> new fid plus acc : -0.6571428571428573 0.0 0.15617492711370237 0.0
edit_dis -> new fid minus prob: 0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> new fid minus acc : 0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> new fid delta prob: -0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> new fid delta acc : -0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gin
dataset name  syn2
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: 0.3714285714285715 0.0 0.46847813411078715 0.0
edit_dis -> new fid plus acc : 0.48571428571428577 0.0 0.3287230320699709 0.0
edit_dis -> new fid minus prob: 0.028571428571428574 0.0 0.9571545189504373 0.0
edit_dis -> new fid minus acc : 0.08571428571428573 0.0 0.8717434402332361 0.0
edit_dis -> new fid delta prob: 0.2 0.0 0.704 0.0
edit_dis -> new fid delta acc : -0.08571428571428573 0.0 0.8717434402332361 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gin
dataset name  syn3
edit_dis -> ori fid plus prob: -0.2 0.0 0.704 0.0
edit_dis -> ori fid plus acc : -0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> ori fid minus prob: -0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> ori fid minus acc : -0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> ori fid delta prob: 0.6 0.0 0.20799999999999982 0.0
edit_dis -> ori fid delta acc : nan nan nan nan

edit_dis -> new fid plus prob: 0.48571428571428577 0.0 0.3287230320699709 0.0
edit_dis -> new fid plus acc : 0.4285714285714286 0.0 0.3965014577259474 0.0
edit_dis -> new fid minus prob: -0.5428571428571429 0.0 0.26570262390670557 0.0
edit_dis -> new fid minus acc : 0.3142857142857143 0.0 0.5440932944606414 0.0
edit_dis -> new fid delta prob: 0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> new fid delta acc : 0.3142857142857143 0.0 0.5440932944606414 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gin
dataset name  syn4
edit_dis -> ori fid plus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : 0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> ori fid minus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> ori fid delta prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : nan nan nan nan

edit_dis -> new fid plus prob: 0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> new fid plus acc : 0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> new fid minus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 0.6546536707079772 0.0 0.15830242337545783 0.0
edit_dis -> new fid delta prob: 0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> new fid delta acc : nan nan nan nan
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gin
dataset name  ba2
edit_dis -> ori fid plus prob: 0.6 0.0 0.20799999999999982 0.0
edit_dis -> ori fid plus acc : -0.2898855178262242 0.0 0.5773517870348748 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : -0.6982532518267538 0.0 0.12283946337339596 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : 0.20291986247835697 0.0 0.6997979681570189 0.0

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0


./redata_0922/gcn
dataset name  mutag
edit_dis -> ori fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: -0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> ori fid minus acc : -0.8285714285714287 0.0 0.04156268221574334 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0
./redata_0922/gin
dataset name  mutag
edit_dis -> ori fid plus prob: -0.942857142857143 0.0 0.004804664723032055 0.0
edit_dis -> ori fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> ori fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> ori fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> ori fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> ori fid delta acc : -1.0 0.0 0.0 0.0

edit_dis -> new fid plus prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid plus acc : -1.0 0.0 0.0 0.0
edit_dis -> new fid minus prob: 1.0 0.0 0.0 0.0
edit_dis -> new fid minus acc : 1.0 0.0 0.0 0.0
edit_dis -> new fid delta prob: -1.0 0.0 0.0 0.0
edit_dis -> new fid delta acc : -1.0 0.0 0.0 0.0
edit_dis -> auc : -1.0 0.0 0.0 0.0
edit_dis -> iou : -1.0 0.0 0.0 0.0

"""

import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame
from scipy.sparse import coo_matrix
import math


from scipy.stats import spearmanr
from matplotlib.pyplot import MultipleLocator

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


def visiual_hypterparameter(dict,path_save):
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
               '0.9',1.0]
    parameter = [0.1,0.3,0.5,0.7,0.9]

    x_axis_dict = {0.1:1, 0.3:2, 0.5:3, 0.7:4, 0.9:5}
    y_axis_dict = {('0.0','0.0'):0, ('0.1','0.0'):1, ('0.3','0.0'):2, ('0.5','0.0'):3, ('0.7','0.0'):4, ('0.9','0.0'):5,
                   ('0.0','0.1'):0+6*1, ('0.1','0.1'):1+6*1, ('0.3','0.1'):2+6*1, ('0.5','0.1'):3+6*1, ('0.7','0.1'):4+6*1, ('0.9','0.1'):5+6*1,
                   ('0.0','0.3'):0+6*2, ('0.1','0.3'):1+6*2, ('0.3','0.3'):2+6*2, ('0.5','0.3'):3+6*2, ('0.7','0.3'):4+6*2, ('0.9','0.3'):5+6*2,
                   ('0.0','0.5'):0+6*3, ('0.1','0.5'):1+6*3, ('0.3','0.5'):2+6*3, ('0.5','0.5'):3+6*3, ('0.7','0.5'):4+6*3, ('0.9','0.5'):5+6*3,
                   ('0.0','0.7'):0+6*4, ('0.1','0.7'):1+6*4, ('0.3','0.7'):2+6*4, ('0.5','0.7'):3+6*4, ('0.7','0.7'):4+6*4, ('0.9','0.7'):5+6*4,
                   ('0.0','0.9'):0+6*5, ('0.1','0.9'):1+6*5, ('0.3','0.9'):2+6*5, ('0.5','0.9'):3+6*5, ('0.7','0.9'):4+6*5, ('0.9','0.9'):5+6*5,}
    y_names = [('0.0','0.0'),('0.1','0.0'),('0.3','0.0'),('0.5','0.0'),('0.7','0.0'),('0.9','0.0')]
    y_idxs = [0,1,2,3,4,5]

    plus_x_lists = []
    plus_y_lists = []
    plus_z_lists = []
    plus_collection_xz = {}

    minus_x_lists = []
    minus_y_lists = []
    minus_z_lists = []
    minus_collection_xz = {}

    # fid_plus_matrix_dict = {}
    # fid_minus_matrix_dict = {}

    for para in parameter:
        data_dict = dict[para]
        for value_count in [0,4]:
            # matrix = np.zeros([len(x_ticks), len(x_ticks)])
            for key in data_dict.keys():
                data = data_dict[key]
                remove_axis = key[0] # remove
                add_axis = key[1]   #  add
                if remove_axis in x_ticks and add_axis in x_ticks:
                    value = data[value_count]
                    # matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value
                    if value_count == 0:
                        plus_x_lists.append(x_axis_dict[para])
                        plus_y_lists.append(y_axis_dict[key])
                        plus_z_lists.append(value)
                        if y_axis_dict[key] in plus_collection_xz.keys():
                            plus_collection_xz[y_axis_dict[key]].append((x_axis_dict[para],value))
                        else:
                            plus_collection_xz[y_axis_dict[key]]=[(x_axis_dict[para], value)]
                        # fid_plus_matrix_dict[para] = matrix
                    elif value_count == 4:
                        minus_x_lists.append(x_axis_dict[para])
                        minus_y_lists.append(y_axis_dict[key])
                        minus_z_lists.append(value)
                        if y_axis_dict[key] in minus_collection_xz.keys():
                            minus_collection_xz[y_axis_dict[key]].append((x_axis_dict[para],value))
                        else:
                            minus_collection_xz[y_axis_dict[key]]=[(x_axis_dict[para], value)]
                        # fid_minus_matrix_dict[para] = matrix

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.collections import PolyCollection,LineCollection
    from mpl_toolkits.mplot3d import Axes3D
    # visual
    fig = plt.figure(dpi=120)
    ax = fig.gca(projection='3d')
    precipitation = [plus_collection_xz[0],plus_collection_xz[1],plus_collection_xz[2],plus_collection_xz[3],
                     plus_collection_xz[4],plus_collection_xz[5]]
    poly = LineCollection(precipitation, facecolors=['b', 'c', 'r', 'm','g','k'])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=y_idxs, zdir='y')
    ax.set_xlabel('alpha')
    ax.set_xticks([0,1,2,3,4,5,6],x_ticks)
    ax.set_xlim3d(0, len(parameter))

    ax.set_ylabel('edit dis')
    ax.set_yticks([0,1,2,3,4,5],y_names)
    ax.set_ylim3d(0, len(y_names))
    ax.set_zlabel('$\hat{FId}_+$')
    ax.set_zlim3d(min(plus_z_lists)*(1-0.1), max(plus_z_lists)*(1+0.1))

    ax.view_init(elev=25, azim=135)

    plt.savefig(path_save + '_plus.png')
    plt.savefig(path_save + '_plus.pdf')
    plt.close()
    # plt.show()

    fig = plt.figure(dpi=120)
    ax = fig.gca(projection='3d')
    precipitation = [minus_collection_xz[0],minus_collection_xz[1],minus_collection_xz[2],minus_collection_xz[3],
                     minus_collection_xz[4],minus_collection_xz[5]]
    poly = LineCollection(precipitation, facecolors=['b', 'c', 'r', 'm','g','k'])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=y_idxs, zdir='y')
    ax.set_xlabel('alpha')
    ax.set_xticks([0,1,2,3,4,5,6],x_ticks)
    ax.set_xlim3d(0, len(parameter))

    ax.set_ylabel('edit dis')
    ax.set_yticks([0,1,2,3,4,5],y_names)
    ax.set_ylim3d(0, len(y_names))
    ax.set_zlabel('$\hat{FId}_+$')
    ax.set_zlim3d(min(plus_z_lists)*(1-0.1), max(plus_z_lists)*(1+0.1))

    ax.view_init(elev=25, azim=135)

    plt.savefig(path_save + '_minus.png')
    plt.savefig(path_save + '_minus.pdf')
    plt.close()



def test_vis():
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    np.random.seed(59)
    month = np.arange(0, 13)
    years = [2016, 2017, 2018, 2019]

    precipitation = []
    for year in years:
        value = np.random.rand(len(month)) * 300
        value[0], value[-1] = 0, 0
        precipitation.append(list(zip(month, value)))

    poly = PolyCollection(precipitation, facecolors=['b', 'c', 'r', 'm'])
    poly.set_alpha(0.7)

    ax.add_collection3d(poly, zs=years, zdir='y')
    ax.set_xlabel('Month')
    ax.set_xlim3d(0, 12)
    ax.set_ylabel('Year')
    ax.set_ylim3d(2015, 2020)
    ax.set_zlabel('Precipitation')
    ax.set_zlim3d(0, 300)

    plt.show()


def test_correction_sp(name,dict):
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



    new_matrix = dict #np.load(newfid_path, allow_pickle=True).item()

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

    for key in new_matrix.keys():
        remove_axis = key[0]  # remove
        add_axis = key[1]  # add
        if remove_axis in x_ticks and add_axis in x_ticks:
            new_data = new_matrix[key]

            edit_distnce = exp_edges_dict[name]*float(remove_axis) + non_exp_edges_dict[name]*float(add_axis)
            editdistance_matrix[dict_ratio[remove_axis], dict_ratio[add_axis]] = edit_distnce

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
        return statisticfloat_list, pvaluefloat_list

    results = []

    print()
    statisticfloat_list, pvaluefloat_list = cal(new_fidplus_matrix_prob)
    print("edit_dis -> new fid plus prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())

    statisticfloat_list, pvaluefloat_list = cal(new_fidplus_matrix_acc)
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ), b=new_fidplus_matrix_acc.reshape(-1, ))
    print("edit_dis -> new fid plus acc :",  np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())

    statisticfloat_list, pvaluefloat_list = cal(new_fidminus_matrix_prob)
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidminus_matrix_prob.reshape(-1, ))
    print("edit_dis -> new fid minus prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fidminus_matrix_acc.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fidminus_matrix_acc)
    print("edit_dis -> new fid minus acc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())

    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fiddelta_matrix_prob.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fiddelta_matrix_prob)
    print("edit_dis -> new fid delta prob:", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=new_fiddelta_matrix_acc.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(new_fiddelta_matrix_acc)
    print("edit_dis -> new fid delta acc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())
    #################
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=auc_matrix.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(auc_matrix)
    print("edit_dis -> auc :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())
    # statisticfloat, pvaluefloat = spearmanr(a=editdistance_matrix.reshape(-1, ),
    #                                         b=iou_matrix.reshape(-1, ))
    statisticfloat_list, pvaluefloat_list = cal(iou_matrix)
    print("edit_dis -> iou :", np.array(statisticfloat_list).mean(),np.array(statisticfloat_list).std(),
                                                np.array(pvaluefloat_list).mean(),np.array(pvaluefloat_list).std())
    results.append(np.array(statisticfloat_list).mean())
    results.append(np.array(pvaluefloat_list).mean())

    return results

def vis_2dfig_his(dict,path_save):

    axis_value=[]
    alpha1_value=[] # plus prob
    alpha2_value=[] # minus prob

    for key in [0.1,0.3,0.5,0.7,0.9]:
        key = str(key)
        x = float(key)
        key_alpha2 = '%.1f'%(1 - x)
        axis_value.append(x)
        alpha1_value.append(dict[key][0])
        alpha2_value.append(dict[key_alpha2][4])

    # plot

    SMALL_SIZE = 8
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

    fig = plt.figure(dpi=120, figsize=(9, 6))
    plt.subplots_adjust(left=0.15, right=0.85)
    ax1 = fig.add_subplot(111)
    ax1.plot(axis_value, alpha1_value, 'g', marker='o',label='$\hat{Fid}_{+}$')
    ax1.set_ylabel('$\hat{Fid}_{+}$')  # Spearman correlation coefficient  of
    ax1.yaxis.label.set_color('g')
    ax1.spines['left'].set_color('g')
    ax1.spines['left'].set_linewidth(1)
    ax1.tick_params(colors ='g',axis='y')
    ax1.legend(loc='lower right')
    new_values = alpha1_value
    length = max(new_values) - min(new_values)
    interval = length / 4
    if interval > 1:
        interval = int(interval)
    else:
        interval = float("%.2f" % interval)
    y_major_locator = MultipleLocator(interval)
    ax1.yaxis.set_major_locator(y_major_locator)

    ax2 = ax1.twinx()
    ax2.plot(axis_value, alpha2_value, 'r',marker='v',label='$\hat{Fid}_{-}$')
    ax2.set_ylabel('$\hat{Fid}_{-}$')  # Spearman correlation coefficient of
    ax2.legend(loc='upper left')
    ax2.yaxis.label.set_color('r')
    ax2.spines['right'].set_color('r')
    ax2.spines['right'].set_linewidth(1)
    ax2.tick_params(colors ='r',axis='y')

    new_values = alpha2_value
    length = max(new_values) - min(new_values)
    interval = length / 4
    if interval > 1:
        interval = int(interval)
    else:
        interval = float("%.2f" % interval)
    y_major_locator = MultipleLocator(interval)
    ax2.yaxis.set_major_locator(y_major_locator)


    plt.savefig(path_save+'.png')
    plt.savefig(path_save+'.pdf')

def vis_2dfig(dict,path_save):
    axis_value = []
    alpha1_value = []  # plus prob
    alpha2_value = []  # minus prob

    for key in [0.1, 0.3, 0.5, 0.7, 0.9]:
        key = str(key)
        x = float(key)
        key_alpha2 = '%.1f' % (1 - x)
        axis_value.append(x)
        alpha1_value.append(dict[key][0])
        alpha2_value.append(dict[key_alpha2][4])

    # plot

    SMALL_SIZE = 10
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 26

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

    fig = plt.figure(dpi=120, figsize=(9, 7))
    plt.subplots_adjust(left=0.17, right=0.82)
    ax1 = fig.add_subplot(111)
    ax1.plot(axis_value, alpha1_value, 'blue', marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')
    ax1.set_ylabel('$Fid_{\\alpha_1,+}$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of
    ax1.set_xlabel('$\\alpha_1, \\alpha_2$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

    ax1.yaxis.label.set_color('b')
    ax1.spines['left'].set_color('b')
    ax1.spines['left'].set_linewidth(1)
    ax1.tick_params(colors='b', axis='y')
    ax1.legend(loc='lower right')
    new_values = alpha1_value
    length = max(new_values) - min(new_values)
    interval = length / 4
    if interval > 1:
        interval = int(interval)
    else:
        interval = float("%.2f" % interval)
    y_major_locator = MultipleLocator(interval)
    ax1.yaxis.set_major_locator(y_major_locator)

    ax2 = ax1.twinx()
    ax2.plot(axis_value, alpha2_value, 'orange', marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$', linewidth='3')
    ax2.set_ylabel('$Fid_{\\alpha_2,-}$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
    ax2.legend(loc='upper left')
    ax2.yaxis.label.set_color('orange')
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(1)
    ax2.tick_params(colors='orange', axis='y')

    new_values = alpha2_value
    length = max(new_values) - min(new_values)
    interval = length / 4
    if interval > 1:
        interval = int(interval)
    else:
        interval = float("%.2f" % interval)
    y_major_locator = MultipleLocator(interval)
    ax2.yaxis.set_major_locator(y_major_locator)


    plt.savefig(path_save+'.png')
    plt.savefig(path_save+'.pdf')

def vis_time_std():

    def vis(axis_x,fid_plus,fid_minus,plus_std,minus_std,save_name='syn3'):
        SMALL_SIZE = 8
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 18

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.11,right=0.88)
        ax1 = fig.add_subplot(111)

        ax1.plot(axis_x, fid_plus, 'g', marker='o', label='$\hat{Fid}_{+}$')
        ax1.plot(axis_x, fid_minus, 'g',marker='v', label='$\hat{Fid}_{-}$')
        ax1.set_ylabel('time(s)')  # Spearman correlation coefficient  of
        ax1.yaxis.label.set_color('g')
        ax1.spines['left'].set_color('g')
        ax1.spines['left'].set_linewidth(1)
        ax1.tick_params(colors ='g',axis='y')
        # ax1.tick_params(axis='y', pad=100)
        ax1.legend(loc='lower right')
        new_values = fid_plus
        new_values.extend(plus_std)
        length = max(new_values)-min(new_values)
        interval = length/4
        if interval >1:
            interval = int(interval)
        else:
            interval = float("%.2f"%interval)
        y_major_locator = MultipleLocator(interval)
        ax1.yaxis.set_major_locator(y_major_locator)

        ax2 = ax1.twinx()
        ax2.plot(axis_x, plus_std, 'r',  marker='s',label='$\hat{Fid}_{+}$')
        ax2.plot(axis_x, minus_std, 'r', marker='d', label='$\hat{Fid}_{-}$')
        ax2.set_ylabel('std')  # Spearman correlation coefficient of
        ax2.yaxis.label.set_color('r')
        ax1.spines['right'].set_color('r')
        ax1.spines['right'].set_linewidth(1)
        ax2.tick_params(colors ='r',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend(loc='upper left')

        new_values = plus_std
        new_values.extend(minus_std)
        length = max(new_values)-min(new_values)
        interval = length/4
        if interval >1:
            interval = int(interval)
        else:
            interval = float("%.2f"%interval)
        y_major_locator = MultipleLocator(interval)
        ax2.yaxis.set_major_locator(y_major_locator)

        # plt.show()
        plt.savefig('./redata/gcn_parameter/%s_M_parameter.png'%save_name)
        plt.savefig('./redata/gcn_parameter/%s_M_parameter.pdf'%save_name)
        pass

    # syn3:
    syn3_axis = [10,20,30,40,50,60,70,80,90,100]
    syn3_time_fid_plus = [1.4709138870239258,2.894583225250244,3.9697797298431396,
                     5.212178945541382,6.471220016479492,7.7203688621521,
                     8.961802005767822,10.22701096534729,11.450666904449463,12.717542171478271]
    syn3_time_fid_minus = [1.472731590270996,2.9043195247650146,3.975313186645508,
                      5.229603290557861,6.486292123794556,7.727260589599609,
                      8.969809770584106,10.227404832839966,11.481876850128174,12.733032464981079]

    syn3_fid_plus_std = [0.0043648593,0.14271583,0.1367638,
                    0.14418675,0.14415692,0.1427828,
                    0.14365564,0.14400621,0.14765604,0.14708538]
    syn3_fid_minus_std = [0.003274013,0.15924235,0.1556471,
                     0.15595903,0.16252126,0.15865299,
                     0.15319948,0.15972579,0.15782745,0.16241184]
    vis(syn3_axis,syn3_time_fid_plus,syn3_time_fid_minus,syn3_fid_plus_std,syn3_fid_minus_std,'syn3')


    #
    syn4_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    syn4_time_fid_plus = [7.178602457046509,13.5716712474823,19.315612077713013,
                          25.485674381256104,32.40154576301575,37.7967164516449,
                          43.034682273864746,51.05045413970947,58.32899069786072,62.11319375038147]
    syn4_time_fid_minus = [7.168165683746338,13.432164907455444,19.27128291130066,
                           25.393564224243164,32.22685670852661,37.67633581161499,
                           42.88003969192505,50.76883935928345,58.23271441459656,61.87667107582092]

    syn4_fid_plus_std = [0.4381617,0.44764876,0.44929212,
                         0.44406092,0.4461473,0.44728422,
                         0.444096,0.4446268,0.4455922,0.44307902]
    syn4_fid_minus_std = [0.22707708,0.23035426,0.2310515,
                          0.22784033,0.22738764,0.23042832,
                          0.22781378,0.23140906,0.23021238,0.22753055]
    vis(syn4_axis,syn4_time_fid_plus,syn4_time_fid_minus,syn4_fid_plus_std,syn4_fid_minus_std,'syn4')

    #
    ba2_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ba2_time_fid_plus = [5.301561594009399,9.892878770828247,14.746927499771118,
                         19.446733951568604,24.025258541107178,28.323596715927124,
                         32.94064140319824,37.50444197654724,43.709330797195435,46.712167501449585]
    ba2_time_fid_minus = [5.316497325897217,9.989225149154663,14.830771684646606,
                          19.549080848693848,24.159324169158936,28.483639240264893,
                          33.10951328277588,37.740357398986816,43.70541548728943,46.85391092300415]

    ba2_fid_plus_std = [0.12177958,0.11897518,0.120345384,
                        0.1204054,0.121377274,0.119027615,
                        0.120040715,0.12071913,0.117427625,0.11884906]
    ba2_fid_minus_std = [0.17206772,0.17115876,0.17157449,
                         0.17129011,0.1715138,0.17150944,
                         0.1713608,0.17096598,0.17120424,0.17116171]
    vis(ba2_axis,ba2_time_fid_plus,ba2_time_fid_minus,ba2_fid_plus_std,ba2_fid_minus_std,'ba2')

    #
    mutag_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mutag_time_fid_plus = [27.832050561904907,53.1815083026886,81.0308768749237,
                           116.51014828681946,128.16099643707275,154.30673551559448,
                           173.0985071659088,204.03238892555237,226.6609673500061,257.73919463157654]
    mutag_time_fid_minus = [27.711063146591187,53.301501989364624,81.30808401107788,
                            117.24894833564758,129.00218224525452,155.4777524471283,
                            174.49821209907532,206.2398443222046,228.74470114707947,260.22960925102234]

    mutag_fid_plus_std = [0.1128193,0.114834264,0.11424077,
                          0.116105445,0.1150566,0.116181545,
                          0.11661216,0.11669696,0.1158969,0.11491974]
    mutag_fid_minus_std = [0.16262963,0.16337402,0.16259556,
                           0.16526364, 0.16319717,0.16491275,
                           0.16448794,0.1627413,0.16332512,0.1644162]

    vis(mutag_axis,mutag_time_fid_plus,mutag_time_fid_minus,mutag_fid_plus_std,mutag_fid_minus_std,'mutag')

def vis_time_stdx(axises,fid_plus_std,fid_minus_std,name):

    def vis(axis_x,plus_std,minus_std,save_name='syn3'):

        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        ax2.plot(axis_x, plus_std, marker='o',markersize=16,label='$Fid_{\\alpha_1,+}$',linewidth='3')  #
        ax2.plot(axis_x, minus_std, marker='v',markersize=16, label='$Fid_{\\alpha_2,-}$',linewidth='3')  # $Fid{\\alpha_2,-}$
        ax2.set_ylabel('mean',fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('M', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend() # loc='upper left'

        new_values = plus_std
        new_values.extend(minus_std)
        length = max(new_values)-min(new_values)
        interval = length/4
        if interval >1:
            interval = int(interval)
        else:
            interval = float("%.2f"%interval)
        # y_major_locator = MultipleLocator(interval)
        # ax2.yaxis.set_major_locator(y_major_locator)

        # plt.show()
        plt.savefig('./redata/gcn_parameter/%s_M_parameter_mean.png'%save_name)
        plt.savefig('./redata/gcn_parameter/%s_M_parameter_mean.pdf'%save_name)


    vis(axises,fid_plus_std,fid_minus_std,name)

def vis_time_mean_stdx(axises,fid_plus_std,fid_minus_std,
                       fid_plus_mean,fid_minus_mean,name):

    def vis_plus(axis_x,plus_mean,plus_std,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        ax2.errorbar(axis_x, plus_mean, yerr=plus_std, fmt='-o', capsize=5, label="Mean ± 1 SD",linewidth=3)
        ax2.set_ylabel('Mean Value', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('Sample Size (M)', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        _min = min(plus_mean - max(plus_std)*1.1)
        _max = max(plus_mean + max(plus_std)*1.1)
        _range = _max - _min
        y_min = _min - _range
        ax2.set_ylim(y_min, _max)
        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # y_major_locator = MultipleLocator(interval)
        # ax2.yaxis.set_major_locator(y_major_locator)

        # plt.show()
        plt.savefig('./redata/gcn_parameter/%s_M_errorbar_plus.png' % save_name)
        plt.savefig('./redata/gcn_parameter/%s_M_errorbar_plus.pdf' % save_name)

    def vis_minus(axis_x, plus_mean, plus_std, save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        ax2.errorbar(axis_x, plus_mean, yerr=plus_std, fmt='-o', capsize=5, label="Mean ± 1 SD",linewidth=3)
        ax2.set_ylabel('Mean Value', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('Sample Size (M)', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # _min = min(fid_plus_mean - max(fid_plus_std) * 1.1)
        _min = min(plus_mean - max(plus_std)*1.1)
        _max = max(plus_mean + max(plus_std)*1.1)
        _range = _max - _min
        y_min = _min - _range
        ax2.set_ylim(y_min, _max)
        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # y_major_locator = MultipleLocator(interval)
        # ax2.yaxis.set_major_locator(y_major_locator)

        # plt.show()
        plt.savefig('./redata/gcn_parameter/%s_M_errorbar_minus.png' % save_name)
        plt.savefig('./redata/gcn_parameter/%s_M_errorbar_minus.pdf' % save_name)


    # def vis(axis_x,plus_std,minus_std,save_name='syn3'):
    #
    #     SMALL_SIZE = 10
    #     MEDIUM_SIZE = 22
    #     BIGGER_SIZE = 26
    #
    #     plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    #     plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    #     plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    #     plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    #     plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    #
    #     fig = plt.figure(dpi=120, figsize=(9, 6))
    #     plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
    #     ax2 = fig.add_subplot(111)
    #
    #     # 'blue',
    #     ax2.plot(axis_x, plus_std, marker='o',markersize=16,label='$Fid_{\\alpha_1,+}$',linewidth='3')  #
    #     ax2.plot(axis_x, minus_std, marker='v',markersize=16, label='$Fid_{\\alpha_2,-}$',linewidth='3')  # $Fid{\\alpha_2,-}$
    #     ax.errorbar(axises, fid_plus_mean, yerr=fid_plus_std, fmt='-o', capsize=5, label="Mean ± 1 SD")
    #     ax2.set_ylabel('mean',fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
    #     ax2.set_xlabel('M', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of
    #
    #     # ax2.yaxis.label.set_color('b')
    #     # ax2.spines['right'].set_color('r')
    #     # ax2.spines['right'].set_linewidth(1)
    #     # ax2.tick_params(colors ='b',axis='y')
    #     # ax2.spines['right'].set_color('r')
    #     ax2.legend() # loc='upper left'
    #
    #     new_values = plus_std
    #     new_values.extend(minus_std)
    #     length = max(new_values)-min(new_values)
    #     interval = length/4
    #     if interval >1:
    #         interval = int(interval)
    #     else:
    #         interval = float("%.2f"%interval)
    #     # y_major_locator = MultipleLocator(interval)
    #     # ax2.yaxis.set_major_locator(y_major_locator)
    #
    #     # plt.show()
    #     plt.savefig('./redata/gcn_parameter/%s_M_parameter_mean.png'%save_name)
    #     plt.savefig('./redata/gcn_parameter/%s_M_parameter_mean.pdf'%save_name)
    #

    vis_plus(axises,fid_plus_mean,fid_plus_std,name)
    vis_minus(axises,fid_minus_mean,fid_minus_std,name)



def vis_fid_explainers():
    # gnn

    def vis(axis,ori_fid_plus,ori_fid_minus,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        # ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')  #
        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$',
        #          linewidth='3')  # $Fid{\\alpha_2,-}$

        ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{+}$', linewidth='3')  #
        ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{-}$',
                 linewidth='3')
        ax2.set_ylabel('mean', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('M', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        plt.show()
        plt.savefig('./data/figure/%s_ori_fid.png'%save_name)
        plt.savefig('./data/figure/%s_ori_fid.pdf'%save_name)

    def vis_ori_plus(axis,ori_fid_plus_dict,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        # ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')  #
        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$',
        #          linewidth='3')  # $Fid{\\alpha_2,-}$
        markers = ['o','v','p','*']
        for i, key in enumerate(axis.keys()):
            explaination_name = key
            ori_fid_plus_data = ori_fid_plus_dict[key]
            ax_data = axis[key]
            ax2.plot(ax_data, ori_fid_plus_data, marker=markers[i], markersize=16, label=explaination_name, linewidth='3')  #

        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{-}$',
        #          linewidth='3')
        ax2.set_ylabel('$Fid_{+}$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('sparsity', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # plt.show()
        plt.savefig('./data/figure/%s_ori_fid+.png'%save_name)
        plt.savefig('./data/figure/%s_ori_fid+.pdf'%save_name)
        plt.close(fig)

    def vis_ours_plus(axis,ours_fid_plus_dict,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        # ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')  #
        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$',
        #          linewidth='3')  # $Fid{\\alpha_2,-}$
        markers = ['o','v','p','*']
        for i, key in enumerate(axis.keys()):
            explaination_name = key
            ori_fid_plus_data = ours_fid_plus_dict[key]
            ax_data = axis[key]
            ax2.plot(ax_data, ori_fid_plus_data, marker=markers[i], markersize=16, label=explaination_name, linewidth='3')  #

        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{-}$',
        #          linewidth='3')
        ax2.set_ylabel('$Fid_{\\alpha_1,+}$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('sparsity', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # plt.show()
        plt.savefig('./data/figure/%s_ours_fid+.png'%save_name)
        plt.savefig('./data/figure/%s_ours_fid+.pdf'%save_name)
        plt.close(fig)

    def vis_ours_minus(axis,ours_fid_minus_dict,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        # ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')  #
        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$',
        #          linewidth='3')  # $Fid{\\alpha_2,-}$
        markers = ['o','v','p','*']
        for i, key in enumerate(axis.keys()):
            explaination_name = key
            ori_fid_plus_data = ours_fid_minus_dict[key]
            ax_data = axis[key]
            ax2.plot(ax_data, ori_fid_plus_data, marker=markers[i], markersize=16, label=explaination_name, linewidth='3')  #

        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{-}$',
        #          linewidth='3')
        ax2.set_ylabel('$Fid_{\\alpha_2,-}$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('sparsity', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # plt.show()
        plt.savefig('./data/figure/%s_ours_fid-.png'%save_name)
        plt.savefig('./data/figure/%s_ours_fid-.pdf'%save_name)
        plt.close(fig)


    def vis_ori_minus(axis,ori_fid_minus_dict,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        # ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')  #
        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$',
        #          linewidth='3')  # $Fid{\\alpha_2,-}$
        markers = ['o','v','p','*']
        for i, key in enumerate(axis.keys()):
            explaination_name = key
            ori_fid_minus_data = ori_fid_minus_dict[key]
            ax_data = axis[key]
            ax2.plot(ax_data, ori_fid_minus_data, marker=markers[i], markersize=16, label=explaination_name, linewidth='3')  #

        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{-}$',
        #          linewidth='3')
        ax2.set_ylabel('$Fid_{-}$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('sparsity', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # plt.show()
        plt.savefig('./data/figure/%s_ori_fid-.png'%save_name)
        plt.savefig('./data/figure/%s_ori_fid-.pdf'%save_name)
        plt.close(fig)

    def vis_auc(axis,auc_dict,save_name='syn3'):
        SMALL_SIZE = 10
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

        fig = plt.figure(dpi=120, figsize=(9, 6))
        plt.subplots_adjust(left=0.17,right=0.98,top=0.98,bottom=0.15)
        ax2 = fig.add_subplot(111)

        # 'blue',
        # ax2.plot(axis, ori_fid_plus, marker='o', markersize=16, label='$Fid_{\\alpha_1,+}$', linewidth='3')  #
        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{\\alpha_2,-}$',
        #          linewidth='3')  # $Fid{\\alpha_2,-}$
        markers = ['o','v','p','*']
        for i, key in enumerate(axis.keys()):
            explaination_name = key
            ori_fid_minus_data = auc_dict[key]
            ax_data = axis[key]
            ax2.plot(ax_data, ori_fid_minus_data, marker=markers[i], markersize=16, label=explaination_name, linewidth='3')  #

        # ax2.plot(axis, ori_fid_minus, marker='v', markersize=16, label='$Fid_{-}$',
        #          linewidth='3')
        ax2.set_ylabel('$AUC$', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient of
        ax2.set_xlabel('sparsity', fontsize=BIGGER_SIZE)  # Spearman correlation coefficient  of

        # ax2.yaxis.label.set_color('b')
        # ax2.spines['right'].set_color('r')
        # ax2.spines['right'].set_linewidth(1)
        # ax2.tick_params(colors ='b',axis='y')
        # ax2.spines['right'].set_color('r')
        ax2.legend()  # loc='upper left'

        # plt.show()
        plt.savefig('./data/figure/%s_ori_auc.png'%save_name)
        plt.savefig('./data/figure/%s_ori_auc.pdf'%save_name)
        plt.close(fig)

    model_select = 'gcn' #''PGIN'
    name_dict= {'treecycles':'syn3', 'treegrids':'syn4','bashapes':'syn1','bacommunity':'syn2', 'ba2motifs':'ba2', 'mutag':'mutag'}
    explainers = ['gnnexplainer','pgexplainer','subgraphx','pgmexplainer'] # ,'subgraphx','pgmexplainer'
    datasets = [ 'treecycles','treegrids','ba2motifs','mutag']  # 'bashapes','bacommunity', 'treecycles','treegrids', 'ba2motifs' ,'mutag'

    for _dataset in datasets:
        axiss = {}
        value_ori_plus = {}
        value_ori_minus = {}
        value_ours_plus = {}
        value_ours_minus = {}
        value_auc = {}
        for index in range(0, 4):

            explainer_name = explainers[index]

            axiss[explainer_name] = []
            value_ori_plus[explainer_name] = []
            value_ori_minus[explainer_name] = []
            value_ours_plus[explainer_name] = []
            value_ours_minus[explainer_name] = []
            value_auc[explainer_name] = []

            explainer_dir = './data/%s' % explainer_name
            dataset_nikename = name_dict[_dataset]
            ori_npy_name_list = glob.glob(explainer_dir+"/%s/a/*%s*dict_ori.npy"%(model_select,dataset_nikename))
            ours_npy_name_list = glob.glob(explainer_dir + "/%s/a/*%s*dict_ours.npy" % (model_select, dataset_nikename))

            ori_data = np.load(ori_npy_name_list[0],allow_pickle=True).item()
            ours_data = np.load(ours_npy_name_list[0],allow_pickle=True).item()

            for key in ori_data.keys():
                if len(str(key)) >2:
                    continue
                ori_fid_plus = ori_data[key][0]
                ori_fid_minus = ori_data[key][4]
                sp = ori_data[key][-2]
                ours_fid_plus = ours_data[key][0]
                ours_fid_minus = ours_data[key][4]

                auc_ = ours_data[key][14]

                axiss[explainer_name].append(sp)
                value_ori_plus[explainer_name].append(ori_fid_plus)
                value_ori_minus[explainer_name].append(ori_fid_minus)
                value_ours_plus[explainer_name].append(ours_fid_plus)
                value_ours_minus[explainer_name].append(ours_fid_minus)
                value_auc[explainer_name].append(auc_)

            # sort and remap
            sort_index = np.argsort(axiss[explainer_name])
            axiss[explainer_name] = np.array(axiss[explainer_name])[sort_index]
            value_ori_plus[explainer_name] = np.array(value_ori_plus[explainer_name])[sort_index]
            value_ori_minus[explainer_name] = np.array(value_ori_minus[explainer_name])[sort_index]
            value_ours_plus[explainer_name] = np.array(value_ours_plus[explainer_name])[sort_index]
            value_ours_minus[explainer_name] = np.array(value_ours_minus[explainer_name])[sort_index]
            value_auc[explainer_name] = np.array(value_auc[explainer_name])[sort_index]

        vis_ori_plus( axiss,value_ori_plus,save_name= dataset_nikename)
        vis_ori_minus( axiss,value_ori_minus,save_name= dataset_nikename)
        vis_ours_plus( axiss,value_ours_plus,save_name= dataset_nikename)
        vis_ours_minus( axiss,value_ours_minus,save_name= dataset_nikename)
        vis_auc( axiss,value_auc,save_name= dataset_nikename)



if __name__ == "__main__":
    # vis_fid_explainers()
    # exit(0)
    #
    datasets = ['syn3','syn4','ba2','mutag']
    for  name  in datasets:
        dir = './redata'
        namelist = glob.glob(dir + '/max_length_*%s*new_fid*_10.npy'%name)
        namelist = sorted(namelist)
        data_dict = {}
        axises = []
        fid_plus_std = []
        fid_minus_std = []

        fid_plus_mean = []
        fid_minus_mean = []
        for path in namelist[1:]:
            key_list = path.split('_')
            key = key_list[2]
            value = int(key)
            axises.append(value)
            data = np.load(path, allow_pickle=True).item()
            xdata = data[('0.0','0.0')]
            fid_plus_mean.append(xdata[0])  # 1, 7
            fid_minus_mean.append(xdata[4])
            fid_plus_std.append(xdata[1])  # 1, 7
            fid_minus_std.append(xdata[5])

        for path in [namelist[0]]:
            key_list = path.split('_')
            key = key_list[2]
            value = int(key)
            axises.append(value)
            data = np.load(path, allow_pickle=True).item()
            xdata = data[('0.0','0.0')]
            fid_plus_mean.append(xdata[0])
            fid_minus_mean.append(xdata[4])
            fid_plus_std.append(xdata[1])  # 1, 7
            fid_minus_std.append(xdata[5])

        vis_time_mean_stdx(axises,fid_plus_std,fid_minus_std,fid_plus_mean,fid_minus_mean,name)

    exit(0)

    # datasets = ['syn3', 'syn4', 'ba2', 'mutag']
    # # name = 'mutag'
    # for name in datasets:
    #     dir = './redata/gcn_parameter'
    #     namelist = glob.glob(dir + '/*_%s_*new_fid*.npy'%name)
    #     results_dict = {}
    #     for path in namelist:
    #         if '0_0' in path:
    #             data = np.load(path, allow_pickle=True).item()
    #             # results_dict[0.1] = data
    #             alpha = 0.1
    #         if '0.3' in path:
    #             data = np.load(path, allow_pickle=True).item()
    #             # results_dict[0.3] = data
    #             alpha = 0.3
    #         if '0.5' in path:
    #             data = np.load(path, allow_pickle=True).item()
    #             # results_dict[0.5] = data
    #             alpha = 0.5
    #         if '0.7' in path:
    #             data = np.load(path, allow_pickle=True).item()
    #             # results_dict[0.7] = data
    #             alpha = 0.7
    #         if '0.9' in path:
    #             data = np.load(path, allow_pickle=True).item()
    #             # results_dict[0.9] = data
    #             alpha = 0.9
    #         resultss = test_correction_sp(name,data)
    #         results_dict[str(alpha)] = resultss
    #
    #     vis_2dfig(results_dict,namelist[0].replace('.npy',''))
    # exit(0)

    # testvalue()
    # gcn
    dir = './redata/gin'

    namelist = glob.glob(dir + '/*new_fid*.npy')
    for path in namelist:
        # if 'syn1' in path:
        #     pass
        # else:
        #     continue
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new_ratio(data, path.replace('.npy', ''))


    # graph
    namelist = glob.glob(dir+'/*ori_fid*.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori_ratio(data, path.replace('.npy',''))



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

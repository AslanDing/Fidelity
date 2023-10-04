import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

# random seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

def visual(src_embedding,gt_embedding,name):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    font1 = {'color': 'black', 'size': 18}

    plt.scatter(src_embedding[:, 0], src_embedding[:, 1],
                c='r', marker='o',label="original graph embedding")
    plt.scatter(gt_embedding[:, 0], gt_embedding[:, 1],
                c='g', marker='o',label="GT embedding")
    plt.legend()
    plt.savefig('./%s_tsne.png'%(name))
    plt.show()
    plt.cla()

def visual1(src_embedding,gt_embedding_plus,gt_embedding_G_E,name):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    font1 = {'color': 'black', 'size': 18}

    plt.scatter(src_embedding[:, 0], src_embedding[:, 1],
                c='r', marker='o',label="original graph embedding")
    plt.scatter(gt_embedding_plus[:, 0], gt_embedding_plus[:, 1],
                c='g', marker='o',label="expl graph embedding G-E")
    plt.scatter(gt_embedding_G_E[:, 0], gt_embedding_G_E[:, 1],
                c='k', marker='o',label="expl graph embedding G-e")

    plt.legend()
    plt.savefig('./%s_tsne_4.png'%(name))
    # plt.show()
    plt.cla()

def visual_label(embedding,label,name):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    font1 = {'color': 'black', 'size': 18}

    label_num = np.max(label)+1

    for i in range(label_num):
        indice = np.argwhere(label==i)
        plt.scatter(embedding[indice, 0], embedding[indice, 1], # c='r',
                marker='o', label="%s label %d"%(name,i))

    plt.legend()
    plt.savefig('./%s_tsne_label.png' % (name))
    # plt.show()
    plt.cla()
def visual_label_list(embedding_list,label,name_list,name="name"):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    font1 = {'color': 'black', 'size': 18}

    label_num = np.max(label)+1
    for embedding, name in zip(embedding_list,name_list):
        for i in range(label_num):
            indice = np.argwhere(label==i)
            plt.scatter(embedding[indice, 0], embedding[indice, 1], # c='r',
                    marker='o', label="%s label %d"%(name,i))

    plt.legend()
    plt.savefig('./%s_tsne_label.png' % name)
    # plt.show()
    plt.cla()

# src
# src_embedding1 = np.load('Figures/embedding_src1.npy')            # ori graph embedding
# src_expl_embedding1 = np.load('Figures/embedding_expl_src1.npy')  # expl graph embedding
# src_embedding = np.load('Figures/embedding_src.npy')            # ori graph embedding
# src_expl_embedding = np.load('Figures/embedding_expl_src.npy')  # expl graph embedding

#
# contrstive_expl_embedding = np.load('./embedding_expl_contrastive1.npy')
name = 'syn2'
src_embedding = np.load('./%s_embedding_src1.npy'%name)
# src_expl_embedding_mius = np.load('./embedding_expl_minus_src_list.npy')
src_expl_embedding_plus = np.load('./%s_embedding_expl_plus_src_list.npy'%name)
src_expl_embedding_minus = np.load('./%s_embedding_expl_minus_src_list.npy'%name)

src_expl_G_e_plus = np.load('./%s_embedding_G_E_list_plus.npy'%name)
src_expl_G_e_minus = np.load('./%s_embedding_G_E_list_minus.npy'%name)

label_list = np.load('./%s_label_list.npy'%name).squeeze()

#
# extend_src_embedding = np.load('Figures/embedding_src_extend.npy')
# extend_expl_embedding = np.load('Figures/embedding_expl_extend.npy')

ts = manifold.TSNE() #(perplexity= 5,learning_rate='auto',init="pca",method = 'exact',n_iter=250)

# x_ts = ts.fit_transform(np.concatenate([src_embedding,src_expl_embedding],axis=0))
#
# visual(x_ts[:src_embedding.shape[0],:],x_ts[src_embedding.shape[0]:src_embedding.shape[0]*2,:],'src')
# exit(0)

x_ts = ts.fit_transform(np.concatenate([src_embedding,
                                        src_expl_embedding_plus,src_expl_embedding_minus,
                                        src_expl_G_e_plus,src_expl_G_e_minus],axis=0))
x_min, x_max = x_ts.min(0), x_ts.max(0)
x_final = (x_ts - x_min) / (x_max - x_min)
s_l = src_embedding.shape[0]
s_p_l = src_expl_embedding_plus.shape[0]
s_p_m = src_expl_embedding_plus.shape[0]
s_g_e_l = src_expl_G_e_plus.shape[0]
s_g_e_m = src_expl_G_e_minus.shape[0]

src_embedding_xt = x_final[:s_l,:]
src_expl_embedding_plus_xt = x_final[s_l:s_l+s_p_l,:]
src_expl_embedding_minus_xt = x_final[s_l+s_p_l:s_l+s_p_l+s_p_m,:]
src_expl_embedding_G_e_plus = x_final[s_l+s_p_l+s_p_m:s_l+s_p_l+s_p_m+s_g_e_l,:]
src_expl_embedding_G_e_minus = x_final[s_l+s_p_l+s_p_m+s_g_e_l:s_l+s_p_l+s_p_m+s_g_e_l+s_g_e_m,:]

######
###### plus
visual1(src_embedding_xt,src_expl_embedding_plus_xt,
        src_expl_embedding_G_e_plus,'plus_all')
visual1(src_embedding_xt,src_expl_embedding_minus_xt,
        src_expl_embedding_G_e_minus,'minus_all')

visual_label(src_embedding_xt,label_list,"G")

visual_label(src_expl_embedding_plus_xt,label_list,"plus Nonexplain")
visual_label(src_expl_embedding_minus_xt,label_list,"minus Explain")

visual_label(src_expl_embedding_G_e_plus,label_list,"plus G-e Nonexplain")
visual_label(src_expl_embedding_G_e_minus,label_list,"minus G-e Explain")
# visual_label(contrastive_expl_embedding_xt,label_list,"contrastive_expl")

visual_label_list([src_embedding_xt,src_expl_embedding_plus_xt,src_expl_embedding_G_e_plus],
                  label_list,["G","Nonexplain","G-e Nonexplain"],"plus")

visual_label_list([src_embedding_xt,src_expl_embedding_minus_xt,src_expl_embedding_G_e_minus],
                  label_list,["G","Explain","G-e Explain"],"minus")

exit(0)

# gt = [0.18208918,-0.23347045,0.05472511]
# plt.bar(range(len(gt)), gt)
# plt.show()

size = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
font1 = {'color':'black','size':18}

x = np.arange(size)
x_axis = ['Ba2Motifs','Mutag','Ba-Shapes','Ba-Community','Tree-Cycles','Tree-Grids']
a = [0.18208918,  0.47859253,0.59777049, 0.46573313,  -0.09014929,  0.57236752]
b = [-0.23347045, 0.0,       0.24977046, 0.0,         0.0,          0.0]
c = [0.05472511,  0.0,       0.0,        0.0,         0.0,          0.0]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='w/o GT training ')
plt.bar(x + width, b, width=width, label='w/ GT training ')
plt.bar(x + 2 * width, c, width=width, label='contrastive training')
plt.legend()
plt.ylabel('Fid-',font1)
plt.xticks(np.arange(6),x_axis,rotation=20)
plt.savefig('./Figures/fid-_ft_compare.png')
plt.show()
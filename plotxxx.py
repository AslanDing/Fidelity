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
    plt.savefig('./tsne_%s.png'%(name))
    plt.show()
    plt.cla()

def visual1(src_embedding,gt_embedding,contrastive_embedding,contrastive_expl_embedding,name):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    font1 = {'color': 'black', 'size': 18}

    plt.scatter(src_embedding[:, 0], src_embedding[:, 1],
                c='r', marker='o',label="original graph embedding f")
    plt.scatter(contrastive_embedding[:, 0], contrastive_embedding[:, 1],
                c='b', marker='o',label="original graph embedding f'")
    plt.scatter(gt_embedding[:, 0], gt_embedding[:, 1],
                c='g', marker='o',label="GT embedding f")
    plt.scatter(contrastive_expl_embedding[:, 0], contrastive_expl_embedding[:, 1],
                c='k', marker='o',label="GT embedding f'")
    plt.legend()
    plt.savefig('./tsne_%s_4.png'%(name))
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
    plt.savefig('./tsne_%s_label.png' % (name))
    # plt.show()
    plt.cla()
def visual_label_list(embedding_list,label,name_list):
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
    plt.savefig('./fig_%s_label.png' % ("name"))
    plt.savefig('./fig_%s_label.pdf' % ("name"))
    # plt.show()
    plt.cla()

def visual_list(embedding_list,label,name_list):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    font1 = {'color': 'black', 'size': 18}
    fig = plt.figure(dpi=300)  # , dpi=60
    for embedding, name in zip(embedding_list,name_list):
        plt.scatter(embedding[:, 0], embedding[:, 1], # c='r',
                marker='o', label="%s"%(name))
    plt.legend()

    # ax = plt.gca()
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # plt.axis('off')
    plt.savefig('./Nfig_%s_labelx.png' % ("name"))
    plt.savefig('./Nfig_%s_labelx.pdf' % ("name"))
    plt.cla()


src_embedding = np.load('./Figures/ba_gt_fig/ba2_embedding_src1.npy')
explain_embedding = np.load('./Figures/ba_gt_fig/ba2_embedding_expl_minus_src_list.npy')
non_explain_embedding = np.load('./Figures/ba_gt_fig/ba2_embedding_expl_plus_src_list.npy')

label_list = np.load('./Figures/ba_gt_fig/ba2_label_list.npy').squeeze()

ts = manifold.TSNE()
x_ts = ts.fit_transform(np.concatenate([src_embedding,explain_embedding,non_explain_embedding],axis=0))

s_l = src_embedding.shape[0]
e_l = explain_embedding.shape[0]
ne_l = non_explain_embedding.shape[0]

src_embedding_xt = x_ts[:s_l,:]
expl_embedding_xt = x_ts[s_l:s_l+ne_l,:]
nonexpl_embedding_xt = x_ts[s_l+ne_l:,:]

visual_list([src_embedding_xt,expl_embedding_xt,nonexpl_embedding_xt],
                  label_list,["$G$","$G_{sub}$","$G-G_{sub}$"])
exit(0)
# src
# src_embedding1 = np.load('Figures/embedding_src1.npy')            # ori graph embedding
# src_expl_embedding1 = np.load('Figures/embedding_expl_src1.npy')  # expl graph embedding
# src_embedding = np.load('Figures/embedding_src.npy')            # ori graph embedding
# src_expl_embedding = np.load('Figures/embedding_expl_src.npy')  # expl graph embedding

#ExplanationEvaluation/datasets/pkls
contrstive_embedding = np.load('./embedding_src_contrastive1.npy')
contrstive_expl_embedding = np.load('./embedding_expl_contrastive1.npy')

src_embedding = np.load('./embedding_src1.npy')
src_expl_embedding = np.load('./embedding_expl_src1.npy')

label_list = np.load('./label_list.npy').squeeze()

#
# extend_src_embedding = np.load('Figures/embedding_src_extend.npy')
# extend_expl_embedding = np.load('Figures/embedding_expl_extend.npy')

ts = manifold.TSNE() #(perplexity= 5,learning_rate='auto',init="pca",method = 'exact',n_iter=250)

# x_ts = ts.fit_transform(np.concatenate([src_embedding,src_expl_embedding],axis=0))
#
# visual(x_ts[:src_embedding.shape[0],:],x_ts[src_embedding.shape[0]:src_embedding.shape[0]*2,:],'src')
# exit(0)

x_ts = ts.fit_transform(np.concatenate([contrstive_embedding,contrstive_expl_embedding,
                                        src_embedding,src_expl_embedding],axis=0))
x_min, x_max = x_ts.min(0), x_ts.max(0)
x_final = (x_ts - x_min) / (x_max - x_min)
c_l = contrstive_embedding.shape[0]
c_e_l = contrstive_expl_embedding.shape[0]
s_l = src_embedding.shape[0]
s_e_l = src_expl_embedding.shape[0]

src_embedding_xt = x_final[c_l+c_e_l:c_l+c_e_l+s_l,:]
src_expl_embedding_xt = x_final[c_l+c_e_l+s_l:,:]
contrastive_embedding_xt = x_final[:c_l,:]
contrastive_expl_embedding_xt = x_final[c_l:c_l+c_e_l,:]


visual1(src_embedding_xt,src_expl_embedding_xt,
        contrastive_embedding_xt,contrastive_expl_embedding_xt,'contrastive1')

visual_label(src_embedding_xt,label_list,"src")
visual_label(src_expl_embedding_xt,label_list,"src_expl")
visual_label(contrastive_embedding_xt,label_list,"contrastive")
visual_label(contrastive_expl_embedding_xt,label_list,"contrastive_expl")

visual_label_list([src_embedding_xt,src_expl_embedding_xt,contrastive_embedding_xt,contrastive_expl_embedding_xt],
                  label_list,["src","src_expl","contrastive","contrastive_expl"])
#
# x_ts = ts.fit_transform(np.concatenate([extend_src_embedding,extend_expl_embedding],axis=0))
# x_min, x_max = x_ts.min(0), x_ts.max(0)
# x_final = (x_ts - x_min) / (x_max - x_min)
# visual(x_final[:extend_src_embedding.shape[0],:],x_final[extend_src_embedding.shape[0]:,:],'extend')
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
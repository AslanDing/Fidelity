import numpy as np
import matplotlib.pyplot as plt

dataset_name = "ba2motifs"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.18316274, 0.18560244, 0.18575895 ,0.18964158 ,0.19628033 ,0.20198597,
 0.20171516 ,0.20280405, 0.1792084,  0.13197245]
fid_p_gnn_x = [0.5 ,       0.53928589, 0.59800643 ,0.63750065, 0.69622116 ,0.74485,
 0.79443592, 0.84285643 ,0.89265062 ,0.94107116 ]
# gt_gnn = [0.59777049]
gt_gnn_ = [0.20196588]
sparse = [0.78470129]
fid_p_pge = [0.25015617, 0.25135883, 0.25241991 ,0.25444533 ,0.25880573, 0.258257,
 0.25427616 ,0.23017812, 0.25417569 ,0.23360179]
fid_p_pge_x = [0.51388701, 0.55184577, 0.60569697 ,0.64345864 ,0.701401  , 0.7494522,
 0.79641489, 0.84485337, 0.89556671, 0.94727295]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn_,'o-', label = "GT")
# plt.plot(sparse, gt_gnn,'o-', label = "W/o GT trining")
# plt.plot(sparse, gt_gnn_,'o-', label = "W/ GT trining")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s_contrastive.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()

dataset_name = "ba2motifs"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.19098105 , 0.18819907 , 0.18774491 , 0.19048099 , 0.19069499 , 0.1810171,
  0.17064416 , 0.15374151 , 0.14093028  ,0.16350379]
fid_p_gnn_x = [0.5 ,       0.53928589, 0.59800643 ,0.63750065, 0.69622116 ,0.74485,
 0.79443592, 0.84285643 ,0.89265062 ,0.94107116 ]
# gt_gnn = [0.59777049]
gt_gnn_ = [-0.23347046]
sparse = [0.78470129]
fid_p_pge = [0.10106552, 0.10903299 ,0.12493612 ,0.12951233 ,0.15716387 ,0.17172886,
 0.17691404, 0.17489533, 0.20370797, 0.22406166]
fid_p_pge_x = [0.51388701, 0.55184577, 0.60569697 ,0.64345864 ,0.701401  , 0.7494522,
 0.79641489, 0.84485337, 0.89556671, 0.94727295]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn_,'o-', label = "GT")
# plt.plot(sparse, gt_gnn,'o-', label = "W/o GT trining")
# plt.plot(sparse, gt_gnn_,'o-', label = "W/ GT trining")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s_contrastive.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()



dataset_name = "bashapes"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.38926853 ,0.39673359, 0.40478613, 0.41134159, 0.42811868, 0.44655229,
 0.47491719, 0.4801092,  0.47559208 ,0.4632163]
fid_p_gnn_x = [0.5,       0.54917458, 0.59925298, 0.64909949, 0.69915201, 0.74962115,
 0.79925617, 0.84926007, 0.8991552,  0.94918499 ]
# gt_gnn = [0.59777049]
gt_gnn_ = [0.49346322]
sparse = [0.988657]
fid_p_pge = [0.4308962,  0.43702012, 0.44232923, 0.44673349, 0.44959469, 0.4523018,
 0.45548216, 0.46165291, 0.46571919, 0.47233542 ]
fid_p_pge_x = [0.5  ,      0.54917458, 0.59925298, 0.64910128, 0.69915201, 0.74962281,
 0.79925617 ,0.84926144, 0.8991552 , 0.94918499]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn_,'o-', label = "GT")
# plt.plot(sparse, gt_gnn,'o-', label = "W/o GT trining")
# plt.plot(sparse, gt_gnn_,'o-', label = "W/ GT trining")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s_contrastive.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()

dataset_name = "bashapes"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.4926103,  0.4926103,  0.4926103,  0.49253209, 0.49247921, 0.49247921,
 0.49271529, 0.49274879, 0.49348206 ,0.49493519]
fid_p_gnn_x = [0.5,       0.54917458, 0.59925298, 0.64909949, 0.69915201, 0.74962115,
 0.79925617, 0.84926007, 0.8991552,  0.94918499 ]
# gt_gnn = [0.59777049]
gt_gnn_ = [0.47468867]
sparse = [0.988657]
fid_p_pge = [0.49346321, 0.49346322, 0.49346321, 0.49346322, 0.49346321, 0.49346322,
 0.49346322 ,0.49346322, 0.49346321, 0.50551239]
fid_p_pge_x = [0.5  ,      0.54917458, 0.59925298, 0.64910128, 0.69915201, 0.74962281,
 0.79925617 ,0.84926144, 0.8991552 , 0.94918499]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn_,'o-', label = "GT")
# plt.plot(sparse, gt_gnn,'o-', label = "W/o GT trining")
# plt.plot(sparse, gt_gnn_,'o-', label = "W/ GT trining")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s_contrastive.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()
exit(0)

dataset_name = "bashapes"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.12984306 ,0.14629425, 0.16293556, 0.19070556, 0.23354062, 0.29259143,
 0.40290262, 0.44168593, 0.4669044,  0.48277862]
fid_p_gnn_x = [0.5, 0.54917458, 0.59925298, 0.64909949, 0.69915201, 0.74962115,
            0.79925617, 0.84926007, 0.8991552,  0.94918499 ]
gt_gnn = [0.59777049]
gt_gnn_ = [0.24977046]
sparse = [0.988657]
fid_p_pge = [0.05258266 ,0.0513992 , 0.05356111, 0.05520542, 0.05487373, 0.05283327,
                0.0516097,  0.05665417, 0.06040075, 0.07616623]
fid_p_pge_x = [0.5, 0.5491759,  0.59925826, 0.64910128, 0.69915201, 0.74962465,
            0.79925617, 0.84926585, 0.8991552,  0.94918499]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
# plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
# plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "W/o GT trining")
plt.plot(sparse, gt_gnn_,'o-', label = "W/ GT trining")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s_extend.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()
exit(0)

dataset_name = "ba2motifs"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.19526557 ,0.19581262 ,0.19596041 ,0.19606427 ,0.1955731  ,0.19707693,
 0.19622946, 0.19758127 ,0.1961532,  0.19687951]
fid_p_gnn_x = [0.5,        0.53928589, 0.59800643, 0.63750065, 0.69622116 ,0.74485,
 0.79443592, 0.84285643 ,0.89265062 ,0.94107116]
gt_gnn = [0.18208918]
gt_gnn_ = [-0.23347045]
sparse = [0.78470129]
fid_p_pge = [0.20217884, 0.20203753, 0.20348009, 0.20223501, 0.20187866, 0.20245931,
 0.19837697, 0.19799859, 0.19454066 ,0.19694011 ]
fid_p_pge_x = [0.51394509, 0.551835,   0.60573543, 0.64347864, 0.70142139, 0.7494822,
 0.79640566, 0.84482513, 0.89551902, 0.94725295]



SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
# plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
# plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "W/o GT trining")
plt.plot(sparse, gt_gnn_,'o-', label = "W/ GT trining")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s_extend.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()
exit(0)

dataset_name = "mutag"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.13726878 , 0.13728779 , 0.134369  ,  0.1255786 ,  0.10975084 , 0.08825768,
  0.06783183 , 0.04047155 , 0.00789739, -0.02627256 ]
fid_p_gnn_x = [0.50000182, 0.54048978 ,0.5920506,  0.64120624 ,0.6916024,  0.74517035,
 0.79152136, 0.84060699 ,0.89111149 ,0.94132345 ]
gt_gnn = [0.14258379]
sparse = [0.88775234]
fid_p_pge = [0.41381962, 0.41381962, 0.41380188, 0.41371727, 0.41301257, 0.41283737,
 0.41384763, 0.41953385, 0.4492972,  0.47111804 ]
fid_p_pge_x = [0.90281865, 0.90281865, 0.90281978, 0.90284677, 0.90306522, 0.90371447,
 0.90486946, 0.90799767, 0.9211354,  0.95284559 ]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)

dataset_name = "mutag"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.20986704, 0.232591 ,  0.2549844 , 0.27444237, 0.29011414, 0.30420982,
 0.30505279, 0.28762883, 0.24631117, 0.16650397]
fid_p_gnn_x = [0.50000182, 0.54048978 ,0.5920506,  0.64120624 ,0.6916024,  0.74517035,
 0.79152136, 0.84060699 ,0.89111149 ,0.94132345 ]
gt_gnn = [0.36448708]
sparse = [0.88775234]
fid_p_pge = [-0.02806597, -0.02806597, -0.02806518, -0.02805962, -0.02795794, -0.02763947,
 -0.02659151, -0.02289884, -0.00417707 , 0.0355776 ]
fid_p_pge_x = [0.90281865, 0.90281865, 0.90281978, 0.90284677, 0.90306522, 0.90371447,
 0.90486946, 0.90799767, 0.9211354,  0.95284559 ]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)


dataset_name = "ba2motifs"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.19526557 ,0.19581262 ,0.19596041 ,0.19606427 ,0.1955731  ,0.19707693,
 0.19622946, 0.19758127 ,0.1961532,  0.19687951]
fid_p_gnn_x = [0.5,        0.53928589, 0.59800643, 0.63750065, 0.69622116 ,0.74485,
 0.79443592, 0.84285643 ,0.89265062 ,0.94107116]
gt_gnn = [0.18208918]
sparse = [0.78470129]
fid_p_pge = [0.20217884, 0.20203753, 0.20348009, 0.20223501, 0.20187866, 0.20245931,
 0.19837697, 0.19799859, 0.19454066 ,0.19694011 ]
fid_p_pge_x = [0.51394509, 0.551835,   0.60573543, 0.64347864, 0.70142139, 0.7494822,
 0.79640566, 0.84482513, 0.89551902, 0.94725295]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)

dataset_name = "ba2motifs"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.19619533 ,0.19646045 ,0.19649441 ,0.19640893 ,0.19538567 ,0.19284467,
 0.19389023 ,0.18321628, 0.1701662 , 0.14030754]
fid_p_gnn_x = [0.5,        0.53928589, 0.59800643, 0.63750065, 0.69622116 ,0.74485,
 0.79443592, 0.84285643 ,0.89265062 ,0.94107116]
gt_gnn = [0.1992873]
sparse = [0.78470129]
fid_p_pge = [0.20040508, 0.20139835, 0.19933248, 0.19933743, 0.20122878 ,0.19776954,
 0.20046326, 0.18846045, 0.24581789, 0.23328982]
fid_p_pge_x = [0.51394509, 0.551835,   0.60573543, 0.64347864, 0.70142139, 0.7494822,
 0.79640566, 0.84482513, 0.89551902, 0.94725295]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)



dataset_name = "treegrids"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.43801628, 0.42948258, 0.41693337, 0.4068694 , 0.38692657, 0.35518925,
 0.3331613,  0.29365534, 0.25893848, 0.21601502 ]
fid_p_gnn_x = [0.5 ,       0.54005494, 0.58941104 ,0.63167984 ,0.68452914, 0.74208711,
 0.78411427 ,0.83627152, 0.87923238 ,0.92789641 ]
gt_gnn = [0.57236752]
sparse = [0.57261553]
fid_p_pge = [0.17250918, 0.13110808, 0.1517902,  0.10538282, 0.23628623, 0.20529284,
 0.18837633, 0.21865696, 0.18638806, 0.17103213]
fid_p_pge_x = [0.50469765 ,0.54677476 ,0.59538186 ,0.63618672 ,0.68590665, 0.74504174,
 0.79006426, 0.83939924, 0.88729616 ,0.929367]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)

dataset_name = "treegrids"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.46531318 ,0.46391951, 0.45306761 ,0.44257564, 0.42795199, 0.41783116,
            0.40509145, 0.36438099, 0.30793622, 0.23544187 ]
fid_p_gnn_x = [0.5 ,       0.54005494, 0.58941104 ,0.63167984 ,0.68452914, 0.74208711,
 0.78411427 ,0.83627152, 0.87923238 ,0.92789641 ]
gt_gnn = [0.5723722]
sparse = [0.57261553]
fid_p_pge = [0.43769328, 0.46116436 ,0.51111941, 0.51144205 ,0.53718873, 0.49225004,
 0.47009446, 0.39178338 ,0.36576014 ,0.34865667]
fid_p_pge_x = [0.50469765 ,0.54677476 ,0.59538186 ,0.63618672 ,0.68590665, 0.74504174,
 0.79006426, 0.83939924, 0.88729616 ,0.929367]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)


dataset_name = "treecycles"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [-0.00644878, -0.02022852, -0.03186104, -0.03903093, -0.04384709, -0.03429249,
 -0.02508922, -0.02154037 ,-0.0208153 , -0.02522175  ]
fid_p_gnn_x = [0.50004386, 0.53925649 ,0.58731229, 0.63486211, 0.68525577, 0.74183339,
 0.78595401, 0.8357921 , 0.88389749, 0.93139773 ]
gt_gnn = [-0.09014929]
sparse = [0.76200367]
fid_p_pge = [-0.01466556 , 0.00185146 ,-0.00282362, -0.0219167 , -0.02142384, -0.04287969,
 -0.08748539, -0.06844384, -0.07087733, -0.08307503]
fid_p_pge_x = [0.50858555 ,0.55531643 ,0.59364837 ,0.65248149 ,0.69264729, 0.75391102,
 0.79455908 ,0.84843862, 0.89729238 ,0.94010607]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8.5,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)


dataset_name = "treecycles"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [-0.02751511,  0.01145159 , 0.06363317  ,0.127355  ,  0.18738717 , 0.21874702,
  0.20509759,  0.16932616 , 0.12644123 , 0.08196149 ]
fid_p_gnn_x = [0.50004386, 0.53925649 ,0.58731229, 0.63486211, 0.68525577, 0.74183339,
 0.78595401, 0.8357921 , 0.88389749, 0.93139773 ]
gt_gnn = [-0.00957516]
sparse = [0.76200367]
fid_p_pge = [0.10428213,  0.10488707,  0.11226265,  0.11997022,  0.11081697,  0.09480391,
  0.12183627 , 0.11044613 , 0.07926961 , 0.06114235 ]
fid_p_pge_x = [0.50858555 ,0.55531643 ,0.59364837 ,0.65248149 ,0.69264729, 0.75391102,
 0.79455908 ,0.84843862, 0.89729238 ,0.94010607]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)



dataset_name = "bacommunity"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.20118618 ,0.21460659 ,0.22875285 ,0.23437528 ,0.2410462 , 0.23777505,
            0.24561059, 0.25178426, 0.26070928, 0.26968357 ]
fid_p_gnn_x = [0.50000101 ,0.54941199 ,0.59949294 ,0.64939109 ,0.69946488 ,0.74965132,
 0.79942863 ,0.84934699, 0.89942314 ,0.94932609 ]
gt_gnn = [0.46573313]
sparse = [0.98235972]
fid_p_pge = [0.10066311, 0.11058016, 0.11959046, 0.1337583,  0.1516369,  0.16800811,
 0.19383898, 0.22537792, 0.24536841, 0.27964756]
fid_p_pge_x = [0.5,        0.54941059, 0.59949351, 0.64938988, 0.69946488 ,0.74965132,
 0.79942704 ,0.84934561, 0.89942245 ,0.94932464]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)

dataset_name = "bacommunity"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.45764283 ,0.45521277, 0.45601928 ,0.45502917, 0.45791801 ,0.45956713,
             0.45806897, 0.4570217 , 0.45613114, 0.44860378 ]
fid_p_gnn_x = [0.50000101 ,0.54941199 ,0.59949294 ,0.64939109 ,0.69946488 ,0.74965132,
 0.79942863 ,0.84934699, 0.89942314 ,0.94932609 ]
gt_gnn = [0.25493578]
sparse = [0.98235972]
fid_p_pge = [0.48365738, 0.48637169, 0.48729543, 0.48608466, 0.49006175 ,0.49266606,
 0.49286551, 0.49844962, 0.49472077, 0.48162964]
fid_p_pge_x = [0.5,        0.54941059, 0.59949351, 0.64938988, 0.69946488 ,0.74965132,
 0.79942704 ,0.84934561, 0.89942245 ,0.94932464]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)


dataset_name = "bashapes"
fid_name = "Fid-"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.12984306 ,0.14629425, 0.16293556, 0.19070556, 0.23354062, 0.29259143,
 0.40290262, 0.44168593, 0.4669044,  0.48277862]
fid_p_gnn_x = [0.5, 0.54917458, 0.59925298, 0.64909949, 0.69915201, 0.74962115,
            0.79925617, 0.84926007, 0.8991552,  0.94918499 ]
gt_gnn = [0.59777049]
sparse = [0.988657]
fid_p_pge = [0.05258266 ,0.0513992 , 0.05356111, 0.05520542, 0.05487373, 0.05283327,
                0.0516097,  0.05665417, 0.06040075, 0.07616623]
fid_p_pge_x = [0.5, 0.5491759,  0.59925826, 0.64910128, 0.69915201, 0.74962465,
            0.79925617, 0.84926585, 0.8991552,  0.94918499]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

# exit(0)

dataset_name = "bashapes"
fid_name = "Fid+"
x_axis = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
fid_p_gnn = [0.59627417,0.59627417,0.59627417,
             0.59619585,0.59614056,0.59614056,
             0.59628171, 0.5962408,  0.59666738, 0.59678191]
fid_p_gnn_x = [0.5, 0.54917458, 0.59925298, 0.64909949, 0.69915201, 0.74962115,
            0.79925617, 0.84926007, 0.8991552,  0.94918499 ]
gt_gnn = [0.50960264]
sparse = [0.988657]
fid_p_pge = [0.59777049, 0.59777048, 0.59777049,
             0.59777048, 0.59777049, 0.59777048,
             0.59777049, 0.59777049, 0.59777049, 0.60495584]
fid_p_pge_x = [0.5, 0.5491759,  0.59925826, 0.64910128, 0.69915201, 0.74962465,
            0.79925617, 0.84926585, 0.8991552,  0.94918499]


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize
font1 = {'color':'black','size':18}
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(fid_p_gnn_x, fid_p_gnn,'o-', label = "GNNExpl.")
plt.plot(fid_p_pge_x, fid_p_pge,'o-', label = "PGExpl. ")
plt.plot(sparse, gt_gnn,'o-', label = "GT")
plt.legend()
plt.ylabel(fid_name,font1)
plt.xlabel('Sparsity',font1)
plt.savefig('./Figures/%s_%s.png'%(dataset_name,fid_name))
plt.clf()
plt.cla()
# plt.show()

exit(0)


from scipy.sparse import coo_matrix, csr_matrix

new_arr = np.arange(0,16)
new_arr = new_arr.reshape([4,4])
print(new_arr)
csr_m = csr_matrix(new_arr)

indexs = [[1,2],[2,3]]
result = csr_m[indexs[0],indexs[1]]
print(result)


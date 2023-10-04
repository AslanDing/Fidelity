import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.training_contrastive import train_node_contrastive,train_graph_contrastive,train_graph,train_node_extend,train_graph_extend

import torch
import numpy as np

_dataset = 'mutag' # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, batchnorm
_model = 'gnn' if _folder == 'replication' else 'ori'

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/models/model_{_model}_{_dataset}.json"

config = Selector(config_path)
extension = (_folder == 'extension')

config = Selector(config_path).args

torch.manual_seed(config.model.seed)
torch.cuda.manual_seed(config.model.seed)
np.random.seed(config.model.seed)

_dataset = config.model.dataset
_explainer = 'PGIN' #

#
# if _dataset[:3] == "syn":
#     train_node_extend(_dataset, _explainer, config.model)
# elif _dataset == "ba2" or _dataset == "mutag":
#     train_graph_extend(_dataset, _explainer, config.model)
# exit(0)

if _dataset[:3] == "syn":
    train_node_contrastive(_dataset, _explainer, config.model)
elif _dataset == "ba2" or _dataset == "mutag":
    train_graph_contrastive(_dataset, _explainer, config.model)

"""
bashapes:
before:
    all final gt_acc:0.22857142857142856  
    explanation test 0.0    
    nonexplanation [[0.18195109 0.22327563 0.55465907 0.04011431]] 0.03333333333333333 
    final train_acc:0.9696428571428571, val_acc: 1.0, test_acc: 1.0
    
bacommunity:
before:
    all final gt_acc:0.15285714285714286  
    explanation test 0.166666   
    nonexplanation [[0.03041783 0.39859346 0.05361462 0.01160464 0.06900415 0.3834916  0.03570093 0.01757281]] 0.48333333333333334
    [0.03041782 0.39859346 0.05361462 0.01160465 0.06900414 0.38349164
    0.03570093 0.01757281]
    0.48333333333333334
    final train_acc:0.9017857142857143, val_acc: 0.7571428571428571, test_acc: 0.7214285714285714

treecycles:
before:
    all final gt_acc:0.2755453501722158    
    explanation test  1.0   
    nonexplanation [0.36428562 0.6357144 ] 0.95
    [0.36428562 0.6357144 ]
    0.95
    final train_acc:0.9425287356321839, val_acc: 0.9770114942528736, test_acc: 0.9431818181818182

treegrids
before:
    all final gt_acc:0.06498781478472786   
    explanation test  0.11418685121107267  
    nonexplanation  [[0.82568085 0.17431915]]  0.1245674740484429
    [0.82568085 0.17431915]
    0.1245674740484429
    final train_acc:0.9634146341463414, val_acc: 0.991869918699187, test_acc: 0.9919354838709677


motifts
before :
    all final  gt_val acc:0.5  explanation test  0.51
    explanation
    [0.8424223  0.15757775] tensor([[0.8563, 0.1437]], device='cuda:0') tensor([[0.8280, 0.1720]], device='cuda:0')
    final  gt_val acc:0.51
    non-explanation
    [0.6678383  0.33216164] tensor([[0.6682, 0.3318]], device='cuda:0') tensor([[0.6674, 0.3326]], device='cuda:0')
    final  gt_val acc:0.51
    final train_acc:0.99375, val_acc: 1.0, test_acc: 0.99
fintune:
    final  gt_val acc:1.0
    final train_acc:1.0, val_acc: 1.0, test_acc: 1.0

mutag
before:
    all final  gt_val acc:0.5718238413649989    
    explanation test :0.8541871921182266
    [0.7449755 0.2550245] tensor([[0.7450, 0.2550]], device='cuda:0') tensor([[nan, nan]], device='cuda:0')
    final  gt_val acc:0.8541871921182266
    
    non-explanation 
    [0.52307224 0.47692782] tensor([[0.5231, 0.4769]], device='cuda:0') tensor([[nan, nan]], device='cuda:0')
    final  gt_val acc:0.5182266009852217
    
    final train_acc:0.8253098875756703, val_acc: 0.8248847926267281, test_acc: 0.8133640552995391
    
after:
    120 final  gt_val acc:0.5621397279225271
    final train_acc:0.7985010089362928, val_acc: 0.8271889400921659, test_acc: 0.804147465437788
    110 final  gt_val acc:0.5536084851279687
    final train_acc:0.7817814932257134, val_acc: 0.783410138248848, test_acc: 0.7995391705069125
    100 final  gt_val acc:0.5842748443624626
    final train_acc:0.8117613144998559, val_acc: 0.8133640552995391, test_acc: 0.804147465437788
    90  final  gt_val acc:0.6110214433940512
    final train_acc:0.8123378495243586, val_acc: 0.8018433179723502, test_acc: 0.7995391705069125
    80 final  gt_val acc:0.5856582891399585
    final train_acc:0.8083021043528394, val_acc: 0.8087557603686636, test_acc: 0.7949308755760369
    70 final  gt_val acc:0.5831219737145492
    final train_acc:0.801671951571058, val_acc: 0.8064516129032258, test_acc: 0.7903225806451613
    30 final  gt_val acc:0.6091768503573899
    final train_acc:0.7990775439607957, val_acc: 0.8087557603686636, test_acc: 0.7880184331797235
    20 final  gt_val acc:0.608485127968642
    final train_acc:0.8005188815220524, val_acc: 0.8133640552995391, test_acc: 0.7949308755760369
    10 final  gt_val acc:0.5992621627853355
    final train_acc:0.8031132891323148, val_acc: 0.8064516129032258, test_acc: 0.7880184331797235
"""


"""
PGIN

bashapes
explanation
[1.8237242e-06 6.5920127e-09 6.3578844e-02 9.3641949e-01]
final gt_acc:0.0
non-explanation
[9.2539054e-01 7.4609473e-02 1.4214450e-15 1.1220215e-30]
final gt_acc:0.08333333333333333
final train_acc:0.9982142857142857, val_acc: 1.0, test_acc: 0.9714285714285714

bacommunity
explanation
[4.9005268e-07 4.5768986e-11 3.7650269e-04 9.9812919e-01 9.1147193e-08
 3.6834139e-10 1.9557751e-04 1.2979934e-03]
final gt_acc:0.0
non-explanation
[1.7120498e-01 1.2314464e-01 2.6649619e-03 1.9002648e-09 1.4432690e-04
 4.4123676e-01 2.2499706e-01 3.6607321e-02]
final gt_acc:0.1
final train_acc:0.9633928571428572, val_acc: 0.75, test_acc: 0.7571428571428571

treecycles
explanation
[0.9948083  0.00519182]
final gt_acc:0.0
non-explanation
[1.000000e+00 8.043824e-11]
final gt_acc:0.0
final train_acc:1.0, val_acc: 1.0, test_acc: 0.9886363636363636

treegrids
explanation
[0.99287814 0.00712192]
final gt_acc:0.0
non-explanation
[0.9933077  0.00669227]
final gt_acc:0.0
final train_acc:0.9776422764227642, val_acc: 0.991869918699187, test_acc: 0.967741935483871

ba2motifs
explanation
[0.5255714  0.47442847] tensor([[0.5318, 0.4682]], device='cuda:0') tensor([[0.5191, 0.4809]], device='cuda:0')
final  gt_val acc:0.525
non-explanation
[0.5376281 0.4623719] tensor([[0.5377, 0.4623]], device='cuda:0') tensor([[0.5375, 0.4625]], device='cuda:0')
final  gt_val acc:0.51
final train_acc:0.99, val_acc: 1.0, test_acc: 1.0

mutag
explanation
[0.34053305 0.65946704] tensor([[0.3405, 0.6595]], device='cuda:0') tensor([[nan, nan]], device='cuda:0')
final  gt_val acc:0.5064039408866995
non-explanation
[0.19797541 0.8020246 ] tensor([[0.1980, 0.8020]], device='cuda:0') tensor([[nan, nan]], device='cuda:0')
final  gt_val acc:0.1103448275862069
final train_acc:0.8630729316805996, val_acc: 0.8387096774193549, test_acc: 0.815668202764977

"""
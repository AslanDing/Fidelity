# This Project is for our work TOWARDS ROBUST FIDELITY FOR EVALUATING EXPLAINABILITY OF GRAPH NEURAL NETWORKS

## In this project, we present our fidelity, a new metric for XGNN evaluation. This project would contain two version of our work, continuous version and discrete version.

## for results reproduce
- generate samples, please run generate_edit_distance.py
- generate ori fidelity results, please run experiment_editdistance_ori_fid.py
- generate our fidelity results, please run experiment_editdistance_new_fid.py

## Continuous Version

### Probability ori. Fidelity results of Ba2Motifs dataset(ACC results can be found in ./pictures), the x-axis means adding non-explanation edges to GT, y-axis means remove edges from GT.
<p align="center">
<img src="./pictures/GNN_ba2_results_ori_fid_1fid_plus prob.png" width = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Original Fidelity+.
</p>

<p align="center">
<img src="./pictures/GNN_ba2_results_ori_fid_1fid_minus prob.png" width = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Original Fidelity+.
</p>

<p align="center">
<img src="./pictures/GNN_ba2_results_ori_fid_1fid_Delta prob.png" width = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Original Fidelity_$\Delta$.
</p>

### Probability our Fidelity results of Ba2Motifs dataset($\Alpha_1 = 0.1$,$\Alpha_2 = 0.9$)(ACC results can be found in ./pictures)
<p align="center">
<img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_plus prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Our Fidelity+.
</p>

<p align="center">
<img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_minus prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Our Fidelity+.
</p>

<p align="center">
<img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_Delta prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Our Fidelity_$\Delta$.
</p>

### The results of TreeCycle dataset can be found in './pictures' folder.




## Discrete Version

### Probability ori. Fidelity results of Ba2Motifs dataset(ACC results can be found in ./pictures)
<p align="center">
<img src="./pictures/GNN_ba2_results_ori_fid_1_kfid_plus prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Original Fidelity+.
</p>

<p align="center">
<img src="./pictures/GNN_ba2_results_ori_fid_1_kfid_minus prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Original Fidelity+.
</p>

<p align="center">
<img src="./pictures/GNN_ba2_results_ori_fid_1_kfid_Delta prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Original Fidelity_$\Delta$.
</p>

### Probability our Fidelity results of Ba2Motifs dataset($k_plus = 1$,$k_minus = 1$)(ACC results can be found in ./pictures)
<p align="center">
<img src="./pictures/ba2_results_new_fid_1_1_seeds_1_k_fid_plus prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Our Fidelity+.
</p>

<p align="center">
<img src="./pictures/ba2_results_new_fid_1_1_seeds_1_k_fid_minus prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Our Fidelity-.
</p>

<p align="center">
<img src="./pictures/ba2_results_new_fid_1_1_seeds_1_k_fid_Delta prob.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Our Fidelity_$\Delta$.
</p>


### The results of TreeCycle dataset can be found in './pictures' folder.

## observation（continuous version observation is shown in the paper, here we focus on discrete version）
- When add non-explanation edges to GT,our fidelity+ decreases while our fidelity-  stay at a certain level.(first row)
-  When remove edges from GT, our fidelity+ stay at a certain level  while our fidelity-  increase.(first coloumn)
-  Under same sparsity, our fidelity have a strong correlation with AUC(diagonal line).
-  Fidelity_$\Delta$ has the maxmium at GT.

### I personally recommend use discrete version fidelity_k with k_plus=1, k_minus=1. It has a linear time complexity when k_plus = k_minus=1. It is consistent with AUC score under same sparsity. The ACC Fidelity is consistent with Prob. Fidelity.



### If this work is helpful for you, please cite our paper.

### Acknowledge. This project is base on \[RE\]-PGExplainer [link](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks/blob/main/README.md)
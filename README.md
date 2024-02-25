# This Project is for our work [TOWARDS ROBUST FIDELITY FOR EVALUATING EXPLAINABILITY OF GRAPH NEURAL NETWORKS(ICLR2024)](https://openreview.net/pdf?id=up6hr4hIQH)


## use fidelity as metric
- If you just want to use our fidelity for evaluation, please use tools folder.
- Please refer to example.py

## for results reproduce
- generate samples, please run generate_edit_distance.py
- generate ori fidelity results, please run experiment_editdistance_ori_fid.py
- generate our fidelity results, please run experiment_editdistance_new_fid.py

<!-- ## Continuous Version -->

#### Probability ori. Fidelity results of Ba2Motifs dataset(ACC results can be found in ./pictures), the x-axis means adding non-explanation edges to GT, y-axis means remove edges from GT. The following three figures are Original Fidelity+,Fidelity-,Fidelity$_\Delta$ $\Delta$,$\delta$.

[//]: # ($\Delta$)

[//]: # (![alt-text-1]&#40;pictures/GNN_ba2_results_ori_fid_1fid_plus prob.png "title-1"&#41; ![alt-text-2]&#40;pictures/GNN_ba2_results_ori_fid_1fid_minus prob.png "title-2"&#41;![alt-text-2]&#40;pictures/GNN_ba2_results_ori_fid_1fid_Delta prob.png "title-2"&#41;)

<center class="ba2">

[//]: # (<img src="./pictures/GNN_ba2_results_ori_fid_1fid_plus prob.png"  width = "33%" alt="" align=center /> )

[//]: # (<img src="./pictures/GNN_ba2_results_ori_fid_1fid_minus prob.png" width = "33%" alt="" align=center />)

[//]: # (<img src="./pictures/GNN_ba2_results_ori_fid_1fid_Delta prob.png" width = "33%" alt="" align=center />)

[//]: # (<br><br>)

[//]: # (<b>Figure 1. </b> Original Fidelity+,Fidelity-,Fidelity_$\Delta$.)
[//]: # (  <tr>)

[//]: # (    <td>Original Fidelity+</td>)

[//]: # (     <td>Original Fidelity-</td>)

[//]: # (     <td>Fidelity_$\Delta$</td>)

[//]: # (  </tr>)

<table>
  <tr>
    <td><img src="./pictures/GNN_ba2_results_ori_fid_1fid_plus prob.png"  width = "100%" alt="" align=center /> </td>
    <td><img src="./pictures/GNN_ba2_results_ori_fid_1fid_minus prob.png"  width = "100%" alt="" align=center /></td>
    <td><img src="./pictures/GNN_ba2_results_ori_fid_1fid_Delta prob.png"  width = "100%" alt="" align=center /></td>
  </tr>
 </table>

</center>



#### Probability our Fidelity results of Ba2Motifs dataset($ \Alpha_1 $ = 0.1,$\Alpha_2 = 0.9$)(ACC results can be found in ./pictures)
<center class="ba2">
<img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_plus prob.png" width = "200" alt="" align=center />
<img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_minus prob.png" width = "200" alt="" align=center />
<img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_Delta prob.png" width = "200" alt="" align=center />
<br><br>
<b>Figure 2. </b> Ours Fidelity+,Fidelity-,Fidelity_$\Delta$.
</center>

#### Probability ori. Fidelity results of TreeCycles dataset(ACC results can be found in ./pictures), the x-axis means adding non-explanation edges to GT, y-axis means remove edges from GT.
<center class="ba2">
<img src="./pictures/GNN_syn3_results_ori_fid_1fid_plus prob.png"  width = "200" alt="" align=center />
<img src="./pictures/GNN_syn3_results_ori_fid_1fid_minus prob.png" width = "200" alt="" align=center />
<img src="./pictures/GNN_syn3_results_ori_fid_1fid_Delta prob.png" width = "200" alt="" align=center />
<br><br>
<b>Figure 3. </b> Original Fidelity+,Fidelity-,Fidelity_$\Delta$.
</center>

#### Probability our Fidelity results of TreeCycles dataset($\Alpha_1 = 0.1$,$\Alpha_2 = 0.9$)(ACC results can be found in ./pictures)
<center class="ba2">
<img src="./pictures/GNN_syn3_results_new_fid_0_0_seeds_1_fid_plus prob.png" width = "200" alt="" align=center />
<img src="./pictures/GNN_syn3_results_new_fid_0_0_seeds_1_fid_minus prob.png" width = "200" alt="" align=center />
<img src="./pictures/GNN_syn3_results_new_fid_0_0_seeds_1_fid_Delta prob.png" width = "200" alt="" align=center />
<br><br>
<b>Figure 4. </b> Ours Fidelity+,Fidelity-,Fidelity_$\Delta$.
</center>


### Acknowledge. This project is base on \[RE\]-PGExplainer [link](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks/blob/main/README.md)

### If this work is helpful for you, please consider citing our paper.

```angular2html
@misc{zheng2023robust,
      title={Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks}, 
      author={Xu Zheng and Farhad Shirani and Tianchun Wang and Wei Cheng and Zhuomin Chen and Haifeng Chen and Hua Wei and Dongsheng Luo},
      year={2023},
      eprint={2310.01820},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# This Project is for our work, $Fidelity_\alpha$ , [TOWARDS ROBUST FIDELITY FOR EVALUATING EXPLAINABILITY OF GRAPH NEURAL NETWORKS(ICLR2024)](https://openreview.net/pdf?id=up6hr4hIQH)


## use fidelity as metric
- If you just want to use our fidelity for evaluation, please use tools folder.
- Please refer to example.py

## for results reproduce
- generate samples, please run generate_edit_distance.py
- generate ori fidelity results, please run experiment_editdistance_ori_fid.py
- generate our fidelity results, please run experiment_editdistance_new_fid.py

<!-- ## Continuous Version -->

#### Probability ori. Fidelity results of Ba2Motifs dataset(ACC results can be found in ./pictures), the x-axis means adding non-explanation edges to GT, y-axis means remove edges from GT. The following three figures are Original $Fidelity+$, $Fidelity-$, $Fidelity_\Delta$.

<center class="ba2">
<table>
  <tr>
    <td><img src="./pictures/GNN_ba2_results_ori_fid_1fid_plus prob.png"  width = "100%" alt="" align=center /> </td>
    <td><img src="./pictures/GNN_ba2_results_ori_fid_1fid_minus prob.png"  width = "100%" alt="" align=center /></td>
    <td><img src="./pictures/GNN_ba2_results_ori_fid_1fid_Delta prob.png"  width = "100%" alt="" align=center /></td>
  </tr>
 </table>

</center>



#### Probability our Fidelity results of Ba2Motifs dataset( $\alpha_1$ = 0.1, $\alpha_2$ = 0.9 )(ACC results can be found in ./pictures). The following three figures are Ours $Fidelity+$, $Fidelity-$, $Fidelity_\Delta$.
<center class="ba2">
<table>
  <tr>
    <td><img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_plus prob.png"  width = "100%" alt="" align=center /> </td>
    <td><img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_minus prob.png"  width = "100%" alt="" align=center /></td>
    <td><img src="./pictures/GNN_ba2_results_new_fid_0_0_seeds_1_fid_Delta prob.png"  width = "100%" alt="" align=center /></td>
  </tr>
 </table>
</center>

#### Probability ori. Fidelity results of TreeCycles dataset(ACC results can be found in ./pictures), the x-axis means adding non-explanation edges to GT, y-axis means remove edges from GT. The following three figures are Original $Fidelity+$, $Fidelity-$, $Fidelity_\Delta$.
<center class="ba2">
<table>
  <tr>
    <td><img src="./pictures/GNN_syn3_results_ori_fid_1fid_plus prob.png"  width = "100%" alt="" align=center /> </td>
    <td><img src="./pictures/GNN_syn3_results_ori_fid_1fid_minus prob.png"  width = "100%" alt="" align=center /></td>
    <td><img src="./pictures/GNN_syn3_results_ori_fid_1fid_Delta prob.png"  width = "100%" alt="" align=center /></td>
  </tr>
 </table>
</center>

#### Probability our Fidelity results of TreeCycles dataset( $\alpha_1$ = 0.1, $\alpha_2$ = 0.9 )(ACC results can be found in ./pictures). The following three figures are Ours $Fidelity+$, $Fidelity-$, $Fidelity_\Delta$.
<center class="ba2">
<table>
  <tr>
    <td><img src="./pictures/GNN_syn3_results_new_fid_0_0_seeds_1_fid_plus prob.png"  width = "100%" alt="" align=center /> </td>
    <td><img src="./pictures/GNN_syn3_results_new_fid_0_0_seeds_1_fid_minus prob.png"  width = "100%" alt="" align=center /></td>
    <td><img src="./pictures/GNN_syn3_results_new_fid_0_0_seeds_1_fid_Delta prob.png"  width = "100%" alt="" align=center /></td>
  </tr>
 </table>
</center>


### Acknowledge. This project is base on \[RE\]-PGExplainer [link](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks/blob/main/README.md)

### If this work is helpful for you, please consider citing our paper.

```angular2html
@article{zheng2023towards,
  title={Towards robust fidelity for evaluating explainability of graph neural networks},
  author={Zheng, Xu and Shirani, Farhad and Wang, Tianchun and Cheng, Wei and Chen, Zhuomin and Chen, Haifeng and Wei, Hua and Luo, Dongsheng},
  journal={arXiv preprint arXiv:2310.01820},
  year={2023}
}
```





# AGCN_Event_Classification 
Official PyTorch implementation of "**Point-Voxel Absorbing Graph Representation Learning for Event Stream based Recognition**". 


## Abstract 
Sampled point and voxel methods are usually employed to downsample the dense events into sparse ones. After that, one popular way is to leverage a graph model which treats the sparse points/voxels as nodes and adopts graph neural networks (GNNs) to learn the representation of event data. Although good performance can be obtained, however, their results are still limited mainly due to two issues. (1) Existing event GNNs generally adopt the additional max (or mean) pooling layer to summarize all node embeddings into a single graph-level representation for the whole event data representation. However, this approach fails to capture the importance of graph nodes and also fails to be fully aware of the node representations. (2) Existing methods generally employ either a sparse point or voxel graph representation model which thus lacks consideration of the complementary between these two types of representation models. To address these issues, in this paper, we propose a novel dual point-voxel absorbing graph representation learning for event stream data representation. To be specific, given the input event stream, we first transform it into the sparse event cloud and voxel grids and build dual absorbing graph models for them respectively. Then, we design a novel absorbing graph convolutional network (AGCN) for our dual absorbing graph representation and learning. The key aspect of the proposed AGCN is its ability to effectively capture the importance of nodes and thus be fully aware of node representations in summarizing all node representations through the introduced absorbing nodes. Finally, the event representations of dual learning branches are concatenated together to extract the complementary information of two cues. The output is then fed into a linear layer for event data classification. Extensive experiments on multiple event-based classification benchmark datasets fully validated the effectiveness of our framework.


## Our Proposed Framework 

<p align="center">
  <img width="85%" src="https://github.com/Event-AHU/AGCN_Event_Classification/blob/main/figure/framework.jpg" alt="Framework"/>
</p> 

## Environment Setting 


## Dataset Download and Pre-processing 

## Training and Testing 


## Experimental Results 

<p align="center">
  <img width="85%" src="https://github.com/Event-AHU/AGCN_Event_Classification/blob/main/figure/feature_vis.jpg" alt="feature_vis"/>
</p> 


<p align="center">
  <img width="85%" src="https://github.com/Event-AHU/AGCN_Event_Classification/blob/main/figure/top5_hardvs.jpg" alt="top5_hardvs"/>
</p> 

## Acknowledgement 


## Citation 
```bibtex
@article{jiang2023eventAGCN,
  title={Point-Voxel Absorbing Graph Representation Learning for Event Stream based Recognition},
  author={Jiang, Bo and Yuan, Chengguo and Wang, Xiao and Bao, Zhimin and Zhu, Lin and Tian, Yonghong and Tang, Jin},
  journal={arXiv preprint arXiv:2306.05239},
  year={2023}
}
```


if you have any problems with this work, please leave an issue. 

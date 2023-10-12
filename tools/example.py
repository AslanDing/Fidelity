import os
import torch

# nodes
model = 'GNN'
dataset = 'syn3'
graphs, features, labels, _, _, test_mask = load_dataset(dataset)
model, checkpoint = model_selector(model,
                                   dataset,
                                   pretrained=True,
                                   return_checkpoint=True)
explanation_labels, indices = load_dataset_ground_truth(dataset)



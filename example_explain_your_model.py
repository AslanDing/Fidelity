import torch

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

graphs, features, labels, _, _, _ = load_dataset('syn1')
graphs = torch.tensor(graphs)
features = torch.tensor(features)
labels = torch.tensor(labels)


# Overwrite these models with your own if you want to
from ExplanationEvaluation.models.GNN_paper import NodeGCN
model = NodeGCN(10, 4)

path = "./ExplanationEvaluation/models/pretrained/GNN/syn1/best_model"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])


# task = 'graph'
task = 'node'


from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
explainer = PGExplainer(model, graphs, features, task)



# We use the same indices as used in the original paper
indices = range(400, 700, 5)
explainer.prepare(indices)


idx = indices[0] # select a node to explain, this needs to be part of the list of indices
graph, expl = explainer.explain(idx)

from ExplanationEvaluation.utils.plotting import plot
plot(graph, expl, labels, 400, 12, 100, 'syn1', show=True)


from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
explanation_labels, indices = load_dataset_ground_truth('syn1')



from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.EfficiencyEvaluation import EfficiencyEvluation

auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
inference_eval = EfficiencyEvluation()



from ExplanationEvaluation.tasks.replication import run_experiment

auc, time = run_experiment(inference_eval, auc_evaluation, explainer, indices)



print(auc)
print(time)


for idx in indices:
    graph, expl = explainer.explain(idx)
    plot(graph, expl, labels, idx, 12, 100, 'syn1', show=True)


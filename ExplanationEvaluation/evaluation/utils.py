import numpy as np
from sklearn.metrics import roc_auc_score


def evaluation_auc(task, explanations, explanation_labels, indices):
    """Determines based on the task which auc evaluation method should be called to determine the AUC score

    :param task: str either "node" or "graph".
    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    if task == 'graph':
        return evaluation_auc_graph(explanations, explanation_labels, indices)
    elif task == 'node':
        return evaluation_auc_node(explanations, explanation_labels)

def evaluation_iou(task, explanations, explanation_labels, indices):
    """Determines based on the task which auc evaluation method should be called to determine the AUC score

    :param task: str either "node" or "graph".
    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    if task == 'graph':
        return evaluation_iou_graph(explanations, explanation_labels, indices)
    elif task == 'node':
        return evaluation_iou_node(explanations, explanation_labels)


def evaluation_auc_graph(explanations, explanation_labels, indices):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []

    for idx, n in enumerate(indices): # Use idx for explanation list and indices for ground truth list

        # Select explanation
        mask = explanations[idx][1].detach().cpu().numpy()
        graph = explanations[idx][0].detach().cpu().numpy()

        # Select ground truths
        edge_list = explanation_labels[0][n]
        edge_labels = explanation_labels[1][n]

        for edge_idx in range(0, edge_labels.shape[0]): # Consider every edge in the ground truth
            edge_ = edge_list.T[edge_idx]
            if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                continue
            t = np.where((graph.T == edge_.T).all(axis=1)) # Determine index of edge in graph
            ################
            if len(t[0])==0:
                continue
            ###############
            # Retrieve predictions and ground truth
            predictions.append(mask[t][0])
            ground_truth.append(edge_labels[edge_idx])

    score = roc_auc_score(ground_truth, predictions)
    return score


def evaluation_auc_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node

        ground_truth_node = []
        prediction_node = []

        # node classification, ground-truth is not a bi-graph
        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i].cpu().numpy()
            idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] #+ explanation_labels[1][idx_edge_rev]
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)


        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score


def evaluation_iou_graph(explanations, explanation_labels, indices):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: IOU, auc
    """
    ground_truth = []
    predictions = []
    iou_lists = []
    for idx, n in enumerate(indices): # Use idx for explanation list and indices for ground truth list

        # Select explanation
        mask = explanations[idx][1].detach().cpu().numpy()
        graph = explanations[idx][0].detach().cpu().numpy()

        # Select ground truths
        edge_list = explanation_labels[0][n]
        edge_labels = explanation_labels[1][n]

        interset = []
        unionset = []
        for edge_idx in range(0, edge_labels.shape[0]): # Consider every edge in the ground truth
            edge_ = edge_list.T[edge_idx]
            if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                continue
            t = np.where((graph.T == edge_.T).all(axis=1)) # Determine index of edge in graph
            ################
            if len(t[0])==0:
                continue
            ###############
            # Retrieve predictions and ground truth
            if mask[t][0]>0.5 and edge_labels[edge_idx]>0.5 :
                interset.append(edge_idx)
                unionset.append(edge_idx)
            elif mask[t][0]>0.5 or edge_labels[edge_idx]>0.5 :
                unionset.append(edge_idx)

            predictions.append(mask[t][0])
            ground_truth.append(edge_labels[edge_idx])
        iou = len(interset)/len(unionset)
        iou_lists.append(iou)
    iou_mean = np.array(iou_lists).mean()
    score = roc_auc_score(ground_truth, predictions)
    return iou_mean , score


def evaluation_iou_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: iou auc
    """
    ground_truth = []
    predictions = []

    iou_lists = []

    for expl in explanations: # Loop over the explanations for each node

        ground_truth_node = []
        prediction_node = []

        interset = []
        unionset = []

        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i].cpu().numpy()
            idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

            if expl[1][i].item() > 0.5 and gt > 0.5:
                interset.append(i)
                unionset.append(i)
            elif expl[1][i].item() > 0.5 or gt > 0.5:
                unionset.append(i)

        iou = len(interset) / len(unionset)
        iou_lists.append(iou)

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    iou_mean = np.array(iou_lists).mean()
    score = roc_auc_score(ground_truth, predictions)
    return iou_mean ,score


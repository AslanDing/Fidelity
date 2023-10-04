import torch
import os

from ExplanationEvaluation.models.GNN_paper import NodeGCN as GNN_NodeGCN
from ExplanationEvaluation.models.GNN_paper import GraphGCN as GNN_GraphGCN
from ExplanationEvaluation.models.PG_paper import NodeGCN as PG_NodeGCN
from ExplanationEvaluation.models.PG_paper import GraphGCN as PG_GraphGCN

def string_to_model(paper, dataset):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    if paper == "GNN":
        if dataset in ['syn1']:
            return GNN_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return GNN_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return GNN_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return GNN_NodeGCN(10, 2)
        elif dataset == "ba2":
            return GNN_GraphGCN(10, 2)
        elif dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "PG":
        if dataset in ['syn1']:
            return PG_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return PG_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return PG_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return PG_NodeGCN(10, 2)
        elif dataset == "ba2":
            return PG_GraphGCN(10, 2)
        elif dataset == "mutag":
            return PG_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_pretrained_path(paper, dataset):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: str; the path to the pre-trined model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = f"{dir_path}/pretrained/{paper}/{dataset}/best_model"
    return path

def get_pretrained_path_extend(paper, dataset):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: str; the path to the pre-trined model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = f"{dir_path}/../../checkpoints/{paper}_extend/{dataset}/best_model"
    return path
def get_pretrained_path_contrastive(paper, dataset):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: str; the path to the pre-trined model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = f"{dir_path}/../../checkpoints/{paper}_contrastive/{dataset}/best_model"
    return path

def model_selector(paper, dataset, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path(paper, dataset)
        checkpoint = torch.load(path)

        print(checkpoint['model_state_dict'].keys())
        print()
        new_dict = {}
        for key in checkpoint['model_state_dict'].keys():
            new_key = key
            if 'weight' in key and 'conv' in key:
                str_list = key.split('.')
                new_key = str_list[0]+'.lin.'+str_list[1]
                new_dict[new_key] = checkpoint['model_state_dict'][key].T
            else:
                new_dict[new_key] = checkpoint['model_state_dict'][key]

        # print(model.state_dict().keys())

        model.load_state_dict(new_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model


def model_selector_extend(paper, dataset, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path_extend(paper, dataset)
        checkpoint = torch.load(path)

        # print(checkpoint['model_state_dict'].keys())
        # print()
        # new_dict = {}
        # for key in checkpoint['model_state_dict'].keys():
        #     new_key = key
        #     if 'weight' in key and 'conv' in key:
        #         str_list = key.split('.')
        #         new_key = str_list[0]+'.lin.'+str_list[1]
        #         new_dict[new_key] = checkpoint['model_state_dict'][key].T
        #     else:
        #         new_dict[new_key] = checkpoint['model_state_dict'][key]

        # print(model.state_dict().keys())

        model.load_state_dict(checkpoint['model_state_dict'])
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model

def model_selector_contrastive(paper, dataset, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path_contrastive(paper, dataset)
        checkpoint = torch.load(path)

        # print(checkpoint['model_state_dict'].keys())
        # print()
        # new_dict = {}
        # for key in checkpoint['model_state_dict'].keys():
        #     new_key = key
        #     if 'weight' in key and 'conv' in key:
        #         str_list = key.split('.')
        #         new_key = str_list[0]+'.lin.'+str_list[1]
        #         new_dict[new_key] = checkpoint['model_state_dict'][key].T
        #     else:
        #         new_dict[new_key] = checkpoint['model_state_dict'][key]

        # print(model.state_dict().keys())

        model.load_state_dict(checkpoint['model_state_dict'])
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model

"""Hierarchical loss for the labels suggested in https://arxiv.org/abs/2210.10929"""


__cmd_doc__ = """Hierarchical loss for the labels"""


import networkx as nx
import numpy as _np
from collections import namedtuple
from functools import partial
from math import log as _log
from pathlib import Path
from typing import Optional, IO, Union

import torch as _torch
from torch import nn as _nn
from torch.utils.data.dataset import TensorDataset as _TensorDataset
from torch.utils.data import DataLoader as _DataLoader
from torch.optim import Adam as _Adam
from torch import Tensor

import vamb.semisupervised_encode as _semisupervised_encode
import vamb.encode as _encode
import vamb.hlosses_fast as _hlosses_fast
import vamb.hier as _hier
import vamb.infer as _infer

if _torch.__version__ < "0.4":
    raise ImportError("PyTorch version must be 0.4 or newer")


def make_graph(taxs):
    table_parent = []
    G = nx.DiGraph()
    G_op = nx.Graph()
    root = "Domain"
    G.add_edge(root, "d_Archaea")
    G_op.add_edge(root, "d_Archaea")
    G.add_edge(root, "d_Bacteria")
    G_op.add_edge(root, "d_Bacteria")
    for i, row in enumerate(taxs):
        try:
            r = row.split(";")
        except AttributeError:  # ignore non-strings in the input
            continue
        if len(r) == 1:
            continue
        for j in range(1, len(r)):
            G.add_edge(r[j - 1], r[j])
            G_op.add_edge(r[j - 1], r[j])
    edges = nx.bfs_edges(G, root)
    nodes = [root] + [v for u, v in edges]
    ind_nodes = {v: i for i, v in enumerate(nodes)}
    for n in nodes:
        if n == root:
            table_parent.append(-1)
        else:
            table_parent.append(ind_nodes[list(G.predecessors(n))[0]])
    return nodes, ind_nodes, table_parent


def collate_fn_labels_hloss(
    num_categories, table_parent, labels
):  # target in form of indices
    return _semisupervised_encode.collate_fn_labels(num_categories, labels)


def collate_fn_concat_hloss(num_categories, table_parent, batch):
    a = _torch.stack([i[0] for i in batch])
    b = _torch.stack([i[1] for i in batch])
    c = _torch.stack([i[2] for i in batch])
    d = [(i[3],) for i in batch]
    return a, b, c, collate_fn_labels_hloss(num_categories, table_parent, d)[0]


def collate_fn_semisupervised_hloss(num_categories, table_parent, batch):
    a = _torch.stack([i[0] for i in batch])
    b = _torch.stack([i[1] for i in batch])
    c = _torch.stack([i[2] for i in batch])
    d = [i[3] for i in batch]
    e = _torch.stack([i[4] for i in batch])
    f = _torch.stack([i[5] for i in batch])
    g = _torch.stack([i[6] for i in batch])
    h = [i[7] for i in batch]
    return (
        a,
        b,
        c,
        collate_fn_labels_hloss(num_categories, table_parent, d)[0],
        e,
        f,
        g,
        collate_fn_labels_hloss(num_categories, table_parent, h)[0],
    )


def make_dataloader_labels_hloss(
    rpkm,
    tnf,
    lengths,
    labels,
    N,
    table_parent,
    batchsize=256,
    destroy=False,
    cuda=False,
):
    _, _, _, batchsize, n_workers, cuda, mask = _semisupervised_encode._make_dataset(
        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
    )
    dataset = _TensorDataset(_torch.Tensor(labels).long())
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_labels_hloss, N, table_parent),
    )

    return dataloader, mask


def make_dataloader_concat_hloss(
    rpkm,
    tnf,
    lengths,
    labels,
    N,
    table_parent,
    no_filter=True,
    batchsize=256,
    destroy=False,
    cuda=False,
):
    (
        depthstensor,
        tnftensor,
        weightstensor,
        batchsize,
        n_workers,
        cuda,
        mask,
    ) = _semisupervised_encode._make_dataset(
        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
    )
    labelstensor = _torch.Tensor(labels).long()[mask]
    dataset = _TensorDataset(
        depthstensor, tnftensor, weightstensor, labelstensor
    )
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_concat_hloss, N, table_parent),
    )
    return dataloader, mask


def permute_indices(n_current: int, n_total: int, seed:int):
    rng = _np.random.default_rng(seed)
    x = _np.arange(n_current)
    to_add = int(n_total / n_current)
    to_concatenate = []
    for _ in range(to_add):
        a = rng.permutation(x)
        b = rng.permutation(x)
        to_concatenate.append(a + b)
    return _np.concatenate(to_concatenate)[:n_total]


def make_dataloader_semisupervised_hloss(
    dataloader_joint,
    dataloader_vamb,
    dataloader_labels,
    N,
    table_parent,
    shapes,
    seed: int,
    batchsize=256,
    destroy=False,
    cuda=False,
):
    n_total = len(dataloader_vamb.dataset)
    indices_unsup_vamb = permute_indices(len(dataloader_vamb.dataset), n_total, seed)
    indices_unsup_labels = permute_indices(len(dataloader_labels.dataset), n_total, seed)
    indices_sup = permute_indices(len(dataloader_joint.dataset), n_total, seed)
    dataset_all = _TensorDataset(
        dataloader_vamb.dataset.tensors[0][indices_unsup_vamb],
        dataloader_vamb.dataset.tensors[1][indices_unsup_vamb],
        dataloader_vamb.dataset.tensors[2][indices_unsup_vamb],
        dataloader_labels.dataset.tensors[0][indices_unsup_labels],
        dataloader_joint.dataset.tensors[0][indices_sup],
        dataloader_joint.dataset.tensors[1][indices_sup],
        dataloader_joint.dataset.tensors[2][indices_sup],
        dataloader_joint.dataset.tensors[3][indices_sup],
    )
    dataloader_all = _DataLoader(
        dataset=dataset_all,
        batch_size=batchsize,
        drop_last=True,
        shuffle=False,
        num_workers=dataloader_joint.num_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_semisupervised_hloss, N, table_parent),
    )
    return dataloader_all


LabelMap = namedtuple("LabelMap", ["to_node", "to_target"])


class VAELabelsHLoss(_semisupervised_encode.VAELabels):
    """Variational autoencoder that encodes only the labels.
    Subclass of one-hot VAELabels class but the labels are structured differently.
    Uses hierarchical loss to utilize the taxonomic tree.
    Instantiate with:
        nlabels: Number of labels

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        - Hyperparameters:
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nlabels,
        nodes,
        table_parent,
        nhiddens=None,
        nlatent=32,
        alpha=None,
        beta=200,
        dropout=0.2,
        cuda=False,
        logfile=None,
    ):
        super(VAELabelsHLoss, self).__init__(
            nlabels,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.nlabels = nlabels
        self.nodes = nodes
        self.logfile = logfile
        self.table_parent = table_parent
        self.tree = _hier.Hierarchy(table_parent)
        self.loss_fn = _hlosses_fast.HierSoftmaxCrossEntropy(self.tree)
        self.pred_helper = _hlosses_fast.HierLogSoftmax(self.tree)
        if self.usecuda:
            self.loss_fn = self.loss_fn.cuda()
            self.pred_helper = self.pred_helper.cuda()
        self.pred_fn = partial(
            lambda log_softmax_fn, theta: log_softmax_fn(theta).exp(),
            self.pred_helper,
        )
        self.specificity = -self.tree.num_leaf_descendants()
        self.not_trivial = self.tree.num_children() != 1
        self.find_lca = _hier.FindLCA(self.tree)
        node_subset = _hier.find_subset_index(self.nodes, self.nodes)
        project_to_subtree = _hier.find_projection(self.tree, node_subset)
        label_to_node = self.tree.parents()
        label_to_subtree_node = project_to_subtree[label_to_node]
        self.eval_label_map = LabelMap(
            to_node=label_to_subtree_node,
            to_target=label_to_subtree_node,
        )

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        ce_labels = self.loss_fn(labels_out[:, 1:], labels_in)

        ce_labels_weight = 1.0  # TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce_labels * ce_labels_weight + kld * kld_weight

        _, labels_in_indices = labels_in.max(dim=1)
        with _torch.no_grad():
            gt_node = self.eval_label_map.to_node[labels_in_indices.cpu()]
            prob = self.pred_fn(labels_out[:, 1:])
        prob = prob.cpu().numpy()
        pred = _infer.argmax_with_confidence(
            self.specificity, prob, 0.5, self.not_trivial
        )
        pred = _hier.truncate_given_lca(gt_node, pred, self.find_lca(gt_node, pred))

        return loss, ce_labels, kld, _torch.tensor(_np.sum(gt_node == pred))

    def trainmodel(
        self,
        dataloader: _DataLoader[tuple[Tensor]],
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 75, 150, 300],
        logfile: Optional[IO[str]] = None,
        modelfile: Union[None, str, Path, IO[bytes]] = None,
    ):
        if lrate < 0:
            raise ValueError(f"Learning rate must be positive, not {lrate}")

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {nepochs}")

        if batchsteps is None:
            batchsteps_set: set[int] = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError("All elements of batchsteps must be integers")
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError("Max batchsteps must not equal or exceed nepochs")
            last_batchsize = dataloader.batch_size * 2 ** len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:  # type: ignore
                raise ValueError(
                    f"Last batch size of {last_batchsize} exceeds dataset length "
                    f"of {len(dataloader.dataset)}. "  # type: ignore
                    "This means you have too few contigs left after filtering to train. "
                    "It is not adviced to run Vamb with fewer than 10,000 sequences "
                    "after filtering. "
                    "Please check the Vamb log file to see where the sequences were "
                    "filtered away, and verify BAM files has sensible content."
                )
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        nlabels = dataloader.dataset.tensors[0].shape  # type: ignore
        optimizer = _Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tBeta:", self.beta, file=logfile)
            print("\tDropout:", self.dropout, file=logfile)
            print("\tN hidden:", ", ".join(map(str, self.nhiddens)), file=logfile)
            print("\tN latent:", self.nlatent, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            print("\tStarting batch size:", dataloader.batch_size, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps_set)))
                if batchsteps_set
                else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tLearning rate:", lrate, file=logfile)
            print("\tN labels:", nlabels, file=logfile, end="\n\n")

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(
                dataloader, epoch, optimizer, sorted(batchsteps_set), logfile
            )

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None


class VAEConcatHLoss(_semisupervised_encode.VAEConcat):
    """Variational autoencoder that encodes the concatenated input of VAMB and labels.
    Subclass of one-hot VAEConcat class but the labels are structured differently.
    Uses hierarchical loss to utilize the taxonomic tree.
    Instantiate with:
        nlabels: Number of labels

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        - Hyperparameters:
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples,
        nlabels,
        nodes,
        table_parent,
        nhiddens=None,
        nlatent=32,
        alpha=None,
        beta=200,
        dropout=0.2,
        cuda=False,
        logfile=None,
    ):
        super(VAEConcatHLoss, self).__init__(
            nsamples,
            nlabels,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.nsamples = nsamples
        self.nlabels = nlabels
        self.nodes = nodes
        self.logfile = logfile
        self.table_parent = table_parent
        self.tree = _hier.Hierarchy(table_parent)
        self.loss_fn = _hlosses_fast.HierSoftmaxCrossEntropy(self.tree)
        self.pred_helper = _hlosses_fast.HierLogSoftmax(self.tree)
        if self.usecuda:
            self.loss_fn = self.loss_fn.cuda()
            self.pred_helper = self.pred_helper.cuda()
        self.pred_fn = partial(
            lambda log_softmax_fn, theta: log_softmax_fn(theta).exp(),
            self.pred_helper,
        )
        self.specificity = -self.tree.num_leaf_descendants()
        self.not_trivial = self.tree.num_children() != 1
        self.find_lca = _hier.FindLCA(self.tree)
        node_subset = _hier.find_subset_index(self.nodes, self.nodes)
        project_to_subtree = _hier.find_projection(self.tree, node_subset)
        label_to_node = self.tree.parents()
        label_to_subtree_node = project_to_subtree[label_to_node]
        self.eval_label_map = LabelMap(
            to_node=label_to_subtree_node,
            to_target=label_to_subtree_node,
        )

    def calc_loss(
        self,
        depths_in,
        depths_out,
        tnf_in,
        tnf_out,
        labels_in,
        labels_out,
        mu,
        logsigma,
        weights,
    ):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        ce_labels = self.loss_fn(labels_out[:, 1:], labels_in)

        ce_labels_weight = 1.0
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = (
            ce * ce_weight
            + sse * sse_weight
            + ce_labels * ce_labels_weight
            + kld * kld_weight
        ) * weights

        _, labels_in_indices = labels_in.max(dim=1)
        with _torch.no_grad():
            gt_node = self.eval_label_map.to_node[labels_in_indices.cpu()]
            prob = self.pred_fn(labels_out[:, 1:])
        prob = prob.cpu().numpy()
        pred = _infer.argmax_with_confidence(
            self.specificity, prob, 0.5, self.not_trivial
        )
        pred = _hier.truncate_given_lca(gt_node, pred, self.find_lca(gt_node, pred))
        return loss, ce, sse, ce_labels, kld, _torch.tensor(_np.sum(gt_node == pred))


class VAEVAEHLoss(_semisupervised_encode.VAEVAE):
    """Bi-modal variational autoencoder that uses TNFs, abundances and labels.
    It contains three individual encoders (VAMB, labels and concatenated VAMB+labels)
    and two decoders (VAMB and labels).
    Subclass of one-hot VAEVAE class but the labels are structured differently.
    Uses hierarchical loss to utilize the taxonomic tree.
    Instantiate with:
        nlabels: Number of labels

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        - Hyperparameters:
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples,
        nlabels,
        nodes,
        table_parent,
        nhiddens=None,
        nlatent=32,
        alpha=None,
        beta=200,
        dropout=0.2,
        cuda=False,
        logfile=None,
    ):
        N_l = max(nlabels, 105)
        self.VAEVamb = _encode.VAE(
            nsamples,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.VAELabels = VAELabelsHLoss(
            N_l,
            nodes,
            table_parent,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
            logfile=logfile,
        )
        self.VAEJoint = VAEConcatHLoss(
            nsamples,
            N_l,
            nodes,
            table_parent,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
            logfile=logfile,
        )

    @classmethod
    def load(cls, path, nodes, table_parent, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.
        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]
        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = _torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary["nsamples"]
        nlabels = dictionary["nlabels"]
        alpha = dictionary["alpha"]
        beta = dictionary["beta"]
        dropout = dictionary["dropout"]
        nhiddens = dictionary["nhiddens"]
        nlatent = dictionary["nlatent"]

        vae = cls(
            nsamples,
            nlabels,
            nodes,
            table_parent,
            nhiddens,
            nlatent,
            alpha,
            beta,
            dropout,
            cuda,
        )
        vae.VAEVamb.load_state_dict(dictionary["state_VAEVamb"])
        vae.VAELabels.load_state_dict(dictionary["state_VAELabels"])
        vae.VAEJoint.load_state_dict(dictionary["state_VAEJoint"])

        if cuda:
            vae.VAEVamb.cuda()
            vae.VAELabels.cuda()
            vae.VAEJoint.cuda()

        if evaluate:
            vae.VAEVamb.eval()
            vae.VAELabels.eval()
            vae.VAEJoint.eval()

        return vae


class VAMB2Label(_nn.Module):
    """Label predictor, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        nhiddens: list of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        nlabels: int,
        nodes,
        table_parent,
        nhiddens: Optional[list[int]] = None,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
    ):
        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError(f"Minimum 1 neuron per layer, not {min(nhiddens)}")

        if beta <= 0:
            raise ValueError(f"beta must be > 0, not {beta}")

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be 0 < alpha < 1, not {alpha}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be 0 <= dropout < 1, not {dropout}")

        super(VAMB2Label, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlabels = nlabels
        self.dropout = dropout

        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip(
            [self.nsamples + self.ntnf] + self.nhiddens, self.nhiddens
        ):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nlabels)
        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

        self.nlabels = nlabels
        self.nodes = nodes
        self.table_parent = table_parent
        self.tree = _hier.Hierarchy(table_parent)
        self.loss_fn = _hlosses_fast.HierSoftmaxCrossEntropy(self.tree)
        self.pred_helper = _hlosses_fast.HierLogSoftmax(self.tree)
        if self.usecuda:
            self.loss_fn = self.loss_fn.cuda()
            self.pred_helper = self.pred_helper.cuda()
        self.pred_fn = partial(
            lambda log_softmax_fn, theta: log_softmax_fn(theta).exp(),
            self.pred_helper,
        )
        self.specificity = -self.tree.num_leaf_descendants()
        self.not_trivial = self.tree.num_children() != 1
        self.find_lca = _hier.FindLCA(self.tree)
        node_subset = _hier.find_subset_index(self.nodes, self.nodes)
        project_to_subtree = _hier.find_projection(self.tree, node_subset)
        label_to_node = self.tree.parents()
        label_to_subtree_node = project_to_subtree[label_to_node]
        self.eval_label_map = LabelMap(
            to_node=label_to_subtree_node,
            to_target=label_to_subtree_node,
        )

    def _predict(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)
        labels_out = reconstruction.narrow(1, 0, self.nlabels)
        return labels_out

    def forward(self, depths, tnf, weights):
        tensor = _torch.cat((depths, tnf), 1)
        labels_out = self._predict(tensor)
        return labels_out

    def calc_loss(self, labels_in, labels_out):
        ce_labels = self.loss_fn(labels_out[:, 1:], labels_in)

        _, labels_in_indices = labels_in.max(dim=1)
        with _torch.no_grad():
            gt_node = self.eval_label_map.to_node[labels_in_indices.cpu()]
            prob = self.pred_fn(labels_out[:, 1:])
        prob = prob.cpu().numpy()
        pred = _infer.argmax_with_confidence(
            self.specificity, prob, 0.5, self.not_trivial
        )
        pred = _hier.truncate_given_lca(gt_node, pred, self.find_lca(gt_node, pred))

        return ce_labels, _torch.tensor(_np.sum(gt_node == pred))

    def predict(self, data_loader) -> _np.ndarray:
        self.eval()

        new_data_loader = _encode.set_batchsize(
            data_loader, data_loader.batch_size, encode=True
        )

        depths_array, _, _ = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlabels), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf, weights in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()
                    weights = weights.cuda()

                # Evaluate
                labels = self(depths, tnf, weights)
                with _torch.no_grad():
                    prob = self.pred_fn(labels[:, 1:])

                if self.usecuda:
                    prob = prob.cpu()

                latent[row : row + len(prob)] = prob
                row += len(prob)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {
            "nsamples": self.nsamples,
            "alpha": self.alpha,
            "beta": self.beta,
            "dropout": self.dropout,
            "nhiddens": self.nhiddens,
            "nlabels": self.nlabels,
            "state": self.state_dict(),
        }

        _torch.save(state, filehandle)

    @classmethod
    def load(
        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True
    ):
        raise NotImplementedError

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_celoss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(
                dataset=data_loader.dataset,
                batch_size=data_loader.batch_size * 2,
                shuffle=True,
                drop_last=True,
                num_workers=data_loader.num_workers,
                pin_memory=data_loader.pin_memory,
                collate_fn=data_loader.collate_fn,
            )

        for depths_in, tnf_in, weights, labels_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True
            labels_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()
                weights = weights.cuda()
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            labels_out = self(depths_in, tnf_in, weights)

            loss, correct_labels = self.calc_loss(labels_in, labels_out)

            loss.mean().backward()
            optimizer.step()

            epoch_celoss += loss.mean().data.item()
            epoch_correct_labels += correct_labels.data.item()

        if logfile is not None:
            print(
                "\tEpoch: {}\tCE: {:.7f}\taccuracy: {:.4f}\tBatchsize: {}".format(
                    epoch + 1,
                    epoch_celoss / len(data_loader),
                    epoch_correct_labels / len(data_loader),
                    data_loader.batch_size,
                ),
                file=logfile,
                flush=True,
            )

            logfile.flush()

        self.eval()
        return data_loader

    def trainmodel(
        self,
        dataloader: _DataLoader[tuple[Tensor]],
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 75, 150, 300],
        logfile: Optional[IO[str]] = None,
        modelfile: Union[None, str, Path, IO[bytes]] = None,
    ):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError(f"Learning rate must be positive, not {lrate}")

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {nepochs}")

        if batchsteps is None:
            batchsteps_set: set[int] = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError("All elements of batchsteps must be integers")
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError("Max batchsteps must not equal or exceed nepochs")
            last_batchsize = dataloader.batch_size * 2 ** len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:  # type: ignore
                raise ValueError(
                    f"Last batch size of {last_batchsize} exceeds dataset length "
                    f"of {len(dataloader.dataset)}. "  # type: ignore
                    "This means you have too few contigs left after filtering to train. "
                    "It is not adviced to run Vamb with fewer than 10,000 sequences "
                    "after filtering. "
                    "Please check the Vamb log file to see where the sequences were "
                    "filtered away, and verify BAM files has sensible content."
                )
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        nlabels = dataloader.dataset.tensors[0].shape  # type: ignore
        optimizer = _Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tBeta:", self.beta, file=logfile)
            print("\tDropout:", self.dropout, file=logfile)
            print("\tN hidden:", ", ".join(map(str, self.nhiddens)), file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            print("\tStarting batch size:", dataloader.batch_size, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps_set)))
                if batchsteps_set
                else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tLearning rate:", lrate, file=logfile)
            print("\tN labels:", nlabels, file=logfile, end="\n\n")

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(
                dataloader, epoch, optimizer, sorted(batchsteps_set), logfile
            )

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

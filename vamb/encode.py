__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader, mask = make_dataloader(depths, tnf)
>>> vae.trainmodel(dataloader)
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 32)
"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import numpy as _np
import torch as _torch
import torch.nn.functional as F
_torch.manual_seed(0)

from math import log as _log

from torch import nn as _nn
from torch.optim import Adam as _Adam
from torch.nn.functional import softmax as _softmax
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.sampler import SubsetRandomSampler as _SubsetRandomSampler
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import vamb.vambtools as _vambtools

if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')

def _make_dataset(rpkm, tnf, batchsize=256, destroy=False, cuda=False):
    """Helper function

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, _np.ndarray) or not isinstance(tnf, _np.ndarray):
        raise ValueError('TNF and RPKM must be Numpy arrays')

    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))

    if len(rpkm) != len(tnf):
        raise ValueError('Lengths of RPKM and TNF must be the same')

    if not (rpkm.dtype == tnf.dtype == _np.float32):
        raise ValueError('TNF and RPKM must be Numpy arrays of dtype float32')

    mask = tnf.sum(axis=1) != 0

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

    if mask.sum() < batchsize:
        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy:
        rpkm = _vambtools.numpy_inplace_maskarray(rpkm, mask)
        tnf = _vambtools.numpy_inplace_maskarray(tnf, mask)
    else:
        # The astype operation does not copy due to "copy=False", but the masking
        # operation does.
        rpkm = rpkm[mask].astype(_np.float32, copy=False)
        tnf = tnf[mask].astype(_np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        _vambtools.zscore(rpkm, axis=0, inplace=True)

    # Normalize arrays and create the Tensors (the tensors share the underlying memory)
    # of the Numpy arrays
    _vambtools.zscore(tnf, axis=0, inplace=True)
    depthstensor = _torch.from_numpy(rpkm)
    tnftensor = _torch.from_numpy(tnf)

    # Create dataloader
    n_workers = 4 if cuda else 0
    return depthstensor, tnftensor, batchsize, n_workers, cuda, mask

def make_dataloader(rpkm, tnf, batchsize=256, destroy=False, cuda=False):
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """
    depthstensor, tnftensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, batchsize=batchsize, destroy=destroy, cuda=cuda)
    dataset = _TensorDataset(depthstensor, tnftensor)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask

def make_dataloader_concat(rpkm, tnf, labels, batchsize=256, destroy=False, cuda=False):
    depthstensor, tnftensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels_int = _np.unique(labels, return_inverse=True)[1]
    one_hot_labels = F.one_hot(_torch.as_tensor(labels_int)).float()
    dataset = _TensorDataset(depthstensor, tnftensor, one_hot_labels)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask

def make_dataloader_labels(rpkm, tnf, labels, batchsize=256, destroy=False, cuda=False):
    depthstensor, tnftensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels_int = _np.unique(labels, return_inverse=True)[1]
    one_hot_labels = F.one_hot(_torch.as_tensor(labels_int)).float()
    if one_hot_labels.shape[1] < 105:
        one_hot_labels = F.pad(one_hot_labels, (1, 105 - one_hot_labels.shape[1]), "constant", 0) # BIG HACK, for when nlabels < 103
    dataset = _TensorDataset(one_hot_labels)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask


def make_dataloader_embeddings(rpkm, tnf, labels, batchsize=256, destroy=False, cuda=False):
    depthstensor, tnftensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels = _torch.as_tensor(labels).float()
    if labels.shape[1] < 105:
        labels = F.pad(labels, (1, 105 - labels.shape[1]), "constant", 0) # BIG HACK, for when nlabels < 103
    dataset = _TensorDataset(labels)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask


def make_dataloader_concat_embeddings(rpkm, tnf, labels, batchsize=256, destroy=False, cuda=False):
    depthstensor, tnftensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels = _torch.as_tensor(labels).float()
    dataset = _TensorDataset(depthstensor, tnftensor, labels)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask


def get_samplers(N_data, no_labels_share, index_start=0):
    indices = list(range(index_start, index_start+N_data))
    _np.random.shuffle(indices)
    split = int(no_labels_share * N_data)
    train_idx, valid_idx_x = indices[split:], indices
    valid_idx_y = [i for i in valid_idx_x]
    _np.random.shuffle(valid_idx_y)
    print(len(train_idx), len(valid_idx_x))
    unsupervised_sampler_x = valid_idx_x
    unsupervised_sampler_y = valid_idx_y
    supervised_sampler = train_idx
    return unsupervised_sampler_x, unsupervised_sampler_y, supervised_sampler

#TODO: refactor
def make_dataloader_semisupervised_random(rpkm, tnf, labels, supervision_level, batchsize=256, destroy=False, cuda=False):
    N_data = len(rpkm)
    index_x, index_y, index_sup = get_samplers(N_data, 1 - supervision_level)
    depthstensor, tnftensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels_int = _np.unique(labels, return_inverse=True)[1]
    one_hot_labels = F.one_hot(_torch.as_tensor(labels_int)).float()

    dataset_vamb = _TensorDataset(depthstensor[index_x], tnftensor[index_x])
    dataloader_vamb = _DataLoader(dataset=dataset_vamb, batch_size=batchsize, drop_last=True,
                            shuffle=True, num_workers=n_workers, pin_memory=cuda)

    dataset_labels = _TensorDataset(one_hot_labels[index_y])
    dataloader_labels = _DataLoader(dataset=dataset_labels, batch_size=batchsize, drop_last=True,
                            shuffle=True, num_workers=n_workers, pin_memory=cuda)

    dataset_joint = _TensorDataset(depthstensor[index_sup], tnftensor[index_sup], one_hot_labels[index_sup])
    dataloader_joint = _DataLoader(dataset=dataset_joint, batch_size=batchsize, drop_last=True,
                            shuffle=True, num_workers=n_workers, pin_memory=cuda)
    indices_all = (index_x, index_y, index_sup)
    return dataloader_joint, dataloader_vamb, dataloader_labels, mask, indices_all


def make_dataloader_semisupervised(dataloader_joint, dataloader_vamb, dataloader_labels, shapes, batchsize=256, destroy=False, cuda=False):
    train_xy_iterator = dataloader_joint.__iter__()
    train_x_iterator = dataloader_vamb.__iter__()
    train_y_iterator = dataloader_labels.__iter__()

    ds, ts, ls = shapes

    d_all = _torch.zeros((1, ds))
    d_u_all = _torch.zeros((1, ds))
    t_all = _torch.zeros((1, ts))
    t_u_all = _torch.zeros((1, ts))
    l_all = _torch.zeros((1, ls))
    l_u_all = _torch.zeros((1, ls))

    for i in range(len(dataloader_vamb)):
        try:
            d, t, l = next(train_xy_iterator)
        except StopIteration:
            train_xy_iterator = dataloader_joint.__iter__()
            d, t, l = next(train_xy_iterator)

        try:
            d_u, t_u = next(train_x_iterator)
        except StopIteration:
            train_x_iterator = dataloader_vamb.__iter__()
            d_u, t_u = next(train_x_iterator)

        try:
            l_u = next(train_y_iterator)
        except StopIteration:
            train_y_iterator = dataloader_labels.__iter__()
            l_u = next(train_y_iterator)

        d_all = _torch.cat([d_all, d])
        t_all = _torch.cat([t_all, t])
        l_all = _torch.cat([l_all, l])
        d_u_all = _torch.cat([d_u_all, d_u])
        t_u_all = _torch.cat([t_u_all, t_u])
        l_u_all = _torch.cat([l_u_all, l_u[0]])

    dataset_all = _TensorDataset(d_all[1:, :], t_all[1:, :], l_all[1:, :], d_u_all[1:, :], t_u_all[1:, :], l_u_all[1:, :])
    print(d_all.shape)
    dataloader_all = _DataLoader(dataset=dataset_all, batch_size=batchsize, drop_last=True,
                        shuffle=True, num_workers=dataloader_joint.num_workers, pin_memory=cuda)
    return dataloader_all


class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
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

    def __init__(self, nsamples, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))

        if nsamples < 1:
            raise ValueError('nsamples must be > 0, not {}'.format(nsamples))

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))

        if beta <= 0:
            raise ValueError('beta must be > 0, not {}'.format(beta))

        if not (0 < alpha < 1):
            raise ValueError('alpha must be 0 < alpha < 1, not {}'.format(alpha))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1, not {}'.format(dropout))

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip([self.nsamples + self.ntnf] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = _nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = _nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(_nn.Linear(nin, nout))
            self.decodernorms.append(_nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf)

        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _encode(self, tensor):
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)

        # Note: This softplus constrains logsigma to positive. As reconstruction loss pushes
        # logsigma as low as possible, and KLD pushes it towards 0, the optimizer will
        # always push this to 0, meaning that the logsigma layer will be pushed towards
        # negative infinity. This creates a nasty numerical instability in VAMB. Luckily,
        # the gradient also disappears as it decreases towards negative infinity, avoiding
        # NaN poisoning in most cases. We tried to remove the softplus layer, but this
        # necessitates a new round of hyperparameter optimization, and there is no way in
        # hell I am going to do that at the moment of writing.
        # Also remove needless factor 2 in definition of latent in reparameterize function.
        logsigma = self.softplus(self.logsigma(tensor))

        return mu, logsigma

    # sample with gaussian noise
    def reparameterize(self, mu, logsigma):
        epsilon = _torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * _torch.exp(logsigma/2)

        return latent

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)

        return depths_out, tnf_out

    def forward(self, depths, tnf):
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + kld * kld_weight

        return loss, ce, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in, tnf_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)

            loss, ce, sse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_sseloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        return data_loader

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()
        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        depths_array, tnf_array = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()

                # Evaluate
                out_depths, out_tnf, mu, logsigma = self(depths, tnf)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'state': self.state_dict(),
                }

        _torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
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

        nsamples = dictionary['nsamples']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None):
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
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError('Last batch size exceeds dataset length')
            batchsteps_set = set(batchsteps)

        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.nhiddens)), file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None


class VAEConcat(VAE):
    """Variational autoencoder that uses labels as concatented input, subclass of VAE.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
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

    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAEConcat, self).__init__(nsamples + nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nsamples = nsamples
        self.nlabels = nlabels

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        labels_out = reconstruction.narrow(1, self.nsamples + self.ntnf, self.nlabels)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)
            labels_out = _softmax(labels_out, dim=1)

        return depths_out, tnf_out, labels_out

    def forward(self, depths, tnf, labels):
        tensor = _torch.cat((depths, tnf, labels), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out, labels_out = self._decode(latent)

        return depths_out, tnf_out, labels_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce, sse, ce_labels, kld, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0
        epoch_celabelsloss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in, tnf_in, labels_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True
            labels_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, labels_out, mu, logsigma = self(depths_in, tnf_in, labels_in)

            loss, ce, sse, ce_labels, kld, correct_labels = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, labels_in, labels_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()
            epoch_celabelsloss += ce_labels.data.item()
            epoch_correct_labels+= correct_labels.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tCE_labels: {:.7f}\tKLD: {:.4f}\taccuracy: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_sseloss / len(data_loader),
                  epoch_celabelsloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  epoch_correct_labels / (len(data_loader)*256),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        return data_loader

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()
        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        depths_array, tnf_array, labels_array = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf, labels in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()
                    labels = labels.cuda()

                # Evaluate
                out_depths, out_tnf, out_labels, mu, logsigma = self(depths, tnf, labels)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent


class VAELabels(VAE):
    """Variational autoencoder that encodes only the labels, subclass of VAE.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
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

    def __init__(self, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAELabels, self).__init__(nlabels - 103, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nlabels = nlabels

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)
        labels_out = reconstruction.narrow(1, 0, self.nlabels)
        return labels_out

    def forward(self, labels):
        mu, logsigma = self._encode(labels)
        latent = self.reparameterize(mu, logsigma)
        labels_out = self._decode(latent)
        return labels_out, mu, logsigma

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce_labels, kld, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_celabelsloss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for labels_in in data_loader:
            labels_in = labels_in[0]
            labels_in.requires_grad = True

            if self.usecuda:
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            labels_out, mu, logsigma = self(labels_in)

            loss, ce_labels, kld, correct_labels = self.calc_loss(labels_in, labels_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_celabelsloss += ce_labels.data.item()
            epoch_correct_labels+= correct_labels.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE_labels: {:.7f}\tKLD: {:.4f}\taccuracy: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celabelsloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  epoch_correct_labels / (len(data_loader)*256),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        return data_loader

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()
        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=0,
                                      pin_memory=data_loader.pin_memory)

        labels_array = data_loader.dataset.tensors
        length = len(labels_array[0])

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for labels in new_data_loader:
                labels = labels[0]
                # Move input to GPU if requested
                if self.usecuda:
                    labels = labels.cuda()

                # Evaluate
                out_labels, mu, logsigma = self(labels)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent


class VAEEmbeddings(VAELabels):
    """Variational autoencoder that encodes only the labels word2vec embeddings, subclass of VAE.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
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

    def __init__(self, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAEEmbeddings, self).__init__(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nlabels = nlabels

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        mse_labels = _nn.L1Loss()(labels_in, labels_out)
        ce_labels_weight = 20. #TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = mse_labels*ce_labels_weight + kld * kld_weight
        return loss, mse_labels, kld, _torch.as_tensor(666)


def kld_gauss(p_mu, p_std, q_mu, q_std):
    p = _torch.distributions.normal.Normal(p_mu, p_std)
    q = _torch.distributions.normal.Normal(q_mu, q_std)
    loss = _torch.distributions.kl_divergence(p, q)
    return loss.mean()


class VAEVAE(object):
    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        self.VAEVamb = VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels = VAELabels(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAEJoint = VAEConcat(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

    def calc_loss_joint(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, 
                            mu_sup, logsigma_sup, 
                            mu_vamb_unsup, logsigma_vamb_unsup,
                            mu_labels_unsup, logsigma_labels_unsup,
                            ):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.VAEVamb.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.VAEVamb.alpha) / _log(self.VAEVamb.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.VAEVamb.alpha

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        sse_weight = self.VAEVamb.alpha / self.VAEVamb.ntnf
        kld_weight = 1 / (self.VAEVamb.nlatent * self.VAEVamb.beta)

        kld_vamb = kld_gauss(mu_sup, logsigma_sup, mu_vamb_unsup, logsigma_vamb_unsup)
        kld_labels = kld_gauss(mu_sup, logsigma_sup, mu_labels_unsup, logsigma_labels_unsup)
        kld = kld_vamb + kld_labels

        loss = ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce, sse, ce_labels, kld_vamb, kld_labels, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        metrics = [
            'loss_vamb', 'ce_vamb', 'sse_vamb', 'kld_vamb', 
            'loss_labels', 'ce_labels_labels', 'kld_labels', 'correct_labels_labels',
            'loss_joint', 'ce_joint', 'sse_joint', 'ce_labels_joint', 'kld_vamb_joint', 'kld_labels_joint', 'correct_labels_joint',
            'loss',
        ]
        metrics_dict = {k: 0 for k in metrics}
        tensors_dict = {k: None for k in metrics}

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in_sup, tnf_in_sup, labels_in_sup, depths_in_unsup, tnf_in_unsup, labels_in_unsup in data_loader:
            depths_in_sup.requires_grad = True
            tnf_in_sup.requires_grad = True
            labels_in_sup.requires_grad = True
            depths_in_unsup.requires_grad = True
            tnf_in_unsup.requires_grad = True
            labels_in_unsup.requires_grad = True

            if self.VAEVamb.usecuda:
                depths_in_sup = depths_in_sup.cuda()
                tnf_in_sup = tnf_in_sup.cuda()
                labels_in_sup = labels_in_sup.cuda()
                depths_in_unsup = depths_in_unsup.cuda()
                tnf_in_unsup = tnf_in_unsup.cuda()
                labels_in_unsup = labels_in_unsup.cuda()

            optimizer.zero_grad()

            _, _, _, mu_sup, logsigma_sup = self.VAEJoint(depths_in_sup, tnf_in_sup, labels_in_sup) # use the two-modality latent space

            depths_out_sup, tnf_out_sup = self.VAEVamb._decode(self.VAEVamb.reparameterize(mu_sup, logsigma_sup)) # use the one-modality decoders
            labels_out_sup = self.VAELabels._decode(self.VAELabels.reparameterize(mu_sup, logsigma_sup)) # use the one-modality decoders

            depths_out_unsup, tnf_out_unsup,  mu_vamb_unsup, logsigma_vamb_unsup = self.VAEVamb(depths_in_unsup, tnf_in_unsup)
            labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup = self.VAELabels(labels_in_unsup)

            tensors_dict['loss_vamb'], tensors_dict['ce_vamb'], tensors_dict['sse_vamb'], tensors_dict['kld_vamb'] = \
                self.VAEVamb.calc_loss(depths_in_unsup, depths_out_unsup, tnf_in_unsup, tnf_out_unsup, mu_vamb_unsup, logsigma_vamb_unsup)
            tensors_dict['loss_labels'], tensors_dict['ce_labels_labels'], tensors_dict['kld_labels'], tensors_dict['correct_labels_labels'] = \
                self.VAELabels.calc_loss(labels_in_unsup, labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup)
            
            tensors_dict['loss_joint'], tensors_dict['ce_joint'], tensors_dict['sse_joint'], tensors_dict['ce_labels_joint'], \
                tensors_dict['kld_vamb_joint'], tensors_dict['kld_labels_joint'], tensors_dict['correct_labels_joint'] = self.calc_loss_joint(
                    depths_in_sup, depths_out_sup, tnf_in_sup, tnf_out_sup, labels_in_sup, labels_out_sup, 
                    mu_sup, logsigma_sup, 
                    mu_vamb_unsup, logsigma_vamb_unsup,
                    mu_labels_unsup, logsigma_labels_unsup,
                )

            tensors_dict['loss'] = tensors_dict['loss_joint'] + tensors_dict['loss_vamb'] + tensors_dict['loss_labels']

            tensors_dict['loss'].backward()
            optimizer.step()

            for k, v in tensors_dict.items():
                metrics_dict[k] += v.data.item()

        metrics_dict['correct_labels_joint'] /= 256
        metrics_dict['correct_labels_labels'] /= 256
        if logfile is not None:
            print(', '.join([k + f' {v/len(data_loader):.6f}' for k, v in metrics_dict.items()]), file=logfile)
            logfile.flush()

        return data_loader

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None):
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
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError('Last batch size exceeds dataset length')
            batchsteps_set = set(batchsteps)

        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _Adam(list(self.VAEVamb.parameters()) + list(self.VAELabels.parameters()) + list(self.VAEJoint.parameters()), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.VAEVamb.usecuda, file=logfile)
            print('\tAlpha:', self.VAEVamb.alpha, file=logfile)
            print('\tBeta:', self.VAEVamb.beta, file=logfile)
            print('\tDropout:', self.VAEVamb.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.VAEVamb.nhiddens)), file=logfile)
            print('\tN latent:', self.VAEVamb.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'nlabels': self.nlabels,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'state_VAEVamb': self.VAEVamb.state_dict(),
                 'state_VAELabels': self.VAELabels.state_dict(),
                 'state_VAEJoint': self.VAEJoint.state_dict(),
                }

        _torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
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

        nsamples = dictionary['nsamples']
        nlabels = dictionary['nlabels']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']

        vae = cls(nsamples, nlabels, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.VAEVamb.load_state_dict(dictionary['state_VAEVamb'])
        vae.VAELabels.load_state_dict(dictionary['state_VAELabels'])
        vae.VAEJoint.load_state_dict(dictionary['state_VAEJoint'])

        if cuda:
            vae.VAEVamb.cuda()
            vae.VAELabels.lcuda()
            vae.VAEJoint.cuda()

        if evaluate:
            vae.VAEVamb.eval()
            vae.VAELabels.eval()
            vae.VAEJoint.eval()

        return vae


class VAEConcatEmbeddings(VAEConcat):
    """Variational autoencoder that uses labels as concatented input, subclass of VAE.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
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

    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAEConcatEmbeddings, self).__init__(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        ce_labels = _nn.L1Loss()(labels_in, labels_out)
        ce_labels_weight = 20. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce, sse, ce_labels, kld, _torch.sum(labels_out_indices == labels_in_indices)



class VAEVAEEmbeddings(VAEVAE):
    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAEVAEEmbeddings, self).__init__(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAEVamb = VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels = VAEEmbeddings(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAEJoint = VAEConcatEmbeddings(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

    def calc_loss_joint(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, 
                            mu_sup, logsigma_sup, 
                            mu_vamb_unsup, logsigma_vamb_unsup,
                            mu_labels_unsup, logsigma_labels_unsup,
                            ):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.VAEVamb.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.VAEVamb.alpha) / _log(self.VAEVamb.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.VAEVamb.alpha

        ce_labels = _nn.L1Loss()(labels_in, labels_out)
        ce_labels_weight = 20. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        sse_weight = self.VAEVamb.alpha / self.VAEVamb.ntnf
        kld_weight = 1 / (self.VAEVamb.nlatent * self.VAEVamb.beta)

        kld_vamb = kld_gauss(mu_sup, logsigma_sup, mu_vamb_unsup, logsigma_vamb_unsup)
        kld_labels = kld_gauss(mu_sup, logsigma_sup, mu_labels_unsup, logsigma_labels_unsup)
        kld = kld_vamb + kld_labels

        loss = ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce, sse, ce_labels, kld_vamb, kld_labels, _torch.sum(labels_out_indices == labels_in_indices)


class SVAE(VAEVAE):
    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        self.VAEVamb = VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAEVamb_star = VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels = VAELabels(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels_star = VAELabels(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

    def calc_loss_joint(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, 
                            mu_sup, logsigma_sup, 
                            mu_vamb_unsup, logsigma_vamb_unsup,
                            mu_labels_unsup, logsigma_labels_unsup,
                            ):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.VAEVamb.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.VAEVamb.alpha) / _log(self.VAEVamb.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.VAEVamb.alpha

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        sse_weight = self.VAEVamb.alpha / self.VAEVamb.ntnf
        kld_weight = 1 / (self.VAEVamb.nlatent * self.VAEVamb.beta)

        kld_vamb = kld_gauss(mu_sup, logsigma_sup, mu_vamb_unsup, logsigma_vamb_unsup)
        kld_labels = kld_gauss(mu_sup, logsigma_sup, mu_labels_unsup, logsigma_labels_unsup)
        kld = kld_vamb + kld_labels

        loss = ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce, sse, ce_labels, kld_vamb, kld_labels, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        metrics = [
            'loss_vamb', 'ce_vamb', 'sse_vamb', 'kld_vamb', 
            'loss_labels', 'ce_labels_labels', 'kld_labels', 'correct_labels_labels',
            'loss_joint', 'ce_joint', 'sse_joint', 'ce_labels_joint', 'kld_vamb_joint', 'kld_labels_joint', 'correct_labels_joint',
            'loss',
        ]
        metrics_dict = {k: 0 for k in metrics}
        tensors_dict = {k: None for k in metrics}

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in_sup, tnf_in_sup, labels_in_sup, depths_in_unsup, tnf_in_unsup, labels_in_unsup in data_loader:
            depths_in_sup.requires_grad = True
            tnf_in_sup.requires_grad = True
            labels_in_sup.requires_grad = True
            depths_in_unsup.requires_grad = True
            tnf_in_unsup.requires_grad = True
            labels_in_unsup.requires_grad = True

            if self.VAEVamb.usecuda:
                depths_in_sup = depths_in_sup.cuda()
                tnf_in_sup = tnf_in_sup.cuda()
                labels_in_sup = labels_in_sup.cuda()
                depths_in_unsup = depths_in_unsup.cuda()
                tnf_in_unsup = tnf_in_unsup.cuda()
                labels_in_unsup = labels_in_unsup.cuda()

            optimizer.zero_grad()

            _, _, _, mu_sup, logsigma_sup = self.VAEJoint(depths_in_sup, tnf_in_sup, labels_in_sup) # use the two-modality latent space

            depths_out_sup, tnf_out_sup = self.VAEVamb._decode(self.VAEVamb.reparameterize(mu_sup, logsigma_sup)) # use the one-modality decoders
            labels_out_sup = self.VAELabels._decode(self.VAELabels.reparameterize(mu_sup, logsigma_sup)) # use the one-modality decoders

            depths_out_unsup, tnf_out_unsup,  mu_vamb_unsup, logsigma_vamb_unsup = self.VAEVamb(depths_in_unsup, tnf_in_unsup)
            labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup = self.VAELabels(labels_in_unsup)

            tensors_dict['loss_vamb'], tensors_dict['ce_vamb'], tensors_dict['sse_vamb'], tensors_dict['kld_vamb'] = \
                self.VAEVamb.calc_loss(depths_in_unsup, depths_out_unsup, tnf_in_unsup, tnf_out_unsup, mu_vamb_unsup, logsigma_vamb_unsup)
            tensors_dict['loss_labels'], tensors_dict['ce_labels_labels'], tensors_dict['kld_labels'], tensors_dict['correct_labels_labels'] = \
                self.VAELabels.calc_loss(labels_in_unsup, labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup)
            
            tensors_dict['loss_joint'], tensors_dict['ce_joint'], tensors_dict['sse_joint'], tensors_dict['ce_labels_joint'], \
                tensors_dict['kld_vamb_joint'], tensors_dict['kld_labels_joint'], tensors_dict['correct_labels_joint'] = self.calc_loss_joint(
                    depths_in_sup, depths_out_sup, tnf_in_sup, tnf_out_sup, labels_in_sup, labels_out_sup, 
                    mu_sup, logsigma_sup, 
                    mu_vamb_unsup, logsigma_vamb_unsup,
                    mu_labels_unsup, logsigma_labels_unsup,
                )

            tensors_dict['loss'] = tensors_dict['loss_joint'] + tensors_dict['loss_vamb'] + tensors_dict['loss_labels']

            tensors_dict['loss'].backward()
            optimizer.step()

            for k, v in tensors_dict.items():
                metrics_dict[k] += v.data.item()

        metrics_dict['correct_labels_joint'] /= 256
        metrics_dict['correct_labels_labels'] /= 256
        if logfile is not None:
            print(', '.join([k + f' {v/len(data_loader):.6f}' for k, v in metrics_dict.items()]), file=logfile)
            logfile.flush()

        return data_loader

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None):
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
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError('Last batch size exceeds dataset length')
            batchsteps_set = set(batchsteps)

        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _Adam(list(self.VAEVamb.parameters()) + list(self.VAELabels.parameters()) + list(self.VAEJoint.parameters()), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.VAEVamb.usecuda, file=logfile)
            print('\tAlpha:', self.VAEVamb.alpha, file=logfile)
            print('\tBeta:', self.VAEVamb.beta, file=logfile)
            print('\tDropout:', self.VAEVamb.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.VAEVamb.nhiddens)), file=logfile)
            print('\tN latent:', self.VAEVamb.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'nlabels': self.nlabels,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'state_VAEVamb': self.VAEVamb.state_dict(),
                 'state_VAELabels': self.VAELabels.state_dict(),
                 'state_VAEJoint': self.VAEJoint.state_dict(),
                }

        _torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
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

        nsamples = dictionary['nsamples']
        nlabels = dictionary['nlabels']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']

        vae = cls(nsamples, nlabels, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.VAEVamb.load_state_dict(dictionary['state_VAEVamb'])
        vae.VAELabels.load_state_dict(dictionary['state_VAELabels'])
        vae.VAEJoint.load_state_dict(dictionary['state_VAEJoint'])

        if cuda:
            vae.VAEVamb.cuda()
            vae.VAELabels.lcuda()
            vae.VAEJoint.cuda()

        if evaluate:
            vae.VAEVamb.eval()
            vae.VAELabels.eval()
            vae.VAEJoint.eval()

        return vae
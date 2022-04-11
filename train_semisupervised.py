import vamb
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data/airways')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--supervision", type=float, default=1.)

args = vars(parser.parse_args())
print(args)

SUP = args['supervision']
CUDA = bool(args['cuda'])
DIRPATH = args['path']
PATH_CONTIGS = f'{DIRPATH}/contigs.fna'
ABUNDANCE_PATH = f'{DIRPATH}/abundance.npz'
MODEL_PATH = 'model_semisupervised.pt'
N_EPOCHS = args['nepoch']
REFERENCE_PATH = f'{DIRPATH}/reference.tsv'

with vamb.vambtools.Reader(PATH_CONTIGS, 'rb') as filehandle:
    tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(filehandle)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)

df_ref = pd.read_csv(REFERENCE_PATH, delimiter='\t', header=None)
classes_dict = {r[0]: r[1] for i, r in df_ref.iterrows()}
classes_order = np.array([classes_dict[c] for c in contignames])

vae = vamb.encode.VAEVAE(nsamples=rpkms.shape[1], nlabels=len(set(classes_order)), cuda=CUDA)
dataloader_joint, dataloader_vamb, dataloader_labels, mask, indices_all = \
    vamb.encode.make_dataloader_semisupervised_random(rpkms, tnfs, classes_order, SUP)

with open(f'indices_{int(SUP*100)}.pickle', 'wb') as handle:
    pickle.dump(indices_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

shapes = (rpkms.shape[1], 103, len(set(classes_order)))
dataloader = vamb.encode.make_dataloader_semisupervised(dataloader_joint, dataloader_vamb, dataloader_labels, shapes)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        batchsteps=None,
        logfile=sys.stdout,
    )
    print('training')

latent = vae.VAEVamb.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_semisupervised_{int(SUP*100)}_vamb.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent)

latent = vae.VAELabels.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_semisupervised_{int(SUP*100)}_labels.npy'
print('Saving latent space: Labels')
np.save(LATENT_PATH, latent)

latent = vae.VAEJoint.encode(dataloader_joint)
LATENT_PATH = f'latent_trained_semisupervised_{int(SUP*100)}_both.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent)

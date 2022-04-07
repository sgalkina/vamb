import vamb
import sys
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data/airways')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)

args = vars(parser.parse_args())
print(args)

CUDA = bool(args['cuda'])
DIRPATH = args['path']
PATH_CONTIGS = f'{DIRPATH}/contigs.fna'
ABUNDANCE_PATH = f'{DIRPATH}/abundance.npz'
MODEL_PATH = 'model_vamb.pt'
N_EPOCHS = args['nepoch']
LATENT_PATH = 'latent_trained_semisupervised.npy'
REFERENCE_PATH = f'{DIRPATH}/reference.tsv'

with vamb.vambtools.Reader(PATH_CONTIGS, 'rb') as filehandle:
    tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(filehandle)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)

df_ref = pd.read_csv(REFERENCE_PATH, delimiter='\t', header=None)
classes_dict = {r[0]: r[1] for i, r in df_ref.iterrows()}
classes_order = np.array([classes_dict[c] for c in contignames])

vae = vamb.encode.VAEConcat(nsamples=rpkms.shape[1], nlabels=len(set(classes_order)), cuda=CUDA)
dataloader, mask = vamb.encode.make_dataloader_concat(rpkms, tnfs, classes_order)
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

latent = vae.encode(dataloader)
print('Saving latent space')
np.save(LATENT_PATH, latent)

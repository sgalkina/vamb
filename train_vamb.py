import vamb
import sys
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--dataset", type=str, default='airways')

args = vars(parser.parse_args())
print(args)

CUDA = bool(args['cuda'])
DATASET = args['dataset']
DIRPATH = f"{args['path']}/{DATASET}"
DEPTH_PATH = f'/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_{DATASET}'
PATH_CONTIGS = f'{DEPTH_PATH}/contigs.fna'
ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
MODEL_PATH = f'model_vamb_{DATASET}.pt'
N_EPOCHS = args['nepoch']
LATENT_PATH = f'latent_trained_vamb_{DATASET}.npy'
REFERENCE_PATH = f'{DIRPATH}/reference.tsv'

with vamb.vambtools.Reader(PATH_CONTIGS, 'rb') as filehandle:
    tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(filehandle)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)

df_ref = pd.read_csv(REFERENCE_PATH, delimiter='\t', header=None)
classes_dict = {r[0]: r[1] for i, r in df_ref.iterrows()}
classes_order = np.array([classes_dict[c] for c in contignames])

vae = vamb.encode.VAE(nsamples=rpkms.shape[1], cuda=CUDA)
dataloader, mask = vamb.encode.make_dataloader(rpkms, tnfs)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
    )
    print('training')

latent = vae.encode(dataloader)
print('Saving latent space')
np.save(LATENT_PATH, latent)

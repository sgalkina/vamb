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
MODEL_PATH = f'model_semisupervised_mmseq_genus.pt'
N_EPOCHS = args['nepoch']
REFERENCE_PATH = f'{DIRPATH}/reference.tsv'

with vamb.vambtools.Reader(PATH_CONTIGS, 'rb') as filehandle:
    tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(filehandle)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)

df_mmseq = pd.read_csv(f'airways_taxonomy_clean_new.tsv', delimiter='\t', header=None)
df_mmseq_genus = df_mmseq[(df_mmseq[2] == 'genus') | (df_mmseq[2] == 'species')]
df_mmseq_genus['genus'] = df_mmseq_genus[8].str.split(';').str[5]
contigs = np.array(contignames)
indices_mmseq = [np.argwhere(contigs == c)[0][0] for c in df_mmseq_genus[0]]
classes_order = list(df_mmseq_genus['genus'])

vae = vamb.encode.VAEVAE(nsamples=rpkms.shape[1], nlabels=len(set(classes_order)), cuda=CUDA)

with open(f'indices_mmseq_genus.pickle', 'wb') as handle:
    pickle.dump(indices_mmseq, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_vamb, mask = vamb.encode.make_dataloader(rpkms, tnfs)
dataloader_joint, mask = vamb.encode.make_dataloader_concat(rpkms[indices_mmseq], tnfs[indices_mmseq], classes_order)
dataloader_labels, mask = vamb.encode.make_dataloader_labels(rpkms[indices_mmseq], tnfs[indices_mmseq], classes_order)

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
LATENT_PATH = f'latent_trained_semisupervised_mmseq_genus_vamb.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent)

latent = vae.VAELabels.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_semisupervised_mmseq_genus_labels.npy'
print('Saving latent space: Labels')
np.save(LATENT_PATH, latent)

latent = vae.VAEJoint.encode(dataloader_joint)
LATENT_PATH = f'latent_trained_semisupervised_mmseq_genus_both.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent)

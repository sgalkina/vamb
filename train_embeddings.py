import vamb
import sys
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
MODEL_PATH = 'model_embedding.pt'
N_EPOCHS = args['nepoch']
LATENT_PATH = 'latent_trained_embeddings_species_mmseq100.npy'
REFERENCE_PATH = f'{DIRPATH}/reference.tsv'
TAXONOMY_PATH = f'{DIRPATH}/taxonomy.tsv'

with vamb.vambtools.Reader(PATH_CONTIGS, 'rb') as filehandle:
    tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(filehandle)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)

df_ref = pd.read_csv(REFERENCE_PATH, delimiter='\t', header=None)
df_tax = pd.read_csv(TAXONOMY_PATH, delimiter='\t', header=None)

genus_dict = {r[0]: r[2] for _, r in df_tax.iterrows()}
species_dict = {r[0]: r[1] for _, r in df_tax.iterrows()}

df_ref['genus'] = df_ref[1].map(genus_dict)
df_ref['species'] = df_ref[1].map(species_dict)
df_ref['g'] = 'g__'
df_ref['s'] = 's__'

df_ref['genus'] = df_ref['g'].str.cat(df_ref['genus'])
df_ref['species'] = df_ref['s'].str.cat(df_ref['species'])

emb_df = pd.read_csv('embeddings_bacteria.tsv', index_col=0)
words = set(emb_df.index)

values_dict = {d: random.uniform(-1, 1) for i, d in enumerate(words)}
# classes_dict = {r[0]: r['species'] for i, r in df_ref.iterrows()}

df_mmseq = pd.read_csv('airways_taxonomy_clean_new.tsv', delimiter='\t', header=None)
classes_dict = {r[0]: 's__' + r[3] for i, r in df_mmseq.iterrows()}

ress = []
for c in contignames:
    g = classes_dict.get(c, '')
    if g in words:
        ress.append(emb_df.loc[g].values)
    else:
        ress.append([0]*150)

embeddings = np.stack(ress)
print(embeddings.shape)

vae = vamb.encode.VAEEmbeddings(nlabels=max(embeddings.shape[1], 106), cuda=CUDA)
dataloader, mask = vamb.encode.make_dataloader_embeddings(rpkms, tnfs, embeddings)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        batchsteps=None,
        logfile=sys.stdout,
        lrate=1e-3,
    )
    print('training')

latent = vae.encode(dataloader)
print('Saving latent space')
np.save(LATENT_PATH, latent)

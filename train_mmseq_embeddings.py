import vamb
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--dataset", type=str, default='airways')
parser.add_argument("--supervision", type=float, default=1.)

args = vars(parser.parse_args())
print(args)

SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = args['dataset']
DIRPATH = f"{args['path']}/{DATASET}"
DEPTH_PATH = f'/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_{DATASET}'
PATH_CONTIGS = f'{DEPTH_PATH}/contigs_2kbp.fna.gz'
ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
MODEL_PATH = f'model_semisupervised_mmseq_{DATASET}_embeddings.pt'
N_EPOCHS = args['nepoch']
REFERENCE_PATH = f'{DIRPATH}/reference.tsv'
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy.tsv'

with vamb.vambtools.Reader(PATH_CONTIGS, 'rb') as filehandle:
    tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(filehandle)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)

emb_df = pd.read_csv('embeddings_bacteria.tsv', index_col=0)
words = set(emb_df.index)

df_mmseq = pd.read_csv(MMSEQ_PATH, delimiter='\t', header=None)
df_mmseq_genus = df_mmseq[(df_mmseq[2] == 'genus') | (df_mmseq[2] == 'species')]
df_mmseq_genus['genus'] = df_mmseq_genus[8].str.split(';').str[5].str.replace('g_', 'g__')
df_mmseq_genus = df_mmseq_genus[df_mmseq_genus['genus'].isin(words)]
contigs = np.array(contignames)
indices_mmseq = [np.argwhere(contigs == c)[0][0] for c in df_mmseq_genus[0]]
classes_order = list(df_mmseq_genus['genus'])

ress = []
for c in df_mmseq_genus['genus']:
    ress.append(emb_df.loc[c].values)
embeddings = np.stack(ress)

vae = vamb.encode.VAEVAEEmbeddings(nsamples=rpkms.shape[1], nlabels=max(embeddings.shape[1], 106), cuda=CUDA)

with open(f'indices_mmseq_{DATASET}_embeddings.pickle', 'wb') as handle:
    pickle.dump(indices_mmseq, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_vamb, mask = vamb.encode.make_dataloader(rpkms, tnfs)
dataloader_joint, mask = vamb.encode.make_dataloader_concat_embeddings(rpkms[indices_mmseq], tnfs[indices_mmseq], embeddings)
dataloader_labels, mask = vamb.encode.make_dataloader_embeddings(rpkms[indices_mmseq], tnfs[indices_mmseq], embeddings)

shapes = (rpkms.shape[1], 103, embeddings.shape[1])
dataloader = vamb.encode.make_dataloader_semisupervised(dataloader_joint, dataloader_vamb, dataloader_labels, shapes)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
    )
    print('training')

latent = vae.VAEVamb.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_semisupervised_mmseq_vamb_{DATASET}_embeddings.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent)

latent = vae.VAELabels.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_semisupervised_mmseq_labels_{DATASET}_embeddings.npy'
print('Saving latent space: Labels')
np.save(LATENT_PATH, latent)

latent = vae.VAEJoint.encode(dataloader_joint)
LATENT_PATH = f'latent_trained_semisupervised_mmseq_both_{DATASET}_embeddings.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent)

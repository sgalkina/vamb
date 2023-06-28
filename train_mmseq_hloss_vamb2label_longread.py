import vamb
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--dataset", type=str, default='airways')
parser.add_argument("--supervision", type=float, default=1.)

args = vars(parser.parse_args())
print(args)

exp_id = '_longread'

SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = 'longread'
DEPTH_PATH = f'/home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout'
COMPOSITION_PATH = f'{DEPTH_PATH}/composition.npz'
ABUNDANCE_PATH = f'{DEPTH_PATH}/abundance.npz'
MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}.pt'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy_2023.tsv'
GT_PATH = f'gt_tax_{DATASET}.csv'

rpkms = np.load(ABUNDANCE_PATH, mmap_mode='r')['matrix']
composition = np.load(COMPOSITION_PATH, mmap_mode='r', allow_pickle=True)
tnfs, lengths = composition['matrix'], composition['lengths']
contignames = composition['identifiers']

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)

print('N contigs', len(contignames))
print('rpkms shape', rpkms.shape)
print('tnfs shape', tnfs.shape)

print('N contigs after mask', len(contignames[mask_vamb]))
print('rpkms shape after mask', rpkms[mask_vamb].shape)
print('tnfs shape after mask', tnfs[mask_vamb].shape)

rpkms = rpkms[mask_vamb]
tnfs, lengths = tnfs[mask_vamb], lengths[mask_vamb]
contignames = contignames[mask_vamb]
all_contigs = set(contignames)

df_mmseq = pd.read_csv(MMSEQ_PATH, delimiter='\t', header=None)
df_mmseq = df_mmseq[~(df_mmseq[2] == 'no rank')]
df_mmseq = df_mmseq[df_mmseq[0].isin(all_contigs)]
print('Total mmseq hits', len(df_mmseq))
print('Genus', len(df_mmseq[(df_mmseq[2] == 'genus') | (df_mmseq[2] == 'species')]))
print('Species', len(df_mmseq[(df_mmseq[2] == 'species')]))

df_mmseq_genus = df_mmseq

ind_map = {c: i for i, c in enumerate(contignames)}
indices_mmseq = [ind_map[c] for c in df_mmseq_genus[0]]

graph_column = df_mmseq_genus[8]
nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

model = vamb.h_loss.VAMB2Label(
     rpkms.shape[1], 
     len(nodes), 
     table_indices, 
     table_true, 
     table_walkdown, 
     nodes, 
     table_parent,
     cuda=CUDA,
)

with open(f'indices_mmseq_genus_{DATASET}{exp_id}.pickle', 'wb') as handle:
    pickle.dump(indices_mmseq, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms[indices_mmseq], tnfs[indices_mmseq], lengths[indices_mmseq], targets, len(nodes), table_parent)

shapes = (rpkms.shape[1], 103, 1, len(nodes))
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    model.trainmodel(
        dataloader_joint,
        nepochs=100,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[],
    )
    print('training')

latent_vamb = model.predict(dataloader_vamb)
LATENT_PATH = f'latent_predict_vamb_{DATASET}{exp_id}.npy'
# print('Saving latent space: Vamb', latent_vamb.shape)
# np.save(LATENT_PATH, latent_vamb)

print('Saving the tree', len(nodes))
TREE_PATH = f'tree_predict_vamb_{DATASET}{exp_id}.npy'
np.save(TREE_PATH, np.array(nodes))
PARENT_PATH = f'parents_predict_vamb_{DATASET}{exp_id}.npy'
np.save(PARENT_PATH, np.array(table_parent))

print('Getting the predictions')
df_gt = pd.DataFrame({'contigs': contignames})
nodes_ar = np.array(nodes)
predictions = []
for i in range(len(df_gt)):
    predictions.append(';'.join(nodes_ar[latent_vamb[i] > 0.5][1:]))
df_gt[f'predictions{exp_id}'] = predictions

df_mmseq_sp = df_mmseq[(df_mmseq[2] == 'species')]
mmseq_map = {k: v for k, v in zip(df_mmseq_sp[0], df_mmseq_sp[8])}
counters = defaultdict(lambda: 0)
preds = []
for i, r in df_gt.iterrows():
    pred_line = r[f'predictions{exp_id}'].split(';')
    try:
        mmseq_line = mmseq_map.get(r['contigs'], '').split(';')
    except AttributeError:
        preds.append(';'.join(pred_line))
        continue
    if mmseq_line[0] != '':
        for i in range(len(mmseq_line)):
            if i < len(pred_line):
                pred_line[i] = mmseq_line[i]
    preds.append(';'.join(pred_line))
df_gt[f'predictions{exp_id}_replace'] = preds

df_gt.to_csv(GT_PATH, index=None)

print('Starting the VAE')

graph_column = df_gt[f'predictions{exp_id}_replace']
nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

vae = vamb.h_loss.VAEVAEHLoss(
     rpkms.shape[1], 
     len(nodes), 
     table_indices, 
     table_true, 
     table_walkdown, 
     nodes, 
     table_parent,
     cuda=CUDA,
     logfile=sys.stdout,
)

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)
dataloader_labels, mask = vamb.h_loss.make_dataloader_labels_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)

shapes = (rpkms.shape[1], 103, 1, len(nodes))
dataloader = vamb.h_loss.make_dataloader_semisupervised_hloss(dataloader_joint, dataloader_vamb, dataloader_labels, len(nodes), table_parent, shapes)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[25, 75, 150],
    )
    print('training')

latent_vamb = vae.VAEVamb.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_vamb_{DATASET}{exp_id}.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent_vamb)

latent_labels = vae.VAELabels.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_labels_{DATASET}{exp_id}.npy'
print('Saving latent space: Labels')
np.save(LATENT_PATH, latent_labels)

latent_both = vae.VAEJoint.encode(dataloader_joint)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_both_{DATASET}{exp_id}.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent_both)

names = contignames

with open(f'clusters_mmseq_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent_both)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_vamb_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_vamb)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_labels_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_labels)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)
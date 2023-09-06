#!/usr/bin/bash
dataset=$1
run_id=$2

    # --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/${dataset}_taxonomy_2023.tsv \
    # --taxonomy_predictions /home/projects/cpr_10006/people/svekut/cami2_urog_out_32_667/results_taxonomy_predictor.csv
    # --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/${dataset}_taxonomy.tsv \

# vamb \
#     --model vaevae \
#     --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out_32_${run_id}_radius \
#     --fasta /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/contigs_2kbp.fna.gz \
#     --rpkm /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/abundance.npz \
#     --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/${dataset}_taxonomy_${run_id}.tsv \
#     -l 32 \
#     -e 200 \
#     -q 25 75 150 \
#     -pe 100 \
#     -pq 25 75 \
#     -o C \
#     --cuda \
#     --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out_32_${run_id}_radius/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out_32_${run_id}_radius/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/contigs_2kbp.fna.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out_32_reclustering_${run_id}_radius \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_cami_${dataset}.hmmout \
    --algorithm kmeans \
    --minfasta 200000

#!/usr/bin/bash
module load tools computerome_utils/2.0
module unload gcc
module load gcc/11.1.0
module load anaconda3/4.4.0
module load minimap2/2.17r941 samtools/1.10

source ~/.bashrc
conda init bash
conda activate vamb_env

# contigs
contigs_path=/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/contigs.fna

query_db=queryDBs/airways_contigs_db
seqTaxDB=/home/projects/cpr_10006/people/paupie/tax_vamb/mmseq2/airways/seqTaxDBs/gtdb_db
tax_result=airways_taxonomy
tax_result_tsv=airways_taxonomy.tsv

#ls $contigs_path |head

# create query database
mmseqs createdb  $contigs_path $query_db

# Taxonomy assignment
mmseqs taxonomy  $query_db  $seqTaxDB  $tax_result  tmp  --threads 25  --tax-lineage 1

# Taxonomy output and TSV
mmseqs createtsv $query_db $tax_result $tax_result_tsv
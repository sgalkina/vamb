#!/usr/bin/bash

run_id=vamb_test_cpu
qsub -d `pwd` -l nodes=1:ppn=4:gpus=1,mem=20gb,walltime=1:00:00:00 -r y -N $run_id -e Out/$run_id.err -o Out/$run_id.out -A cpr_10006 -W group_list=cpr_10006 computerome.sh
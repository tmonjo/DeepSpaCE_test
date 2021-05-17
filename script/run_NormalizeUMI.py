#! /usr/bin/python
# coding=utf-8

import subprocess
import pandas as pd

pheno_file_name = "./sample_list.txt"

sample_list = pd.read_csv(pheno_file_name, sep='\t', comment='#')

for i in range(len(sample_list)):
    
    f = open("./cluster_scripts/run_NormalizeUMI__"+sample_list.sampleName[i]+".sh", 'w')

    f.write("""#$ -S /usr/bin/bash
#$ -cwd
#$ -V
#$ -pe def_slot 8
#$ -l s_vmem=8G,mem_req=8G,os7
""")
    
    f.write("""
export OMP_NUM_THREADS=8
""")

    f.write("""
/usr/local/package/r/3.6.0/bin/Rscript ./NormalizeUMI.R /home/monjo/DeepSpaCE/data """+sample_list.sampleName[i]+""" 1000 1000
""")

    f.close()

    subprocess.call(["qsub","-o","./out","-e","./err","./cluster_scripts/run_NormalizeUMI__"+sample_list.sampleName[i]+".sh"])

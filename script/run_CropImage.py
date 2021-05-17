#! /usr/bin/python
# coding=utf-8

import subprocess
import pandas as pd

subprocess.call(["sed","-i","-e"'s/args=\[\]//',"CropImage.py"])

pheno_file_name = "./sample_list.txt"

sample_list = pd.read_csv(pheno_file_name, sep='\t', comment='#')

for i in range(len(sample_list)):
    
    f = open("./cluster_scripts/run_CropImage__"+sample_list.sampleName[i]+".sh", 'w')

    f.write("""#$ -S /usr/bin/bash
#$ -cwd
#$ -V
#$ -pe def_slot 8
#$ -l s_vmem=8G,mem_req=8G,os7
""")
    
    f.write("""
export OMP_NUM_THREADS=8

/home/monjo/.local/bin/pipenv shell
""")

    for extraSize in [150]:
        for quantileRGB in [80]:
            f.write("""
python ./CropImage.py --rootDir /home/monjo/DeepSpaCE/data --sampleName """+sample_list.sampleName[i]+""" --transposeType """+str(sample_list.transposeType[i])+""" --radiusPixel """+str(sample_list.radiusPixel[i])+""" --extraSize """+str(extraSize)+""" --quantileRGB """+str(quantileRGB)+"""
""")

    f.close()

    subprocess.call(["qsub","-o","./out","-e","./err","./cluster_scripts/run_CropImage__"+sample_list.sampleName[i]+".sh"])

# DeepSpaCE

The Deep learning model for Spatial gene Clusters and Expression (DeepSpaCE) is a method that predicts spatial gene-expression levels and transcriptomic cluster types from tissue section images using deep learning.


# Table of Contents
- [Requirement](#requirement)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

# Requirement
* Singularity >= 3.7

# Requirement (if Singularity is not availeble)
* Python 3.6
* R 3.6

# Installation
## Setup Python and R environment (if Singularity is not availeble)
Create a Python environment ([Pipenv](https://pipenv.pypa.io/))
    
    pipenv install Pipfile    
    
Create an R environment


# Usage
## Preprocessing 1: Section image files

    singularity exec DeepSpaCE.sif \
        python CropImage.py \
            --rootDir /home/$USER/DeepSpaCE/data \
            --sampleName Human_Breast_Cancer_Block_A_Section_1 \
            --transposeType 0 \
            --radiusPixel 75 \
            --extraSize 150 \
            --quantileRGB 80


## Preprocessing 2: Satial expression data measured by Visium

    singularity exec DeepSpaCE.sif \
        Rscript NormalizeUMI.R \
            --rootDir /home/$USER/DeepSpaCE/data \
            --sampleName Human_Breast_Cancer_Block_A_Section_1 \
            --threshold_count 1000 \
            --threshold_gene 1000

## Run DeepSpaCE
    singularity exec DeepSpaCE.sif \
        python DeepSpaCE.py \
            --rootDir　/home/$USER/DeepSpaCE/ \
            --dataDir /home/$USER/DeepSpaCE/data \
            --batch_size 128 \
            --num_epochs 10 \
            --lr 1e-4 \
            --weight_decay 1e-4 \
            --model vgg16 \
            --clusteringMethod graphclust \
            --rm_cluster \
            --cuda \
            --full \
            --quantileRGB 80 \
            --seed 0 \
            --threads 8 \
            --GPUs 1 \
            --early_stop_max 5 \
            --extraSize 150 \
            --sampleNames_train Human_Breast_Cancer_Block_A_Section_1 \
            --sampleNames_test Human_Breast_Cancer_Block_A_Section_1 \
            --sampleNames_semi None \
            --semi_option normal \
            --geneSymbols ESR1,ERBB2,MKI67 \
            --augmentation flip,crop,color,random \
            --cross_index 0

# Citation
The DeepSpaCE pre-print:  
Taku Monjo, Masaru Koido, Satoi Nagasawa, Yutaka Suzuki, and Yoichiro Kamatani “Efficient prediction of a spatial transcriptomics profile better characterizes breast cancer tissue sections without costly experimentation” bioRxiv (2021)
https://www.biorxiv.org/content/10.1101/2021.04.22.440763v1


# License
GNU General Public License v3.0


# DeepSpaCE

The Deep learning model for Spatial gene Clusters and Expression (DeepSpaCE) is a method that predicts spatial gene-expression levels and transcriptomic cluster types from tissue section images using deep learning.


# Table of Contents
- [Requirement](#requirement)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

# Requirement
* Singularity >=3.7

# Requirement (if Singularity is not availeble)
* Python 3.6
* R 3.6

# Installation
## 1. Create a Python environment ([Pipenv](https://pipenv.pypa.io/))

    pipenv install Pipfile
    
## 2. Create an R environment



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
        




# Citation
...

# License
GNU General Public License v3.0


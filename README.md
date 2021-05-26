# DeepSpaCE

The Deep learning model for Spatial gene Clusters and Expression (DeepSpaCE) is a method that predicts spatial gene-expression levels and transcriptomic cluster types from tissue section images using deep learning.


# Table of Contents
- [Requirement](#requirement)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

# Requirement
* Python 3.6
* R 3.6

# Installation
## 1. Create a Python environment ([Pipenv](https://pipenv.pypa.io/))

    pipenv install Pipfile
    
## 2. Create an R environment



# Usage
## Preprocessing of section image files

    python CropImage.py \
        --rootDir /home/$USER/DeepSpaCE/data \
        --sampleName Human_Breast_Cancer_Block_A_Section_1 \
        --transposeType 0 \
        --radiusPixel 75 \
        --extraSize 150 \
        --quantileRGB 80


## Preprocessing of spatial expression data measured by Visium

    Rscript NormalizeUMI.R \
        --rootDir /home/$USER/DeepSpaCE/data \
        --sampleName Human_Breast_Cancer_Block_A_Section_1 \
        --threshold_count 1000 \
        --threshold_gene 1000

## Run DeepSpaCE

    python DeepSpaCE.py [options]




# Citation
...

# License
GNU General Public License v3.0


#!/usr/bin/env python
# coding: utf-8

# In[5]:


import glob
import os.path as osp
import random
import numpy as np
import pandas as pd
import json
import pickle
import sys
import time
import math
import subprocess
import argparse

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline
matplotlib.use('Agg')

from matplotlib.ticker import MaxNLocator 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold

from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix

import csv
import gzip
import os
import scipy.io
import cv2

import albumentations as albu
from albumentations.pytorch import ToTensor

#subprocess.call(['cp','-rp','/home/monjo/Visium/PredictGene/script/BasicLib.py','.'])
#subprocess.call(['cp','-rp','/home/monjo/Visium/PredictGene/script/DeepSpaceLib.py','.'])
sys.path.append('./')

from DeepSpaceLib import makeDataList
from DeepSpaceLib import makeTrainDataloader
from DeepSpaceLib import make_model
from DeepSpaceLib import run_train
from DeepSpaceLib import makeTestDataloader
from DeepSpaceLib import run_test
from DeepSpaceLib import makeDataListSemi
from DeepSpaceLib import makeSemiDataloader
from DeepSpaceLib import predict_semi_label


# In[ ]:





# In[71]:


parser = argparse.ArgumentParser(description='Predict Gene')
parser.add_argument('--rootDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/')
parser.add_argument('--dataDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/data')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train (default: 100')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--model', type=str, choices=['vgg16'], default='vgg16',
                    help='vgg16')
parser.add_argument('--clusteringMethod', type=str, choices=['graphclust', 'kmeans_2_clusters', 'kmeans_3_clusters', 'kmeans_4_clusters', 'kmeans_5_clusters', 'kmeans_6_clusters', 'kmeans_7_clusters','kmeans_8_clusters', 'kmeans_9_clusters', 'kmeans_10_clusters'], default='graphclust')
parser.add_argument('--rm_cluster', type=str, default='')
parser.add_argument('--cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--full', action='store_true',
                    help='enables full training')
parser.add_argument('--quantileRGB', type=int, default=80)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--GPUs', type=int, default=1)
parser.add_argument('--early_stop_max', type=int, default=5,
                    help='how many epochs to wait for loss improvement (default: 5)')
parser.add_argument('--extraSize', type=int, default=150)
parser.add_argument('--sampleNames_train', type=str, default='Human_Breast_Cancer_Block_A_Section_1')
parser.add_argument('--sampleNames_test', type=str, default='Human_Breast_Cancer_Block_A_Section_1')
parser.add_argument('--sampleNames_semi', type=str, default='None')
parser.add_argument('--semi_option', type=str, choices=['normal', 'random', 'permutation'], default='normal')
parser.add_argument('--geneSymbols', type=str, default='SPARC,IFI27,COL10A1,COL1A2,COL3A1,COL5A2,FN1,POSTN,CTHRC1,COL1A1,THBS2,PDGFRL,COL8A1,SULF1,MMP14,ISG15,IL32,MXRA5,LUM,DPYSL3,CTSK')
parser.add_argument('--augmentation', type=str, default='flip,crop,color,random')
parser.add_argument('--cross_index', type=int, default=0)
parser.add_argument('--ClusterPredictionMode', action='store_true')

args = parser.parse_args()


# In[72]:


print(args)


# In[73]:


rootDir = args.rootDir
print("rootDir: "+str(rootDir))

dataDir = args.dataDir
print("dataDir: "+str(dataDir))

batch_size = args.batch_size * args.GPUs
print("batch_size: "+str(batch_size))

num_epochs = args.num_epochs
print("num_epochs: "+str(num_epochs))

lr = args.lr
print("lr: "+str(lr))

weight_decay = args.weight_decay
print("weight_decay: "+str(weight_decay))

model = args.model
print("model: "+str(model))

clusteringMethod = args.clusteringMethod
print("clusteringMethod: "+str(clusteringMethod))

cuda = args.cuda and torch.cuda.is_available()
print("cuda: "+str(cuda))

full = args.full
full = True
print("full: "+str(full))

quantileRGB = args.quantileRGB
print("quantileRGB: "+str(quantileRGB))

seed = args.seed
print("seed: "+str(seed))

threads = args.threads
print("threads: "+str(threads))

early_stop_max = args.early_stop_max
print("early_stop_max: "+str(early_stop_max))

extraSize = args.extraSize
print("extraSize: "+str(extraSize))

augmentation = args.augmentation
print("augmentation: "+augmentation)

semi_option = args.semi_option
print("semi_option: "+str(semi_option))

cross_index = args.cross_index
print("cross_index: "+str(cross_index))

ClusterPredictionMode = args.ClusterPredictionMode
print("ClusterPredictionMode: "+str(ClusterPredictionMode))


# In[74]:


if args.rm_cluster == '':
    rm_cluster = ''
else:
    rm_cluster = [int(i) for i in args.rm_cluster.split(',')]

print("rm_cluster: "+str(rm_cluster))


# In[48]:


sampleNames_train = args.sampleNames_train.split(',')
print("sampleNames_train: "+str(sampleNames_train))


# In[49]:


sampleNames_test = args.sampleNames_test.split(',')
print("sampleNames_test: "+str(sampleNames_test))


# In[50]:


if sampleNames_train == sampleNames_test:
    train_equals_test = True
else:
    train_equals_test = False

print("train_equals_test: "+str(train_equals_test))


# In[51]:


sampleNames_semi = args.sampleNames_semi.split(',')
print("sampleNames_semi: "+str(sampleNames_semi))


# In[52]:


geneSymbols = args.geneSymbols.split(',')
print(geneSymbols)


# In[53]:


size = 224
print("size: "+str(size))

mean = (0.485, 0.456, 0.406)
print("mean: "+str(mean))

std = (0.229, 0.224, 0.225)
print("std: "+str(std))


# In[54]:


print("### Set seeds ###")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.set_num_threads(threads)


# In[55]:


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[56]:


print("### Check GPU availability ###")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# # make data_list (teacher)

# In[57]:


data_list_teacher = makeDataList(rootDir=dataDir,
                                 sampleNames=sampleNames_train,
                                 clusteringMethod=clusteringMethod,
                                 extraSize=extraSize,
                                 geneSymbols=geneSymbols,
                                 quantileRGB=quantileRGB,
                                 seed=seed,
                                 cross_index=cross_index,
                                 train_equals_test=train_equals_test,
                                 is_test=False,
                                 rm_cluster=rm_cluster)

if train_equals_test:
    data_list_teacher_tmp = data_list_teacher.copy()
    data_list_teacher = data_list_teacher_tmp.query('phase != "test"').copy()


data_list_teacher.to_csv("../out/data_list_teacher.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_teacher: "+str(data_list_teacher.shape))
data_list_teacher.head()


# # make dataloader (teacher)

# In[58]:


dataloaders_dict_teacher = makeTrainDataloader(rootDir=rootDir,
                                               data_list_df=data_list_teacher,
                                               geneSymbols=geneSymbols,
                                               size=size,
                                               mean=mean,
                                               std=std,
                                               augmentation=augmentation,
                                               batch_size=batch_size,
                                               ClusterPredictionMode=ClusterPredictionMode)

print("### save dataloader ###")
with open("../out/dataloaders_dict_teacher.pickle", mode='wb') as f:
    pickle.dump(dataloaders_dict_teacher, f)


# In[ ]:





# # make network model (teacher)

# In[59]:


print("### make model ###")
if ClusterPredictionMode:
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(data_list_teacher['Cluster'].unique()),
                                       full=full)
else:
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(geneSymbols),
                                       full=full)


# # set optimizer

# In[60]:


print("### set optimizer ###")
optimizer = optim.Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)


# In[ ]:





# # Training (teacher)

# In[61]:


print("### run train ###")
run_train(net=net,
          dataloaders_dict=dataloaders_dict_teacher,
          optimizer=optimizer,
          num_epochs=num_epochs,
          device=device,
          early_stop_max=early_stop_max,
          name='teacher',
          ClusterPredictionMode=ClusterPredictionMode)


# In[ ]:





# # Test (teacher)

# In[65]:


data_list_test = makeDataList(rootDir=dataDir,
                              sampleNames=sampleNames_test,
                              clusteringMethod=clusteringMethod,
                              extraSize=extraSize,
                              geneSymbols=geneSymbols,
                              quantileRGB=quantileRGB,
                              seed=seed,
                              cross_index=cross_index,
                              train_equals_test=True,
                              is_test=True,
                              rm_cluster=rm_cluster)

if train_equals_test:
    data_list_test = data_list_teacher_tmp.query('phase == "test"').copy()


data_list_test['phase'] = 'valid'
data_list_test.to_csv("../out/data_list_test.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_test: "+str(data_list_test.shape))
data_list_test.head()


# In[66]:


dataloaders_dict_test = makeTestDataloader(rootDir=rootDir,
                                           data_list_df=data_list_test,
                                           geneSymbols=geneSymbols,
                                           size=size,
                                           mean=mean,
                                           std=std,
                                           augmentation=augmentation,
                                           batch_size=batch_size,
                                           ClusterPredictionMode=ClusterPredictionMode)

# save dataloader
with open("../out/DataLoader_test.pickle", mode='wb') as f:
    pickle.dump(dataloaders_dict_test, f)


# In[67]:


### Test ###
data_list_test_teacher, net_best = run_test(data_list_df=data_list_test,
                                            dataloaders_dict=dataloaders_dict_test,
                                            geneSymbols=geneSymbols,
                                            device=device,
                                            name="teacher",
                                            ClusterPredictionMode=ClusterPredictionMode,
                                            rm_cluster=rm_cluster)

data_list_test_teacher.to_csv("../out/data_list_test_teacher.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_test_teacher: "+str(data_list_test_teacher.shape))
data_list_test_teacher.head()


# # Semi-supervised

# In[ ]:


if sampleNames_semi == ["None"]:
    sys.exit()
else:
    print("### Semi-supervised ###")


# In[ ]:


ImageSet = [[0,1],[2,3],[4,5],[6,7],[8,9]]
print("ImageSet: "+str(ImageSet))


# In[ ]:





# In[ ]:


for i_semi in range(5):
    if sampleNames_semi == ["TCGA"]:
        data_list_semi = makeDataListSemi(rootDir=rootDir,
                                          sampleNames=sampleNames_semi,
                                          semiType="TCGA",
                                          ImageSet=ImageSet[i_semi],
                                          semiName="semi"+str(i_semi+1))
    elif sampleNames_semi == ["ImageNet"]:
        data_list_semi = makeDataListSemi(rootDir=rootDir,
                                          sampleNames=sampleNames_semi,
                                          semiType="ImageNet",
                                          ImageSet=ImageSet[i_semi],
                                          semiName="semi"+str(i_semi+1))
    else:
        data_list_semi = makeDataListSemi(rootDir=rootDir,
                                          sampleNames=sampleNames_semi,
                                          semiType="Visium",
                                          ImageSet=ImageSet[i_semi],
                                          semiName="semi"+str(i_semi+1))

    # make semi dictionary
    dataloaders_dict_semi = makeSemiDataloader(rootDir=rootDir,
                                               data_list_df=data_list_semi,
                                               size=size,
                                               mean=mean,
                                               std=std,
                                               augmentation=augmentation,
                                               batch_size=batch_size)

    # save semi dataloader
    with open("../out/dataloaders_dict_semi"+str(i_semi+1)+".pickle", mode='wb') as f:
        pickle.dump(dataloaders_dict_semi, f)

    # predict semi labels
    data_list_semi = predict_semi_label(net=net_best,
                                        data_list_semi=data_list_semi,
                                        dataloaders_dict_semi=dataloaders_dict_semi,
                                        geneSymbols=geneSymbols,
                                        ClusterPredictionMode=ClusterPredictionMode)
    
    if ClusterPredictionMode:
        if semi_option == "permutation":
            data_list_semi['Cluster'] = random.sample(data_list_semi['Cluster'].tolist(), len(data_list_semi))
        
        elif semi_option == "random":
            data_list_semi['Cluster'] = random.randint(0, max(data_list_semi['Cluster']))
        
        elif semi_option == "normal":
            pass
    else:
        if semi_option == "permutation":
            for gene in geneSymbols:
                data_list_semi[gene] = minmax_scale(data_list_semi[gene])
                data_list_semi[gene] = random.sample(data_list_semi[gene].tolist(), len(data_list_semi))

        elif semi_option == "random":
            for gene in geneSymbols:
                data_list_semi[gene] = [random.uniform(0, 1) for i in range(len(data_list_semi))]

        elif semi_option == "normal":
            for gene in geneSymbols:
                data_list_semi[gene] = minmax_scale(data_list_semi[gene])


    print("data_list_semi"+str(i_semi+1)+": "+str(data_list_semi.shape))
    data_list_semi.head()

    data_list_semi.to_csv("../out/data_list_semi"+str(i_semi+1)+".txt", index=False, sep='\t', float_format='%.6f')


    ## add semi dataset
    data_list_student = pd.concat([data_list_teacher, data_list_semi])
    data_list_teacher = data_list_student.copy()
    
    data_list_student = data_list_student.reset_index(drop=True)
    
    ### 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    count = 0
    for train_index, test_index in kf.split(data_list_student.index, data_list_student.index):
        if count == cross_index:
            data_list_student.loc[data_list_student.index[train_index], 'phase'] = 'train'
            data_list_student.loc[data_list_student.index[test_index], 'phase'] = 'valid'

        count += 1
    
    data_list_student.to_csv("../out/data_list_student"+str(i_semi+1)+".txt", index=False, sep='\t', float_format='%.6f')

    print("data_list_student"+str(i_semi+1)+": "+str(data_list_student.shape))
    data_list_student.head()

    
    dataloaders_dict_student = makeTrainDataloader(rootDir=rootDir,
                                                   data_list_df=data_list_student,
                                                   geneSymbols=geneSymbols,
                                                   size=size,
                                                   mean=mean,
                                                   std=std,
                                                   augmentation=augmentation,
                                                   batch_size=batch_size,
                                                   ClusterPredictionMode=ClusterPredictionMode)

    print("### save dataloader ###")
    with open("../out/dataloaders_dict_student"+str(i_semi+1)+".pickle", mode='wb') as f:
        pickle.dump(dataloaders_dict_student, f)

    
    print("### make model ###")
    if ClusterPredictionMode:
        net, params_to_update = make_model(use_pretrained=True,
                                           num_features=len(data_list_teacher['Cluster'].unique()),
                                           full=full)
    else:
        net, params_to_update = make_model(use_pretrained=True,
                                           num_features=len(geneSymbols),
                                           full=full)
    
    print("### set optimizer ###")
    optimizer = optim.Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)

    
    ### Training ###
    print("### run train ###")
    run_train(net=net,
              dataloaders_dict=dataloaders_dict_student,
              optimizer=optimizer,
              num_epochs=num_epochs,
              device=device,
              early_stop_max=early_stop_max,
              name='student'+str(i_semi+1),
              ClusterPredictionMode=ClusterPredictionMode)

    
    ### validation ###
    data_list_test_student, net_best = run_test(data_list_df=data_list_test,
                                                dataloaders_dict=dataloaders_dict_test,
                                                geneSymbols=geneSymbols,
                                                device=device,
                                                name="student"+str(i_semi+1),
                                                ClusterPredictionMode=ClusterPredictionMode,
                                                rm_cluster=rm_cluster)

    data_list_test_student.to_csv("../out/data_list_test_student"+str(i_semi+1)+".txt", index=False, sep='\t', float_format='%.6f')

    print("data_list_test_student"+str(i_semi+1)+": "+str(data_list_test_student.shape))
    data_list_test_student.head() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





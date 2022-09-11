# COCOA
## COCOA: Cross Modality Contrastive Learning for Sensor Data [COCOA paper](https://dl.acm.org/doi/10.1145/3550316)

### Abstract
Self-Supervised Learning (SSL) is a new paradigm for learning discriminative representations without labeled data and has reached comparable or even state-of-the-art results compared to supervised counterparts. Contrastive Learning (CL) is one
of the most well-known approaches in SSL that attempts to learn general, informative representations of data. CL methods have been mostly developed for applications in computer vision and natural language processing where only a single sensor
modality is used. A majority of pervasive computing applications, however, exploit data from a range of different sensor modalities. While existing CL methods are limited to learning from one or two data sources, we propose COCOA (Cross mOdality
COntrastive leArning), a self-supervised model that employs a novel objective function to learn quality representations from multisensor data by computing the cross-correlation between different data modalities and minimizing the similarity
between irrelevant instances. We evaluate the effectiveness of COCOA against eight recently introduced state-of-the-art self-supervised models and two supervised baselines across five public datasets. We show that COCOA achieves superior
classification performance to all other approaches. Also, COCOA is far more label-efficient than the other baselines, including the fully supervised model using only one-tenth of available labeled data

## Model architecture 
 ![alt text](https://github.com/cruiseresearchgroup/COCOA/blob/main/images/COCOA.png?raw=true)


## Positive and Negative selection
 ![alt text](https://github.com/cruiseresearchgroup/COCOA/blob/main/images/sampling.pdf)

### Usage
    main.py [-h] --datapath DATAPATH --output OUTPUT [--dataset DATASET]
               [--loss LOSS] [--sim SIM] [--gpu GPU] [--tsne TSNE]
               [--window WINDOW] [--labeleff LABELEFF] [--code CODE]
               [--beta BETA] [--epoch EPOCH] [--batch BATCH]
               [--eval_freq EVAL_FREQ] [--temp TEMP] [--tau TAU] [--lr LR]
               [--mode MODE]


### Cite
    @article{10.1145/3550316,
    author = {Deldari, Shohreh and Xue, Hao and Saeed, Aaqib and Smith, Daniel V. and Salim, Flora D.},
    title = {COCOA: Cross Modality Contrastive Learning for Sensor Data},
    year = {2022},
    publisher = {Association for Computing Machinery},
    volume = {6},
    number = {3},
    url = {https://doi.org/10.1145/3550316},
    doi = {10.1145/3550316},
    journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
    articleno = {108},
    numpages = {28}
    }

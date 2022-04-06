# RODD Official Implementation of 2022 CVPRW Paper
## RODD: A Self-Supervised Approach for Robust Out-of-Distribution Detection
Introduction: 
Recent studies have addressed the concern of detecting and rejecting the out-of-distribution (OOD) samples as a major challenge in the safe deployment of deep learning (DL) models. It is desired that the DL model should only be confident about the in-distribution (ID) data which reinforces the driving principle of the OOD detection. In this work, we propose a simple yet effective generalized OOD detection method independent of out-of-distribution
datasets. Our approach relies on self-supervised feature learning of the training samples, where the embeddings lie on a compact low-dimensional space. Motivated by the recent studies that show self-supervised adversarial contrastive learning helps robustifying the model, we empirically show that a pre-trained model with selfsupervised contrastive learning yields a better model for uni-dimensional feature learning in the latent space. The method proposed in this work, referred to as RODD, outperforms SOTA detection performance on extensive suite of benchmark datasets on OOD detection tasks.
## Dataset Preparation
### In-Distribution Datasets
CIFAR-10 and CIFAR-100 are in-distribution datasets which will be automatically downloaded during training
### OOD Datasets
Create a folder 'data' in the root 'RODD' folder<br />
Download following OOD datasets in the 'data' folder. <br />
[Places](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz)<br />
[Textures (Download the entire dataset)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)<br />
All other OOD Datasets such as ImagNetc, ImageNetr, LSUNr, LSUNc, iSUN and SVHN can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1MLz5C3EjQbAd1M2yktviM0qENXg4jvfz?usp=sharing)
## **Pre-Training**
### For CIFAR-10:
```bash
python pretrain.py --dataset cifar10
```
### For CIFAR-100:
```bash
python pretrain.py --dataset cifar100
```
## **Fine-Tuning**
### For CIFAR-10:
```bash
python fine_tune.py --dataset cifar10
```
### For CIFAR-100:
```bash
python fine_tune.py --dataset cifar100
```
## **Evaluation**
### For CIFAR-10:
```bash
python extract_features in-dataset cifar10
```
```bash
python evaluate_original
```
### For CIFAR-100:
```bash
python extract_features in-dataset cifar100
```
```bash
python evaluate_original
```

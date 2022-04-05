# RODD
## RODD: A Self-Supervised Approach for Robust Out-of-Distribution Detection
Introduction: 
Recent studies have addressed the concern of detecting and rejecting the out-of-distribution (OOD) samples as a major challenge in the safe deployment of deep learning (DL) models. It is desired that the DL model should only be confident about the in-distribution (ID) data which reinforces the driving principle of the OOD detection. In this work, we propose a simple yet effective generalized OOD detection method independent of out-of-distribution
datasets. Our approach relies on self-supervised feature learning of the training samples, where the embeddings lie on a compact low-dimensional space. Motivated by the recent studies that show self-supervised adversarial contrastive learning helps robustifying the model, we empirically show that a pre-trained model with selfsupervised contrastive learning yields a better model for uni-dimensional feature learning in the latent space. The method proposed in this work, referred to as RODD, outperforms SOTA detection performance on extensive suite of benchmark datasets on OOD detection tasks.
## **Pre-Training**
### For CIFAR-10:
python pretrain.py --dataset cifar10
### For CIFAR-100:
python pretrain.py --dataset cifar100
## **Fine-Tuning**
### For CIFAR-10:
python fine_tune.py --dataset cifar10
### For CIFAR-100:
python fine_tune.py --dataset cifar100
## **Evaluation**
### For CIFAR-10:
python extract_features in-dataset cifar10
python evaluate_original
### For CIFAR-100:
python extract_features in-dataset cifar100
python evaluate_original

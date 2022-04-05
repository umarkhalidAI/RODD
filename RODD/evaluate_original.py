from __future__ import print_function
import numpy as np
from sklearn import metrics
import argparse

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

parser = argparse.ArgumentParser(description='Evaluating Out-of-distribution Detection')
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
parser.add_argument('--featurepath', default="./features/WideResNet40-CIFAR10-RODD", type=str,
                    help='path to model')


def main():
    args = parser.parse_args()
    exp_dir = args.featurepath
    train = np.squeeze(np.load(exp_dir + '/featuresTrain_in.npy'))
    test_in = np.squeeze(np.load(exp_dir + '/featuresTest_in.npy'))
    labels = np.squeeze(np.load(exp_dir + '/labelsTrain_in.npy'))
    data_out = ['LSUN_crop', 'LSUN_resize', 'Places', 'iSUN', 'dtd', 'svhn', 'Imagenet_crop', 'Imagenet_resize']
    for dataName in data_out:
        out_data = dataName
        process_features(exp_dir, out_data)

def process_features(exp_dir, out_dataset, train, test_in, labels):
    test = np.squeeze(np.load(exp_dir + '/features_out_' + out_dataset + '.npy'))
    ###################################
    train = preprocess(train, labels)
    test = preprocess(test)
    test_in = preprocess(test_in)
    ####################################
    first_sing_vecs = []
    for l, data in enumerate(train):
        if len(data) != 0:
            u = calculate_sing_vec(data)
            first_sing_vecs.append(u)

    first_sing_vecs = np.array(first_sing_vecs)
    score_in = OOD_test(test_in, score_func, first_sing_vecs)
    target_in = np.zeros_like(score_in)
    score_out = OOD_test(test, score_func, first_sing_vecs)
    # print(np.mean(score_in) / (np.mean(score_in) + np.mean(score_out)),
    #       np.mean(score_out) / (np.mean(score_in) + np.mean(score_out)))
    target_out = np.ones_like(score_out)
    targets = np.concatenate((target_in, target_out))
    scores = np.concatenate((score_in, score_out))

    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    # print(thresholds)
    fpr95 = fpr[tpr >= 0.95][0]
    print('FPR @95TPR:', fpr95)

    print('Detection Error:', np.min(0.5 * (1 - tpr) + 0.5 * fpr))
    AUROC = metrics.auc(fpr, tpr)
    print('AUC: ', AUROC)
    return fpr95, AUROC


def score_func(D, first_sing_vecs):
    measure = first_sing_vecs
    corr = correlation(measure, D)
    score = np.arccos(corr)
    if len(corr.shape) == 3:
        score = np.min(score, axis=1)
        score = np.min(score, axis=0)

    elif len(corr.shape) == 2:
        score = np.min(score, axis=0)

    return score


def calculate_sing_vec(A):
    try:
        import irlb
        # print('irlb package is installed for fast svd, using irlb')
        USV = irlb.irlb(A, 2)
    except ImportError:
        # print('No irlb package installed for fast svd, using numpy')
        USV = np.linalg.svd(A)
    first_sing_vec = USV[0][:, 0]
    return first_sing_vec


def preprocess(D, labels=None):
    if labels is None:
        data = np.array(D)
        D_out = data.transpose()
    else:
        D_out = []
        for l in set(labels):

            data = np.array(D[labels == l])
            if len(data) != 0:
                D_out.append(data.transpose())
    return D_out


def correlation(A, B):
    corr = np.matmul(A, B)
    if len(B.shape) == 2:
        corr /= np.linalg.norm(B, axis=0) + 1e-4
    elif len(B.shape) == 3:
        corr /= np.linalg.norm(B, axis=1)[:, None, :] + 1e-8
    corr = np.abs(corr)
    return corr


def OOD_test(D, score_func, first_sing_vecs):
    score = score_func(D, first_sing_vecs)
    return score


if __name__ == '__main__':
    main()



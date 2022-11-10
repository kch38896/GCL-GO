from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
from pathlib import Path
import pickle


def auprc(ytrue, ypred):
    p, r, t = precision_recall_curve(ytrue, ypred)
    return auc(r, p)


def main(ytrue1, ypred1):
    fmax = 0
    prec = []
    rc = []
    ytrue = []
    ypred = []

    for i in range(len(ytrue1)):
        if np.sum(ytrue1[i]) > 0:
            ytrue.append(ytrue1[i])
            ypred.append(ypred1[i])

    for t in range(1, 101):
        thres = t / 100.

        thres_value = np.ones((len(ytrue), len(ytrue[0])), dtype=np.float32) * thres

        pred_values = np.greater(ypred, thres_value).astype(int)

        tp_matrix = pred_values * ytrue

        tp = np.sum(tp_matrix, axis=1, dtype=np.int32)
        tpfp = np.sum(pred_values, axis=1)
        tpfn = np.sum(ytrue, axis=1)

        avgprs = []

        for i in range(len(tp)):
            if tpfp[i] != 0:
                avgprs.append(tp[i] / float(tpfp[i]))

        if len(avgprs) == 0:
            continue
        avgpr = np.mean(avgprs)
        avgrc = np.mean(tp / tpfn)
        avgpr = float(avgpr)
        avgrc = float(avgrc)

        prec.append(avgpr)
        rc.append(avgrc)
        if avgrc == 0 and avgpr == 0:
            f1 = 0
        else:
            f1 = 2 * avgpr * avgrc / (avgpr + avgrc)

        fmax = max(fmax, f1)

    return fmax, auprc(np.array(ytrue).flatten(), np.array(ypred).flatten())


if __name__ == '__main__':
    ytrue = np.load('Data/CAFA3/test/MFO/MFO_true.npy')
    ypred = np.load('Data/CAFA3/test/MFO/MFO_pred_epoch19.npy')
    print(main(ytrue, ypred))



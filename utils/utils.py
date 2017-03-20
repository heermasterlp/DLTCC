import numpy as np


# return the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readable()]

    # print prob
    pred = np.argsort(prob[::-1])

    # get top1 label
    top1 = synset[pred[0]]
    print("Top1:", top1, prob[pred[0]])
    return top1


# print the prob
def print_prob(prob):
    # print prob
    print(max(prob[0]), " : ", min(prob[0]))
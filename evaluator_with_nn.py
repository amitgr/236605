from random import Random

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import cross_val_score, cross_val_predict

import common
from common import TripletGenerator, DebletGenerator
import itertools
import numpy as np

import map_reducer

random = Random(5)


def TrainModelAndEvaluate(train_set, fitter, negative_weight):
  def ToStringFeaturesTokenized(key_fn):
    hasher = FeatureHasher(input_type='string')
    raw_X = [list(k) for k in [key_fn(x) for x in train_set]]
    return hasher.transform(raw_X)
  #
  # random.shuffle(train_set)
  # brand_features = ToStringFeatures(lambda x: x.Brand)
  # gtin_features = ToStringFeaturesTokenized(lambda x: x.GTIN)
  #
  # X = [x.ToNumericFeatures() for x in train_set]
  # from scipy.sparse import hstack
  # X = hstack([X, brand_features])
  # # X = hstack([X, brand_features, gtin_features])
  # y = [x.result for x in train_set]
  # y = np.asarray(y)
  # from sklearn.preprocessing import normalize
  # X = normalize(X, axis=0)
  # print(X.shape)
  # print(y.shape)
  # # print(cross_val_score(fitter, X, y, cv=10))
  # negative_weights = list([1 if x.result else negative_weight for x in train_set])
  conf_mat = common.cross_val(train_set, fitter, negative_weight, cv=5)
  tn = conf_mat.tn
  fn = conf_mat.fn
  tp = conf_mat.tp
  fp = conf_mat.fp
  if tn + fn < 50:
    print('Did not tag enough negative samples, only ', tn + fn)
    return 0, 0
  print('sum True Negatives: ', tn, ' ; sum False Negatives: ', fn, ' ; sum True Positives: ', tp,
        ' ; sum False Positives: ', fp, )
  precision = tn / (tn + fn)
  print('Precision of negatives: ', precision)
  recall = tn / (tn + fp)
  print('Recall of negatives: ', recall)
  return conf_mat.nprecision(), conf_mat.nrecall()


def LogisticRegressionLearnAndEvaluate(train_set):
  from sklearn.linear_model import LogisticRegression
  regularization = ['l1', 'l2']
  c_values = [1000, 300, 100, 30, 10, 3, 1, 0.3, 0.1, 0.03]
  negative_weights = list(np.arange(start=0.1, stop=10, step=0.1)) # + list(np.arange(start=1, stop=4, step=0.25))
  c_values = [10, 100, 300, 1000]
  regularization = ['l1']
  # negative_weights=[0.5,1]
  dict = {}
  for l, c, negative_weight in itertools.product(regularization, c_values, negative_weights):
    print(l, ': c=', c, ' negative weight=', negative_weight)
    lr = LogisticRegression(class_weight={True: 1, False: negative_weight}, penalty=l, C=c)
    precision, recall = TrainModelAndEvaluate(train_set, lr)
    if (precision, recall) != (0, 0):
      dict[(l, c)] = dict.get((l, c), [])
      dict[(l, c)].append((precision, recall))
  return dict


def SvmLearnAndEvaluate(train_set):
  from sklearn import svm
  kernels = ['linear', 'poly', 'rbf', 'sigmoid']
  kernels = ['poly']

  # first run settings:
  negative_weights = np.arange(start=0.5, stop=4, step=0.5)
  c_values = [0.01, 0.1, 1, 10, 100]
  gamas = [0.01, 0.1, 1, 10, 100]
  coefs = [-100, -10, -1, -0.1, 0.0, 0.1, 1, 10, 100]

  # rbf:
  #negative_weights = np.arange(start=0.2, stop=5, step=0.1)
  #c_values = [0.1, 0.3, 1, 10, 30, 100]
  #gamas=[3, 10, 30]
  #coefs = [0.0]


  dict = {}
  for kernel, c, gama, coef, negative_weight in itertools.product(kernels, c_values, gamas, coefs, negative_weights):
    print("Kernel: " + kernel, "c=", c, "gama=", gama, "coef=", coef, "Negative weight: ", negative_weight)
    model = svm.SVC(kernel=kernel, cache_size=1000, class_weight={True: 1, False: negative_weight}, C=c, gamma=gama, coef0=coef)
    precision, recall = TrainModelAndEvaluate(train_set, model)
    if (precision, recall) != (0, 0):
      dict[(c, gama, coef)] = dict.get((c, gama, coef), [])
      dict[(c, gama, coef)].append((precision, recall))
  return dict

def SKNNLearnAndEvaluate(train_set):
  from sklearn.neural_network import MLPClassifier
  from sknn.mlp import Classifier, Layer
  #alphas = [1]
  #learning_rate_inits = [0.01, 0.1, 1, 10, 100]
  regularization_types = ['l2']
  negative_weights = np.arange(start=0.5, step=0.5, stop=4)
  dict = {}
  for layer1, layer2, negative_weight in itertools.product([5], [5], negative_weights):
    layers = [
      Layer(type='Linear', units=layer1),
      Layer(type='Rectifier', units=layer2),
      Layer(type='Softmax')
    ]
    batch_size = 100
    clf = Classifier(layers=layers, batch_size=batch_size, n_iter=1)
    precision, recall = TrainModelAndEvaluate(train_set, clf, negative_weight = negative_weight)
    if (precision, recall) != (0, 0):
      dict[(layer1, layer2)] = dict.get((layer1, layer2), [])
      dict[(layer1, layer2)].append((precision, recall))
    return dict

def NNLearnAndEvaluate(train_set):
  from sklearn.neural_network import MLPClassifier
  #alphas = [1]
  #learning_rate_inits = [0.01, 0.1, 1, 10, 100]
  regularization_types = ['l2']
  negative_weights = np.arange(start=0.2, step=0.2, stop=5)
  # for amit:
  alphas = [0.1, 0.3, 1, 3]
  # for fagfagal:
  alphas.extend([10, 30, 100, 300])

  dict = {}
  for layer1, layer2, alpha, negative_weight in itertools.product([100], [50], alphas, negative_weights):
    print("layer1={}, layer2={},negative_weights={},alpha={}".format(layer1, layer2, negative_weight, alpha))
    layers = [layer1, layer2]
    #   Layer(type='Linear', units=layer1),
    #   Layer(type='Rectifier', units=layer2),
    #   Layer(type='Softmax')
    # ]
    batch_size = 50
    clf = MLPClassifier(hidden_layer_sizes=(layer1, layer2), alpha=alpha, solver='lbfgs',
                        learning_rate_init=0.001)
    precision, recall = TrainModelAndEvaluate(train_set, clf, negative_weight = negative_weight)
    if (precision, recall) != (0, 0):
      dict[(layer1, layer2, alpha)]= dict.get((layer1, layer2,  alpha), [])
      dict[(layer1, layer2, alpha)].append((precision, recall))
  return dict

def GenerateFalses(triplets, number):
  return [random.choice(triplets).ArtificialFalse() for _ in range(number)]


def RemoveDuplicatesKeepOrder(triplets):
  s = set()
  # return triplets
  for t in triplets:
    if not (t in s):
      s.add(t)
      yield t


def partition(pred, iterable):
  from itertools import tee, filterfalse
  t1, t2 = tee(iterable)
  return list(filterfalse(pred, t1)), list(filter(pred, t2))

@common.timeit
def main():
  with open('UPI Conf. Score_model data.csv', 'r') as csvfile:
    triplets = [triplet for triplet in DebletGenerator(csvfile)]
    # triplets = []
    triplets.extend(list(map_reducer.DebletsFromSqlDump()))
    triplets = list(RemoveDuplicatesKeepOrder(triplets))
    random.shuffle(triplets)
    TEST_SET_SIZE = len(triplets) // 8
    test_set = triplets[0:TEST_SET_SIZE]
    train_set = triplets[TEST_SET_SIZE:len(triplets)]

    negatives, positives = partition(lambda x: x.result, train_set)
    print('neg/(neg+pos)=', len(negatives) / (len(positives) + len(negatives)))

    #NNLearnAndEvaluate(train_set)

    # dict = LogisticRegressionLearnAndEvaluate(train_set)
    # dict = SvmLearnAndEvaluate(train_set)
    dict = NNLearnAndEvaluate(train_set)
    print(dict)
    print('lines dump for spreadsheet:')
    for key, values in dict.items():
      print('c = ', key)
      print('precision\t', '\t'.join([str(x) for x, _ in values]))
      print('recall\t', '\t'.join([str(y) for _, y in values]))


if __name__ == '__main__':
  main()

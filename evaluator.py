from random import Random

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import cross_val_score, cross_val_predict

import common
from common import TripletGenerator
import itertools
import numpy as np

import map_reducer

random = Random(5)


def TrainModelAndEvaluate(train_set, fitter):
  def ToStringFeatures(key_fn):
    hasher = FeatureHasher(input_type='string')
    raw_X = [key_fn(x) for x in train_set]
    return hasher.transform(raw_X)

  def ToStringFeaturesTokenized(key_fn):
    hasher = FeatureHasher(input_type='string')
    raw_X = [list(k) for k in [key_fn(x) for x in train_set]]
    return hasher.transform(raw_X)

  random.shuffle(train_set)
  brand_features = ToStringFeatures(lambda x: x.Brand)
  gtin_features = ToStringFeaturesTokenized(lambda x: x.GTIN)

  X = [x.ToNumericFeatures() for x in train_set]
  from scipy.sparse import hstack
  X = hstack([X, brand_features])
  # X = hstack([X, brand_features, gtin_features])
  y = [x.result for x in train_set]
  from sklearn.preprocessing import normalize
  X = normalize(X, axis=0)
  # print(cross_val_score(fitter, X, y, cv=10))
  predictions = cross_val_predict(fitter, X, y, cv=5)
  confusion_matrix = {}
  for i in range(len(predictions)):
    if not predictions[i]:
      if not train_set[i].result:
        confusion_matrix['tn'] = confusion_matrix.get('tn', 0) + 1
      else:
        confusion_matrix['fn'] = confusion_matrix.get('fn', 0) + 1
    else:
      if not train_set[i].result:
        confusion_matrix['fp'] = confusion_matrix.get('fp', 0) + 1
      else:
        confusion_matrix['tp'] = confusion_matrix.get('tp', 0) + 1
  tn = confusion_matrix.get('tn', 0)
  fn = confusion_matrix.get('fn', 0)
  tp = confusion_matrix.get('tp', 0)
  fp = confusion_matrix.get('fp', 0)
  if tn + fn < 50:
    print('Did not tag enough negative samples, only ', tn + fn)
    return 0, 0
  print('sum True Negatives: ', tn, ' ; sum False Negatives: ', fn, ' ; sum True Positives: ', tp,
        ' ; sum False Positives: ', fp, )
  precision = tn / (tn + fn)
  print('Precision of negatives: ', precision)
  recall = tn / (tn + fp)
  print('Recall of negatives: ', recall)
  return precision, recall


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
  kernels = ['sigmoid']

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

def NNLearnAndEvaluate(train_set):
  from sklearn.neural_network import MLPClassifier
  l2 = 1
  clf = MLPClassifier(hidden_layer_sizes=(16, 8, 4, 2), alpha=l2, solver='lbfgs')
  TrainModelAndEvaluate(train_set, clf)

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
  with open('E:\\Academic\\data_science\\UPI Conf. Score_model data.csv', 'r') as csvfile:
    triplets = [triplet for triplet in TripletGenerator(csvfile)]
    # triplets = []
    triplets.extend(list(map_reducer.TripletsFromSqlDump()))
    triplets = list(RemoveDuplicatesKeepOrder(triplets))
    random.shuffle(triplets)
    TEST_SET_SIZE = len(triplets) // 8
    test_set = triplets[0:TEST_SET_SIZE]
    train_set = triplets[TEST_SET_SIZE:len(triplets)]

    negatives, positives = partition(lambda x: x.result, train_set)
    print('neg/(neg+pos)=', len(negatives) / (len(positives) + len(negatives)))

    #NNLearnAndEvaluate(train_set)

    # dict = LogisticRegressionLearnAndEvaluate(train_set)
    dict = SvmLearnAndEvaluate(train_set)
    print(dict)
    print('lines dump for spreadsheet:')
    for key, values in dict.items():
      print('c = ', key)
      print('precision\t', '\t'.join([str(x) for x, _ in values]))
      print('recall\t', '\t'.join([str(y) for _, y in values]))


if __name__ == '__main__':
  main()

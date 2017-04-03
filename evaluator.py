from random import Random

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import cross_val_score, cross_val_predict

from common import TripletGenerator
import itertools
import numpy as np

import map_reducer

random = Random(5)

def TrainModelAndEvaluate(train_set, fitter):
  def ToStringFeatures(key_fn):
    return FeatureHasher(input_type='string').transform([key_fn(x) for x in train_set])

  def ToStringFeaturesTokenized(key_fn):
    hasher = FeatureHasher(input_type='string')
    raw_X = [list(k) for k in [key_fn(x) for x in train_set]]
    return hasher.transform(raw_X)

  random.shuffle(train_set)
  brand_features = ToStringFeatures(lambda x: x.Brand)
  gtin_features = ToStringFeaturesTokenized(lambda x: x.GTIN)

  X = [x.ToNumericFeatures() for x in train_set]
  from scipy.sparse import hstack
  X = hstack([X, brand_features, gtin_features])
  y = [x.result for x in train_set]
  from sklearn.preprocessing import normalize
  X = normalize(X, axis=0)
  #print(cross_val_score(fitter, X, y, cv=10))
  predictions = cross_val_predict(fitter, X, y, cv=10)
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
    return
  print('sum True Negatives: ', tn, ' ; sum False Negatives: ', fn, ' ; sum True Positives: ', tp, ' ; sum False Positives: ', fp, )
  print('Precision of negatives: ', tn / (tn + fn))
  print('Recall of negatives: ', tn / (tn + fp))

def LogisticRegressionLearnAndEvaluate(train_set):
  from sklearn.linear_model import LogisticRegression
  c_values = [300, 100, 30, 10, 3, 1, 0.3]
  negative_weights = list(np.arange(start=0.5, stop=2.1, step=0.1))
  for c, negative_weight in itertools.product(c_values, negative_weights):
    print('c= ', c, ' negative weight=', negative_weight)
    lr = LogisticRegression(class_weight={True: 1, False: negative_weight}, C=c)
    TrainModelAndEvaluate(train_set, lr)


def SvmLearnAndEvaluate(train_set):
  from sklearn import svm
  for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    print("Kernel: " + kernel)
    TrainModelAndEvaluate(train_set, svm.SVC(kernel=kernel))

def GenerateFalses(triplets, number):
  return [random.choice(triplets).ArtificialFalse() for _ in range(number)]

def RemoveDuplicatesKeepOrder(triplets):
  s = set()
  #return triplets
  for t in triplets:
    if not (t in s):
      s.add(t)
      yield t

def partition(pred, iterable):
  from itertools import tee, filterfalse
  t1, t2 = tee(iterable)
  return list(filterfalse(pred, t1)), list(filter(pred, t2))

with open('E:\\Academic\\data_science\\UPI Conf. Score_model data.csv', 'r') as csvfile:
  triplets = [triplet for triplet in TripletGenerator(csvfile)]
  #triplets = []
  triplets.extend(list(map_reducer.TripletsFromSqlDump()))
  triplets = list(RemoveDuplicatesKeepOrder(triplets))
  random.shuffle(triplets)
  TEST_SET_SIZE = len(triplets) // 8
  test_set = triplets[0:TEST_SET_SIZE]
  train_set = triplets[TEST_SET_SIZE:len(triplets)]

  negatives, positives = partition(lambda x: x.result, train_set)
  print('neg/(neg+pos)=', len(negatives)/(len(positives)+len(negatives)))

  LogisticRegressionLearnAndEvaluate(train_set)
  # SvmLearnAndEvaluate(train_set)
  import platform;

  print(platform.architecture())

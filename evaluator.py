from random import Random

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import cross_val_score, cross_val_predict

from common import TripletGenerator
import map_reducer

random = Random(5)

def TrainModelAndEvaluate(train_set, fitter):
  random.shuffle(train_set)
  hasher = FeatureHasher(input_type='string')
  raw_X = [x.Brand for x in train_set]
  string_features = hasher.transform(raw_X)
  X = [x.ToNumericFeatures() for x in train_set]
  from scipy.sparse import hstack
  X = hstack([X, string_features])

  y = [x.result for x in train_set]
  from sklearn.preprocessing import normalize
  X = normalize(X, axis=0)
  print(cross_val_score(fitter, X, y, cv=10))
  predictions = cross_val_predict(fitter, X, y, cv=10)
  print("sum trues: ", sum(predictions), " num predictions: ", len(predictions))

def LogisticRegressionLearnAndEvaluate(train_set):
  from sklearn.linear_model import LogisticRegression
  TrainModelAndEvaluate(train_set, LogisticRegression())


def SvmLearnAndEvaluate(train_set):
  from sklearn import svm
  for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    print("Kernel: " + kernel)
    TrainModelAndEvaluate(train_set, svm.SVC(kernel=kernel))

def GenerateFalses(triplets, number):
  return [random.choice(triplets).ArtificialFalse() for _ in range(number)]

def RemoveDuplicatesKeepOrder(triplets):
  s = set()
  for t in triplets:
    if s.

def partition(pred, iterable):
  from itertools import tee, filterfalse
  t1, t2 = tee(iterable)
  return list(filterfalse(pred, t1)), list(filter(pred, t2))

with open('E:\\Academic\\data_science\\UPI Conf. Score_model data.csv', 'r') as csvfile:
  triplets = [triplet for triplet in TripletGenerator(csvfile)]
  triplets.extend(list(map_reducer.TripletsFromSqlDump()))
  RemoveDuplicatesKeepOrder(triplets)
  random.shuffle(triplets)
  TEST_SET_SIZE = len(triplets) // 8
  test_set = triplets[0:TEST_SET_SIZE]
  train_set = triplets[TEST_SET_SIZE:len(triplets)]

  negatives, positives = partition(lambda x: x.result, train_set)
  print('neg/pos=', len(negatives)/(len(positives)+len(negatives)))

  SvmLearnAndEvaluate(train_set)
  import platform;

  print(platform.architecture())

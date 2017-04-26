import csv
import functools
import numpy as np

import time

import sklearn
from collections import namedtuple as nt
from itertools import groupby
from random import Random

class Deblet(nt('Deblet', ['MPN', 'Brand', 'result'])):
  def ToNumericFeatures(self):
    mpn = self.MPN.zfill(50)
    return [ord(x) for x in mpn]

class Triplet(nt('Triplet', ['GTIN', 'MPN', 'Brand', 'result'])):
  def ArtificialFalse(self):
    gtin = self.GTIN
    mpn = self.MPN
    brand = self.Brand
    r = Random()
    #choice = r.choice(["GTIN", "MPN"])
    choice = r.choice(["MPN"])
    if choice == "GTIN":
      index = r.randint(0, len(gtin) - 1)
      gtin = gtin[:index] + str((int(gtin[index]) + r.randint(1, 9)) % 10) + gtin[index + 1:]
    else:
      while mpn != self.MPN:
        mpn = r.shuffle(mpn)

    return Triplet(GTIN=gtin, MPN=mpn, Brand=brand, result=False)

  def ToNumericFeatures(self):
    gtin = self.GTIN.zfill(14)
    result_value = [int(x) for x in gtin]
    mpn = self.MPN.zfill(50)
    result_value.extend([ord(x) for x in mpn])
    return result_value

def DebletGenerator(csvfile):
  reader = csv.reader(csvfile)
  d = groupby(reader, lambda x: x[0])
  for k, raw_values in d:
    values = list(raw_values)
    try:
      def filter_aux(s):
        values_ = list(filter(lambda x: x[1] == s, values))[0]
        return values_[2]

      mpn = filter_aux("MPN")
      brand = filter_aux("Brand")
      result = all(x[-1] == "1" for x in values)
      yield Deblet(MPN=mpn, Brand=brand, result=result)
    except:
      pass

def TripletGenerator(csvfile):
  reader = csv.reader(csvfile)
  d = groupby(reader, lambda x: x[0])
  for k, raw_values in d:
    values = list(raw_values)
    try:
      def filter_aux(s):
        values_ = list(filter(lambda x: x[1] == s, values))[0]
        return values_[2]

      gtin = filter_aux("GTIN")
      mpn = filter_aux("MPN")
      brand = filter_aux("Brand")
      result = all(x[-1] == "1" for x in values)
      yield Triplet(GTIN=gtin, MPN=mpn, Brand=brand, result=result)
    except:
      pass

class ConfusionMatrix(nt('ConfusionMatrix', ['fp', 'tp', 'fn', 'tn'])):
  def nrecall(self):
    return self.tn / (self.tn + self.fp)

  def nprecision(self):
    return self.tn / (self.tn + self.fn)

def cross_val(x, y, fitter, w, cv):
  sample_size = x.shape[0]
  block_length = sample_size // cv
  perm = np.random.permutation(sample_size)
  def get_perm(what):
    return what
  x_perm = get_perm(x)
  y_perm = get_perm(y)
  w_perm = get_perm(w)
  y_result = []
  for i in range(cv):
    def split():
      test_start = i * block_length
      test_end = (i + 1) * block_length
      what_train = what[:test_start, test_end:]
      what_test = what[test_start:test_end]
      return what_train, what_test
    _x_train, x_test, y_train, y_test, w_train, w_test = sklearn.cross_validation.
    fitter.fit(x_train, y_train, w_train)
    y_result.extend(fitter.predict(x_test))
  def calc_conf_matrix():
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for y_r, y_p in zip(y_result, y_perm):
      if not y_r and not y_p:
        tn += 1
      if not y_r and y_p:
        fn += 1
      if y_r and y_p:
        tp += 1
      if y_r and not y_p:
        fp += 1
    return ConfusionMatrix(fp=fp, tp=tp, fn=fn, tn=tn)
  return calc_conf_matrix(y_result, y_perm)


NUMBER_OF_GTIN_DIGITS = 14
NUMBER_OF_MPN_CHARACTERS = 50

def main():
  with open('UPI Conf. Score_model data.csv', 'r') as csvfile:
    x = list(DebletGenerator(csvfile))
    positives = sum([1 for e in x if e.result])
    negatives = sum([1 for e in x if not e.result])
    print("positives: ", positives)
    print("negatives: ", negatives)

if __name__ == '__main__':
  main()

def timeit(func):
  @functools.wraps(func)
  def newfunc(*args, **kwargs):
    startTime = time.time()
    func(*args, **kwargs)
    elapsedTime = time.time() - startTime
    print('---function [{}] finished in {} ms'.format(
      func.__name__, int(elapsedTime * 1000)))
  return newfunc
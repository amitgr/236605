import csv
import functools

import time
from collections import namedtuple as nt
from itertools import groupby
from random import Random

class Triplet (nt('Triplet', ['GTIN', 'MPN', 'Brand', 'result'])):
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


NUMBER_OF_GTIN_DIGITS = 14
NUMBER_OF_MPN_CHARACTERS = 50

def timeit(func):
  @functools.wraps(func)
  def newfunc(*args, **kwargs):
    startTime = time.time()
    func(*args, **kwargs)
    elapsedTime = time.time() - startTime
    print('---function [{}] finished in {} ms'.format(
      func.__name__, int(elapsedTime * 1000)))
  return newfunc
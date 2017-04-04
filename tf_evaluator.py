import itertools

import tensorflow as tf

import common
import map_reducer
import tf_utils
from common import TripletGenerator

def partition(pred, iterable):
  from itertools import tee, filterfalse
  t1, t2 = tee(iterable)
  return list(filterfalse(pred, t1)), list(filter(pred, t2))


def TfColumns():
  brand = tf.contrib.layers.sparse_column_with_hash_bucket("brand", hash_bucket_size=5000)
  digit_keys = [str(i) for i in range(10)]
  mpns = [tf.contrib.layers.sparse_column_with_hash_bucket("mpn_" + str(i), hash_bucket_size=1000) for i in
          range(common.NUMBER_OF_MPN_CHARACTERS)]
  gtins = [tf.contrib.layers.sparse_column_with_keys("gtin_" + str(i), keys=digit_keys) for i in
           range(common.NUMBER_OF_GTIN_DIGITS)]
  return [brand] + mpns + gtins


def main():
  tf.logging.set_verbosity(tf.logging.FATAL)
  with open('UPI Conf. Score_model data.csv', 'r') as csvfile:
    triplets = [triplet for triplet in TripletGenerator(csvfile)]
    TEST_SET_SIZE = len(triplets) // 8
    triplets.extend(list(map_reducer.TripletsFromSqlDump()))
    test_set = triplets[0:TEST_SET_SIZE]
    train_set = triplets[TEST_SET_SIZE:len(triplets)]
    for steps, learning_rate in itertools.product([1, 10, 50, 100, 500], [1, 0.1, 0.01, 0.001]):
      fit(learning_rate, steps, train_set, test_set)


@common.timeit
def fit(learning_rate, steps, train_set, test_set):
  from timeit import default_timer as timer
  print("steps: {}, learning_rate: {}".format(steps, learning_rate))
  m = CreateFitter(learning_rate)
  m.fit(input_fn=lambda: tf_utils.ToTensors(train_set), steps=steps)
  results = m.evaluate(input_fn=lambda: tf_utils.ToTensors(test_set), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def CreateFitter(learning_rate):
  return tf.contrib.learn.LinearClassifier(
    feature_columns=TfColumns(),
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))


if __name__ == '__main__':
  main()


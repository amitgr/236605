from tensorflow.contrib.metrics.python.ops import metric_ops
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
  brand = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket("brand", hash_bucket_size=5000), 5000)
  digit_keys = [str(i) for i in range(10)]
  mpns = [tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket("mpn_" + str(i), hash_bucket_size=1000),1000) for i in
          range(common.NUMBER_OF_MPN_CHARACTERS)]
  gtins = [tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_keys("gtin_" + str(i), keys=digit_keys), 10) for i in
           range(common.NUMBER_OF_GTIN_DIGITS)]
  return [brand] + mpns + gtins


def main():
  tf.logging.set_verbosity(tf.logging.FATAL)
  with open('UPI Conf. Score_model data.csv', 'r') as csvfile:
    triplets = [triplet for triplet in TripletGenerator(csvfile)]
    triplets.extend(list(map_reducer.TripletsFromSqlDump()))
    TEST_SET_SIZE = len(triplets) // 8
    test_set = triplets[0:TEST_SET_SIZE]
    train_set = triplets[TEST_SET_SIZE:len(triplets)]
    for steps, learning_rate in itertools.product([1, 10, 50, 100, 500], [1, 0.1, 0.01, 0.001]):
      pass
    learning_rate = 0.01
    steps = 100
    fit(learning_rate, steps, train_set, test_set)


@common.timeit
def fit(learning_rate, steps, train_set, test_set):
  from timeit import default_timer as timer
  print("steps: {}, learning_rate: {}".format(steps, learning_rate))
  m = CreateFitter(learning_rate, classifier='nn')
  m.fit(input_fn=lambda: tf_utils.ToTensors(train_set, 100.0), steps=steps)
  from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
  def CreateSpec(metric_fn):
    return MetricSpec(
      metric_fn=metric_fn,
    )
  metrics = {
    "false_positives": CreateSpec(metric_ops.streaming_false_positives),
    "false_negatives": CreateSpec(metric_ops.streaming_false_negatives),
    "true_positives": CreateSpec(metric_ops.streaming_true_positives),
    "true_negatives": CreateSpec(metric_ops.streaming_true_negatives),
  }
  results = m.evaluate(input_fn=lambda: tf_utils.ToTensors(test_set), steps=1, metrics=metrics)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def CreateFitter(learning_rate, classifier='lr'):
  if classifier == 'lr':
    return tf.contrib.learn.LinearClassifier(
        feature_columns=TfColumns(),
        weight_column_name='weight',
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  elif classifier == 'nn':
    return tf.contrib.learn.DNNRegressor(feature_columns=TfColumns(),
        hidden_units=[2])

if __name__ == '__main__':
  main()


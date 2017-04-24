import tensorflow as tf

from common import NUMBER_OF_GTIN_DIGITS, NUMBER_OF_MPN_CHARACTERS
from tensorflow.python.framework import constant_op

def ToTensors(triplets, negative_weight = 1.0):
  count = len(triplets)
  brands = [x.Brand for x in triplets]
  brand_tensors = tf.SparseTensor(indices=[[i, 0] for i in range(count)],
                  values=brands,
                  dense_shape=[count, 1])


  gtins = [x.GTIN.zfill(NUMBER_OF_GTIN_DIGITS) for x in triplets]
  gtin_tensorss = {("gtin_" + str(i)): tf.constant([str(g[i]) for g in gtins], shape=[count, 1]) for i in range(NUMBER_OF_GTIN_DIGITS)}

  mpns = [x.MPN.zfill(NUMBER_OF_MPN_CHARACTERS) for x in triplets]
  mpn_tensorss = {("mpn_" + str(i)): tf.SparseTensor(
    indices=[[i, 0] for i in range(count)],
    values = [str(x[i]) for x in mpns],
    dense_shape=[count, 1]
  ) for i in range(NUMBER_OF_MPN_CHARACTERS)}

  weights = constant_op.constant([1.0 if x.result else negative_weight for x in triplets])
  labels = tf.constant([x.result for x in triplets])
  weights = tf.reshape(weights, shape=(-1, 1))
  return {"brand": brand_tensors, 'weight': weights, **gtin_tensorss, **mpn_tensorss}, labels

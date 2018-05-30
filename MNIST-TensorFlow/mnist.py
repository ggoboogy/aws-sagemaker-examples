import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist

INPUT_TENSOR_NAME = "inputs"

def estimator_fn(run_config, params):
  feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[784])]
  return tf.estimator.DNNClassifier(feature_columns=feature_columns,
      hidden_units=[256, 32],
      n_classes=10,
      config=run_config)

def train_input_fn(training_dir, params):
  data_set = mnist.read_data_sets(training_dir, reshape=True)
  return tf.estimator.inputs.numpy_input_fn(
      x = {INPUT_TENSOR_NAME: data_set.train.images},
      y = data_set.train.labels.astype(np.int32),
      num_epochs=None,
      shuffle=True)()

def eval_input_fn(training_dir, params):
  data_set = mnist.read_data_sets(training_dir, reshape=True)
  return tf.estimator.inputs.numpy_input_fn(
      x = {INPUT_TENSOR_NAME: data_set.test.images},
      y = data_set.test.labels.astype(np.int32),
      num_epochs=1,
      shuffle=False)()

def serving_input_fn(params):
  feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[784])}
  return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()

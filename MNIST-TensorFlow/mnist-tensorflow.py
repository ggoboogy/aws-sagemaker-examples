import os, sagemaker
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets import mnist
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role

# constants
ROLE = "" # TODO Input IAM Role
BUCKET = "" # TODO Input Bucket Name
OUTPUT_PATH = "s3://{}/hands-on/mnist-tensorflow".format(BUCKET)
#ROLE = get_execution_role()
ENDPOINT_NAME = "mnist-tensorflow-jhy"

def show_image(arr):
  pixels = arr.reshape(28, 28)
  plt.imshow(pixels, cmap="gray")
  plt.show()
    
if __name__  == "__main__":
  # download MNIST dataset
  data_set = mnist.read_data_sets('data', reshape=True)
  
  # upload local datasets to S3 bucket
  sagemaker_session = sagemaker.Session()
  inputs = sagemaker_session.upload_data(path='data', bucket=BUCKET,
      key_prefix='hands-on/mnist-tensorflow/data')

  # create model with custom tensorflow code
  mnist_estimator = TensorFlow(role=ROLE,
      entry_point='mnist.py',
      output_path=OUTPUT_PATH,
      code_location=OUTPUT_PATH,
      training_steps=1000,
      evaluation_steps=100,
      train_instance_count=2,
      train_instance_type='ml.c4.xlarge')

  # train model
  mnist_estimator.fit(inputs)

  # deploy model to an endpoint
  mnist_predictor = mnist_estimator.deploy(initial_instance_count=1,
      instance_type='ml.m4.xlarge',
      endpoint_name=ENDPOINT_NAME)

  # inference
  result = mnist_predictor.predict(data_set.test.images[0])
  #show_image(data_set.test.images[0])
  print("label ==> %d" %(data_set.test.labels[0]))

  # delete endpoint & endpoint configuration
  sagemaker.Session().delete_endpoint(mnist_predictor.endpoint)
  cmd = ("aws sagemaker delete-endpoint-config --endpoint-config-name %s"
      %ENDPOINT_NAME)
  os.system(cmd)

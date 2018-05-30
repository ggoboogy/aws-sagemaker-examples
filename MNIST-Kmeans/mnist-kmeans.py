import pickle, gzip, numpy, urllib.request, json, sagemaker, os
from sagemaker import KMeans

# constants
BUCKET = "" # TODO Input Bucket Name
OUTPUT_PATH = "s3://{}/hands-on/mnist-kmeans/output".format(BUCKET)
ROLE = "" # TODO Input IAM Role
ENDPOINT_NAME = "mnist-kmeans-ec2-jhy"

def show_image(arr):
  pixels = arr.reshape(28, 28)
  plt.imshow(pixels, cmap="gray")
  plt.show()

def get_mnist_dataset():
  urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz",
      "mnist.pkl.gz")
  f = gzip.open('mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
  f.close()
  return train_set, valid_set, test_set
  
if __name__ == "__main__":
  # get MNIST dataset
  train_set, valid_set, test_set = get_mnist_dataset()

  # create model using built-in k-means algorithm
  kmeans = KMeans(role=ROLE,
                  train_instance_count=1,
                  #train_instance_type='local',
                  train_instance_type='ml.c4.4xlarge',
                  output_path=OUTPUT_PATH,
                  k=10)
  # train model
  kmeans.fit(kmeans.record_set(train_set[0]))

  # deploy model to endpoint
  kmeans_predictor = kmeans.deploy(initial_instance_count=2,
                                   instance_type='ml.m4.xlarge',
                                   endpoint_name=ENDPOINT_NAME)
  # test model
  input_set = test_set

  clustered_data = [[] for i in range(0, 10)]
  for i in range(0, len(input_set[0])):
    result = kmeans_predictor.predict(input_set[0][i].reshape(1, 784))[0]
    predicted_cluster = int(result.label['closest_cluster'].float32_tensor.values[0])
    clustered_data[predicted_cluster].append(i)

  for i in range(0, 10):
    print("Cluster " + str(i) + "\n" + "="*80)
    cnt = [0 for i in range(0, 10)]
    for data in clustered_data[i]:
        cnt[input_set[1][data]] += 1
        #show_image(input_set[0][data])
    print(*[k for k in range(0, 10)], sep="\t")
    print("-"*80)
    print(*cnt, sep="\t", end="\n\n")

  # delete endpoint & endpoint configuration
  sagemaker.Session().delete_endpoint(kmeans_predictor.endpoint)
  cmd = ("aws sagemaker delete-endpoint-config --endpoint-config-name %s"
      %ENDPOINT_NAME)
  os.system(cmd)

import pickle, gzip, numpy, urllib.request, json, sagemaker
#import matplotlib.pyplot as plt
from sagemaker import KMeans

# constants
BUCKET = "input-bucket-name"
OUTPUT_PATH = "s3://{}/hands-on/mnist-kmeans/output".format(BUCKET)
ROLE = "input-iam-role"

def show_image(arr):
  pixels = arr.reshape(28, 28)
  plt.imshow(pixels, cmap="gray")
  plt.show()

if __name__ == "__main__":
  # download and read MNIST dataset
  urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
  f = gzip.open('mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
  f.close()
  
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
  kmeans_predictor = kmeans.deploy(initial_instance_count=1,
                                   instance_type='ml.m4.xlarge')
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

  sagemaker.Session().delete_endpoint(kmeans_predictor.endpoint)

# public
Artifacts and Code Distribution for web publications, machine learning, and deep learning applications..

### Computer Vision
- [Dogs vs Cats Image Classification](keras-image-classification-dogs-cats/keras-image-classification-dogs-cats.ipynb) (Keras): Given an image of a dog or cat, determine which animal is it using Keras Framework and CNN Architecture

### NLP
- [Classify News Articles](word2vec-news-articles) (gensim/sklearn): Using gensim, create a word2vec model based on news articles. Using the same model, create a logistic regression model that inputs the word vectors and returns a category or type of news articles. 

### Audio
- [Dogs vs Cats Audio Classification](pytorch-audio-classification-catdogs) (pytorch): Using librosa to generate spectrograms to be fed into CNN model, determine if the audio file is a cat or dog sound. 
- 
### GAN
- [GAN - Dog Images](pytorch-DCGAN-Dog-Images/pytorch-DCGAN-Dog-Images.ipynb) (pytorch): Using the [Deep Convolutional Generative Adversarial Network (DCGAN) proposed by Radford, Metz and Chintala](https://arxiv.org/pdf/1511.06434.pdf), create a GAN model that creates new dog images from existing images.
  
### Classification
- [Predict Customer Churn](xgboost-bank-churn/xgboost-churn.ipynb) (XGboost): Use XGboost algorithm to predict customer churn based on customer demographic and behavior. 
- [Iris Classification](keras-classification-iris/NeuralNetwork_Iris_Classification.ipynb) (Keras): Using Toy Iris Dataset, classify which Iris flower based on the width and length of stem and petals by using an Artificial Neural Network.
- [Iris Classification](sagemaker-deploy-model-by-artifact) (XGboost): Train a model using xgboost framework, save the model artifact in s3, and load for inferencing as an endpoint 
- [Breast Cancer Detection](knn-breast-cancer/knn-breast-cancer.ipynb) (sklearn): Using breast cancer measurements, classify which patient has breast cancer using sklearn framework and AWS SageMaker.
- [Stellar Classification](sklearn-gbc-stellar/SkLearn_GBC_star.ipynb) (sklearn): Using the Gradient-Boosting Classifier algorithm from the sklearn framework, predict the stellar classification based on mreasurements. 
- [Wine Classification](sklearn-pipeline-wine/sklearn-pipeline-wine.ipynb) (sklearn): Develop a pipeline to operationalize a machine learning workload. Using toy wine dataset, predict the type of wine among 3 classes.
  
### Regression
- [Possum Age Regression](keras-regression-possum) (Keras): Using possum measurements, predict the age of a possum using Keras Deep Learning framework.
- [Iris SepalLength Regresion](linreg-iris/LinearLearner-iris.ipynb) (SageMaker): Using Iris measurements, predict the size of SepalLengthCm using SepalWidthCm, PetalLengthCm, and PetalWidthCm. Utilizes AWS built-in algorithm, LinearLearner.
- [Laptop Prices Regression](pytorch-regression-laptop-price/laptop-regression.ipynb) (pytorch): Using laptop measurements, predict the price of a laptop using pytorch framework.
- [Linear Regression](tf-regression-generic/tf-regression-generic.ipynb) (tensorflow): Using synethic data, create a regression model to showcase tensorflow/keras framework.

### Clustering
- [k-means customer segmentation](sklearn-kmeans-customers-segmentation/sklearn-kmeans-customer.ipynb) (sklearn): Perform customer segmentation for market analysis using demographic data


### Cloud
## Amazon Web Services (AWS)
- [AWS Kinesis Anomaly Detection](kinesis-anomaly-detection) (Kinesis): Create a Kinesis Data Stream to receive events from an emitter. Report anomalies based on SQL query and AWS Random Cut Forest Algorithm.
- [Breast Cancer Detection](knn-breast-cancer/knn-breast-cancer.ipynb) (SageMaker): Using breast cancer measurements, classify which patient has breast cancer using sklearn framework and AWS SageMaker.
- [Iris SepalLength Regresion](linreg-iris/LinearLearner-iris.ipynb) (SageMaker): Using Iris measurements, predict the size of SepalLengthCm using SepalWidthCm, PetalLengthCm, and PetalWidthCm. Utilizes AWS built-in algorithm, LinearLearner.
- [Deploy a model Artifact](sagemaker-deploy-model-by-artifact) (SageMaker): Train a model using xgboost framework, save the model artifact in s3, and load for inferencing as an endpoint

### MLOps
- [Classification Pipelines](sklearn-pipeline-wine/sklearn-pipeline-wine.ipynb) (sklearn): Develop a pipeline to operationalize a machine learning workload. Using toy wine dataset, predict the type of wine among 3 classes.

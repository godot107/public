# public
Artifacts and Code Distribution for web publications, machine learning, and deep learning applications.

### Computer Vision
- [Dogs vs Cats Image Classification](keras-image-classification-dogs-cats/keras-image-classification-dogs-cats.ipynb) (Keras):Given an image of a dog or cat, determine which animal is it using Keras Framework and CNN Architecture

### NLP
- [Classify News Articles](word2vec-news-articles) (gensim/sklearn): [Original Publication](https://medium.com/@manwill/applying-word2vec-to-news-articles-338ac1344099) Using gensim, create a word2vec model based on news articles. Using the same model, create a logistic regression model that inputs the word vectors and returns a category or type of news articles. 

### Audio
- [Dogs vs Cats Audio Classification](pytorch-audio-classification-catdogs) (pytorch): [Original Publication](https://medium.com/mlearning-ai/dogs-vs-cats-audio-classification-56175ce58429) Using librosa to generate spectrograms to be fed into CNN model, determine if the audio file is a cat or dog sound. 
- 
### GAN
- [GAN - Dog Images](pytorch-DCGAN-Dog-Images/pytorch-DCGAN-Dog-Images.ipynb) (pytorch): Using the [Deep Convolutional Generative Adversarial Network (DCGAN) proposed by Radford, Metz and Chintala](https://arxiv.org/pdf/1511.06434.pdf), create a GAN model that creates new dog images from existing images.
  
### Classification
- [Predict Customer Churn](xgboost-bank-churn/xgboost-churn.ipynb) (XGboost): [Original Publication](https://medium.com/@manwill/predicting-customer-churn-using-aws-sagemaker-xgboost-462a9bc0907) Use XGboost algorithm to predict customer churn based on customer demographic and behavior. 
- [Iris Classification](keras-classification-iris/NeuralNetwork_Iris_Classification.ipynb) (Keras): [Original Publication](https://medium.com/@manwill/iris-classification-using-a-keras-neural-network-39d735d11fda)  Using Toy Iris Dataset, classify which Iris flower based on the width and length of stem and petals by using an Artificial Neural Network.
- [Iris Classification](sagemaker-deploy-model-by-artifact) (XGboost): Train a model using xgboost framework, save the model artifact in s3, and load for inferencing as an endpoint 
- [Breast Cancer Detection](knn-breast-cancer/knn-breast-cancer.ipynb) (sklearn): [Original Publication](https://medium.com/@manwill/using-aws-sagemaker-k-nn-to-predict-breast-cancer-340ffbb40049)  Using breast cancer measurements, classify which patient has breast cancer using sklearn framework and AWS SageMaker.
- [Stellar Classification](sklearn-gbc-stellar/SkLearn_GBC_star.ipynb) (sklearn): [Original Publication](https://medium.com/@manwill/predicting-the-stars-with-gradientboostingclassifier-with-sklearn-76d10a1abf30)  Using the Gradient-Boosting Classifier algorithm from the sklearn framework, predict the stellar classification based on mreasurements. 
- [Wine Classification](sklearn-pipeline-wine/sklearn-pipeline-wine.ipynb) (sklearn): Develop a pipeline to operationalize a machine learning workload. Using toy wine dataset, predict the type of wine among 3 classes.
  
### Regression
- [Possum Age Regression](keras-regression-possum) (Keras): [Original Publication](https://medium.com/@manwill/predicting-a-possums-age-using-keras-artificial-neural-network-28398401f40a) Using possum measurements, predict the age of a possum using Keras Deep Learning framework.
- [Iris SepalLength Regresion](linreg-iris/LinearLearner-iris.ipynb) (SageMaker): Using Iris measurements, predict the size of SepalLengthCm using SepalWidthCm, PetalLengthCm, and PetalWidthCm. Utilizes AWS built-in algorithm, LinearLearner.
- [Laptop Prices Regression](pytorch-regression-laptop-price/laptop-regression.ipynb) (pytorch): Using laptop measurements, predict the price of a laptop using pytorch framework.
- [Linear Regression](tf-regression-generic/tf-regression-generic.ipynb) (tensorflow): Using synethic data, create a regression model to showcase tensorflow/keras framework.

### Clustering
- [k-means customer segmentation](sklearn-kmeans-customers-segmentation/sklearn-kmeans-customer.ipynb) (sklearn):[Original Publication](https://medium.com/@manwill/customer-segmentation-with-k-means-clustering-b911986b9b40)  Perform customer segmentation for market analysis using demographic data

### Amazon Web Services (AWS)
- [AWS Kinesis Anomaly Detection](kinesis-anomaly-detection) (Kinesis): [Original Publication](https://medium.com/@manwill/implementing-aws-kinesis-data-analytics-anomaly-detection-in-5-steps-8714a3727543) Create a Kinesis Data Stream to receive events from an emitter. Report anomalies based on SQL query and AWS Random Cut Forest Algorithm.
- [Breast Cancer Detection](knn-breast-cancer/knn-breast-cancer.ipynb) (SageMaker): [Original Publication](https://medium.com/@manwill/using-aws-sagemaker-k-nn-to-predict-breast-cancer-340ffbb40049) Using breast cancer measurements, classify which patient has breast cancer using sklearn framework and AWS SageMaker.
- [Iris SepalLength Regresion](linreg-iris/LinearLearner-iris.ipynb) (SageMaker): Using Iris measurements, predict the size of SepalLengthCm using SepalWidthCm, PetalLengthCm, and PetalWidthCm. Utilizes AWS built-in algorithm, LinearLearner.
- [Deploy a model Artifact](sagemaker-deploy-model-by-artifact) (SageMaker): Train a model using xgboost framework, save the model artifact in s3, and load for inferencing as an endpoint

### MLOps
- [Classification Pipelines](sklearn-pipeline-wine/sklearn-pipeline-wine.ipynb) (sklearn): [Original Publication](https://medium.com/@manwill/building-a-pipeline-with-sklearn-f8f04e7c649d)  Develop a pipeline to operationalize a machine learning workload. Using toy wine dataset, predict the type of wine among 3 classes.

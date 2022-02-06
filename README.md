"# pylightning-docker-sagemaker-imdb" \n
Sample Project to show how to Train a Deep Learning BERT Model on IMDB data using SageMaker and Pytorch Lightning Framework. \n
A Docker file is built using the Pytorch GPU image and Pytorch Lightning framework is installed in the image. \n
After that the GPU image is pushed to ECR on AWS. \n
The ECR image is then used in the Estimator function of a Sagemaker session.
This Pytorch Lightning Sagemaker code was built using this AWS example https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own/container

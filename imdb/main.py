# S3 prefix
prefix = "DEMO-pytorch-imdb"

# Define IAM role
import boto3
import re
import sagemaker as sage
from time import gmtime, strftime

import os
import numpy as np
import pandas as pd

role='arn:aws:iam::XXXXXXXX:role/service-role/AmazonSageMaker-ExecutionRole-20210413T225818'
sess = sage.Session()

from sagemaker.debugger import TensorBoardOutputConfig

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path='s3://{}/output/tensorboard".format(sess.default_bucket())',
    container_local_output_path='/opt/ml/output/tensorboard'
)

account = sess.boto_session.client("sts").get_caller_identity()["Account"]
region = sess.boto_session.region_name
image = "{}.dkr.ecr.{}.amazonaws.com/pylightning-sagemaker:latest".format(account, region)

tree = sage.estimator.Estimator(
    image,
    role,
    1,
    "ml.g4dn.2xlarge",
    output_path="s3://{}/output".format(sess.default_bucket()),
    sagemaker_session=sess,
    script_mode= True,
    metric_definitions=[
        {"Name": "train:loss", "Regex": "train_loss=([0-9\\.]+)"},
        {"Name": "eval:loss", "Regex": "val_loss=([0-9\\.]+)"},
        {"Name": "eval:acc", "Regex": "val_acc=([0-9\\.]+)"},
        {"Name": "test:loss", "Regex": "(\')?test_loss(\')?:(\s)?([0-9\\.]+)"},
        {"Name": "test:acc", "Regex": "(\')?test_acc(\')?:(\s)?([0-9\\.]+)"},
        
    ],
    hyperparameters={"epochs": 0,"gpus": 1},
    enable_sagemaker_metrics=True,
    tensorboard_output_config=tensorboard_output_config
)

tree.fit()


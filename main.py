import os
import shutil
import seaborn as sns
import datetime

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, TFRobertaModel, TFRobertaForSequenceClassification, BertTokenizer, TFBertForSequenceClassification
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K


tf.get_logger().setLevel('ERROR')

dataset = "/dbfs/mnt/syn-dwh-datascience/NG_AI/gerulata/DataSet/v3ds_BALANCED_CUTTED_2019plus.csv"
PATH="/dbfs/mnt/syn-dwh-datascience/NG_AI/gerulata/slovakbert"
tokenizer = RobertaTokenizer.from_pretrained(PATH, local_files_only=True)
model = TFRobertaModel.from_pretrained(PATH, local_files_only=True, hidden_dropout_prob=0.3)

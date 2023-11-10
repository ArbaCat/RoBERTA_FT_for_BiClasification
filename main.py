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

dataset = PATH_TO_DATASET
PATH=PATH_TO_MODEL
tokenizer = RobertaTokenizer.from_pretrained(PATH, local_files_only=True)
model = TFRobertaModel.from_pretrained(PATH, local_files_only=True, hidden_dropout_prob=0.3)

df = pd.read_csv(dataset)

le = LabelEncoder()
labels_encoded = le.fit_transform(df['xxx'])

texts_train, texts_test, labels_train, labels_test = train_test_split(df['yyyy'], labels_encoded, test_size=0.2, random_state=42)

def encode_examples(texts, labels):
  input_ids_list = []
  attention_mask_list = []
  label_list = []
  for text, label in zip(texts, labels):
    bert_input = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_attention_mask=True)
    input_ids_list.append(bert_input['input_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append(label)
  return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_list, 'attention_mask': attention_mask_list}, label_list))

train_dataset = encode_examples(texts_train, labels_train).shuffle(100).take(10000).batch(16)
test_dataset = encode_examples(texts_test, labels_test).take(1000).batch(16)

inp, out = next(iter(train_dataset)) # a batch from train_dataset
#print(inp, '\n\n', out)

class BERTForClassification(tf.keras.Model):
    
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.bert(inputs['input_ids'], attention_mask=inputs['attention_mask'])[1]
        return self.fc(x)

def matthews_correlation(y_true, y_pred):
  '''Vypočíta Matthews correlation coefficient
  y_pred_pos = K.round(K.clip(y_pred, 0, 1))
  y_pred_neg = 1 - y_pred_pos

  y_pos = K.round(K.clip(y_true, 0, 1))
  y_neg = 1 - y_pos

  tp = K.sum(y_pos * y_pred_pos)
  tn = K.sum(y_neg * y_pred_neg)

  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)

  numerator = (tp * tn - fp * fn)
  denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

  return numerator / (denominator + K.epsilon())

classifier = BERTForClassification(model, num_classes=2)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy', matthews_correlation]
)

history = classifier.fit(
    train_dataset,
    epochs=5
)

classifier.evaluate(test_dataset)

# Vizualizácia presnosti
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='trénovacia presnosť')
plt.plot(history.history['val_accuracy'], label = 'testovacia presnosť')
plt.xlabel('Epocha')
plt.ylabel('Presnosť')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Vizualizácia straty
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='trénovacia strata')
plt.plot(history.history['val_loss'], label = 'testovacia strata')
plt.xlabel('Epocha')
plt.ylabel('Strata')
plt.ylim([0, 1])
plt.legend(loc='upper right')

plt.show()

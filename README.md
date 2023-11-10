# RoBERTA_FT_for_BiClasification
This Repository is for code from RoBerta Fine tuning

# Text Classification with BERT

This project uses the BERT (Bidirectional Encoder Representations from Transformers) model for text classification. The code is written in Python and uses the TensorFlow and Transformers libraries.

## Dependencies

The following Python libraries are required:

- os
- shutil
- seaborn
- datetime
- tensorflow
- pandas
- numpy
- matplotlib
- transformers
- sklearn
- keras

## Dataset

The dataset should be a CSV file with two columns:
- 'xxx': The label column
- 'yyyy': The text column

The path to the dataset should be specified in the `PATH_TO_DATASET` variable.

## Model

The model used is the Roberta model from the Transformers library. The path to the pretrained model should be specified in the `PATH_TO_MODEL` variable.

## Training

The model is trained using the Adam optimizer, with a learning rate of 1e-5. The loss function used is SparseCategoricalCrossentropy. The metrics used are accuracy and the Matthews correlation coefficient.

The training data is split into a training set and a test set, with 80% of the data used for training and 20% used for testing.

## Evaluation

The model's performance is evaluated on the test set. The accuracy and loss for both the training and test sets are plotted for each epoch.

## Usage

To run the code, simply execute the Python script. The plots will be displayed at the end of the script.

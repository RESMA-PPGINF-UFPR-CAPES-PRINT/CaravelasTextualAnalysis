# -*- coding: utf-8 -*-
"""
EXPERIMENTO COM BERT-MULTILANG (TENSORFLOW) + REGRESSAO LOGISTICA
"""

import nltk
import numpy as np
import pandas as pd
import sklearn
import imblearn
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from os.path import exists
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from statistics import mean, stdev


preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"

"""## Métodos Auxiliares"""

def executaBERT(X_train, y_train, X_validation):

  X_train, X_validation = vetorizarBERT(X_train, X_validation)

  # treinando modelo de regressao logistica
  # segundo documentação
  # solver='liblinear' - usar para datasets pequenos e problemas binarios
  # The “balanced” mode uses the values of y to automatically adjust weights
  #    inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
  clf = LogisticRegression(multi_class='auto', class_weight='balanced', random_state=42, max_iter=200)
  clf.fit(X_train, y_train)
  y_predicted = clf.predict(X_validation)
  y_predicted_prob = clf.predict_proba(X_validation)

  return clf, y_predicted, y_predicted_prob

bert_preprocess_model = hub.KerasLayer(preprocess_url)
bert_model = hub.KerasLayer(encoder_url)

def vetorizarBERT(X_train, X_validation):

  text_preprocessed = bert_preprocess_model(X_train)
  bert_results = bert_model(text_preprocessed)
  X_train = bert_results['pooled_output']

  text_preprocessed = bert_preprocess_model(X_validation)
  bert_results = bert_model(text_preprocessed)
  X_validation = bert_results['pooled_output']

  return X_train.numpy(), X_validation.numpy()

# vetorizarBERT(X_train, X_validation)

# fonte: https://www.datageeks.com.br/processamento-de-linguagem-natural/

def get_metrics(y_test, y_predicted, binario=True):

    if binario is True:
      average = 'binary'
      pos_label = 'ACEITA'
    else:
      average = 'weighted'
      pos_label = None

    precision = precision_score(y_test, y_predicted, average=average, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_test, y_predicted, average=average, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_predicted, average=average, pos_label=pos_label, zero_division=0)
    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1

def print_metrics(y_test, y_predicted_counts, binario=True):
  accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts, binario)
  print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

historico = []
col_rotulo = 'rotulo_adaptado2'
coluna = 'texto'
FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 30
FILE_HISTORICO = 'historico/rlbert_similaridade.csv'
columns={0:'precision', 1:'precision_stdev',
         2:'recall', 3:'recall_stdev',
         4:'f1', 5:'f1_stdev',
         6:'rotulo', 7:'distancia',8:'tam_treino'}

for distancia in [96,100]:

  BASE = '../../bases/base_limpa_similaridade_exp1_'+str(distancia)+'_3.csv'
  df1 = pd.read_csv(BASE, sep=';')

  X = df1
  y = df1[col_rotulo]

  # no video dos 300 parametros, o menino fala que esse metodo tenta
  # garantir que a distribuicao entre as classes seja igual nas bases
  # de treino e teste
  scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
  kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=RANDOM_STATE)

  # loop nos folds
  for k, (train_index, val_index) in enumerate(kf.split(X, y)):

    # balanceamento dos dados
    # mudando do seed aumento as chances de todos os dados serem usados no treinamento
    X_train, y_train = X.loc[train_index], y.loc[train_index]
    # if b['balancear'] is True:
    #   rus = RandomUnderSampler(random_state=(k+10))
    #   X_train, y_train = rus.fit_resample(X_train, y_train)

    # print('fold',k, len(X_train))

    X_validation, y_validation = X.loc[val_index], y.loc[val_index]

    X_train = X_train[coluna]
    X_validation = X_validation[coluna]

    # clf, vectorizer, y_predicted, y_predicted_prob = executaTFIDF(X_train, y_train, X_validation)
    clf, y_predicted, y_predicted_prob = executaBERT(X_train, y_train, X_validation)

    accuracy, precision, recall, f1 = get_metrics(y_validation, y_predicted, True)
    scores['accuracy'].append(accuracy)
    scores['precision'].append(precision)
    scores['recall'].append(recall)
    scores['f1'].append(f1)

    # print(f1)


  historico.append([
                    mean(scores['accuracy']), stdev(scores['accuracy']),
                    mean(scores['precision']), stdev(scores['precision']),
                    mean(scores['recall']), stdev(scores['recall']),
                    mean(scores['f1']), stdev(scores['f1']),
                    distancia, len(X) ])

  """## Salvando o histórico depois de treinar o modelos"""
  dfh = pd.DataFrame.from_dict(historico)
  dfh.rename(columns=columns, inplace=True)
  dfh.to_csv(FILE_HISTORICO, index=False)

  print('média', mean(scores['precision']))

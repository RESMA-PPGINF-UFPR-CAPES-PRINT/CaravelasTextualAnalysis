# -*- coding: utf-8 -*-
"""
EXPERIMENTO COM REGRESSAO LOGISTICA + BERT-MULTILANG (TENSORFLOW)

Experimenta limpezas obrigatorias
"""
import emoji
import nltk
import numpy as np
import pandas as pd
import sklearn
import imblearn
import os
import re

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

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.get_logger().setLevel('ERROR')

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
  clf = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced', random_state=42)
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

def matriz(y_true, y_predito):
  tn, fp, fn, tp = confusion_matrix(y_true, y_predito, normalize='pred').ravel()
  return tn, fp, fn, tp

# 'true', the confusion matrix is normalized over the true conditions (100% somando a linha);
# 'pred', the confusion matrix is normalized over the predicted conditions (100% somando a coluna, daquilo que ele previu o qto ele acertou
# 'all', the confusion matrix is normalized by the total number of samples;
# None (default) mostra numeros absolutoes

# auxiliares que tratam as métricas
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


def separa_emojis(texto):
    lista = emoji.distinct_emoji_list(texto)
    for e in lista:
       texto = texto.replace(e, ' ' + e + ' ')
    return texto


def remover_underline_sequencia(s):
    e = r'\b_{1,}\b'
    return re.sub(e, " ", s)

def remover_underline(s):
    e = r'_{1,}'
    return re.sub(e, " ", s)

def remover_tudo(texto):
    return re.sub(r'\W+', " ", texto)


"""### Experimento Base Normal"""

BASE = '../../bases/base_limpa_exp_ponto.csv'
df1 = pd.read_csv(BASE, sep=';')

df1['lower'] = df1['texto'].str.lower()
df1['underline_sequencia'] = df1['texto'].apply(lambda t: remover_underline_sequencia(t))
df1['underline'] = df1['texto'].apply(lambda t: remover_underline(t))
df1['emoji_separado'] = df1['texto'].apply(lambda t: separa_emojis(t))
df1['alfanum'] = df1['texto'].apply(lambda t: remover_tudo(t))

FILE_HISTORICO = 'historico/rlbert_exp_limpeza.csv'
colunas = ['lower','underline_sequencia', 'underline','emoji_separado','alfanum']

vazias = []


col_rotulo = 'rotulo_adaptado2'
# col_rotulo = 'rotulo_original'
TEST_SIZE = 30
historico = []
FOLDS = 5
RANDOM_STATE = 42
SALVAR_MODELO = False
columns={0:'accuracy', 1:'accuracy_stdev',
         2:'precision', 3:'precision_stdev',
         4:'recall', 5:'recall_stdev',
         6:'f1', 7:'f1_stdev',
         8:'coluna', 9: 'rotulo', 10: 'tam_dev', 11: 'tam_treino'}


# loop nas colunas
for i, coluna in enumerate(colunas):

  X = df1.copy()

  # Exclui amostras vazias
  if coluna in vazias:
    X.drop(X[X[coluna].str.strip()==''].index, inplace=True)
    X.drop(X[X[coluna].str.strip().isna()].index, inplace=True)
    X.reset_index(inplace=True)

  y = X[col_rotulo]

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

    clf, y_predicted, y_predicted_prob = executaBERT(X_train, y_train, X_validation)

    accuracy, precision, recall, f1 = get_metrics(y_validation, y_predicted, True)
    scores['accuracy'].append(accuracy)
    scores['precision'].append(precision)
    scores['recall'].append(recall)
    scores['f1'].append(f1)

    # print(f1)

    # salva o modelo e vetorizador
    if SALVAR_MODELO is True:
      model_path = 'modelos/best_tfidf_'+p+'_'+ coluna + '_fold' + str(k) + '.pkl'
      pickle.dump(clf, open(model_path, 'wb'))
      vectorizer_path = 'modelos/vetorizer_'+p+'_'+ coluna + '_fold' + str(k) + '.pkl'
      pickle.dump(vectorizer, open(vectorizer_path, "wb"))

  historico.append([
                    mean(scores['accuracy']), stdev(scores['accuracy']),
                    mean(scores['precision']), stdev(scores['precision']),
                    mean(scores['recall']), stdev(scores['recall']),
                    mean(scores['f1']), stdev(scores['f1']),
                    coluna, col_rotulo, len(X), len(X_train)])

  """## Salvando o histórico depois de treinar o modelos"""
  df = pd.DataFrame.from_dict(historico)
  df.rename(columns=columns, inplace=True)
  df.to_csv(FILE_HISTORICO, index=False)

  print('média', mean(scores['precision']))

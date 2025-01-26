# -*- coding: utf-8 -*-
"""
Treina modelos com bases que cobinam pre-processamentos
"""

import nltk
import numpy as np
import os
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from statistics import mean, stdev


def executaTFIDF(X_train, y_train, X_validation, ngram_range=(1,1)):

  # lowercase=True (default)
  vectorizer = TfidfVectorizer(ngram_range=ngram_range)
  X_train = vectorizer.fit_transform(X_train)
  X_validation = vectorizer.transform(X_validation)

  # treinando modelo de regressao logistica
  # segundo documentação
  # solver='liblinear' - usar para datasets pequenos e problemas binarios
  # The “balanced” mode uses the values of y to automatically adjust weights
  #    inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
  clf = LogisticRegression(solver='lbfgs', multi_class='auto', class_weight='balanced', random_state=42)
  clf.fit(X_train, y_train)
  y_predicted = clf.predict(X_validation)
  y_predicted_prob = clf.predict_proba(X_validation)

  return clf, vectorizer, y_predicted, y_predicted_prob

  
def get_metrics(y_test, y_predicted):

    precision = precision_score(y_test, y_predicted, average='binary', pos_label='ACEITA', zero_division=0)
    recall = recall_score(y_test, y_predicted, average='binary', pos_label='ACEITA', zero_division=0)
    f1 = f1_score(y_test, y_predicted, average='binary', pos_label='ACEITA', zero_division=0)

    return precision, recall, f1

"""### Modelo"""

PATH = '../../bases/combinadas/'

dfg = pd.read_csv(PATH+'grid_stem.csv', sep=';')
#dfg = dfg[ (dfg['mencao']!='remover') & (dfg['numero']!='remover') & (dfg['url']!='tokenizar') & (dfg['emoji']!='tokenizar')]
dfg = dfg[ (dfg['mencao']!='remover') & (dfg['numero']!='remover') & (dfg['url']!='tokenizar') & (dfg['emoji']!='tokenizar') & (dfg['emoji']!='remover')]

print('QT',len(dfg))

coluna = 'normalizado'

col_rotulo = 'rotulo_adaptado2'
TEST_SIZE = 30
FOLDS = 5
RANDOM_STATE = 42

for param in dfg.itertuples():

  BASE = PATH + str(param.id) +'_' + param.nome+'.csv'

  if os.path.exists(BASE) is False:
    print('faltou', BASE)
    break

  X = pd.read_csv(BASE, sep=';')

  if len(X[X[coluna].str.strip()==''])>0 or len(X[X[coluna].str.strip().isna()])>0:
    X.drop(X[X[coluna].str.strip()==''].index, inplace=True)
    X.drop(X[X[coluna].str.strip().isna()].index, inplace=True)
    X.reset_index(inplace=True)

  y = X[col_rotulo]

  scores = {'precision': [], 'recall': [], 'f1': []}
  kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=RANDOM_STATE)

  for k, (train_index, val_index) in enumerate(kf.split(X, y)):

    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_validation, y_validation = X.loc[val_index], y.loc[val_index]
    X_train = X_train[coluna]
    X_validation = X_validation[coluna]

    clf, vectorizer, y_predicted, y_predicted_prob = executaTFIDF(X_train, y_train, X_validation)

    precision, recall, f1 = get_metrics(y_validation, y_predicted)
    scores['precision'].append(precision)
    scores['recall'].append(recall)
    scores['f1'].append(f1)

  dfg.loc[dfg['id']==param.id, 'precision'] = mean(scores['precision'])
  dfg.loc[dfg['id']==param.id, 'precision_stdev'] = stdev(scores['precision'])
  dfg.loc[dfg['id']==param.id, 'recall'] = mean(scores['recall'])
  dfg.loc[dfg['id']==param.id, 'recall_stdev'] = stdev(scores['recall'])
  dfg.loc[dfg['id']==param.id, 'f1'] = mean(scores['f1'])
  dfg.loc[dfg['id']==param.id, 'f1_stdev'] = stdev(scores['f1'])
  dfg.loc[dfg['id']==param.id, 'tam_dev'] = len(X)
  dfg.loc[dfg['id']==param.id, 'tam_treino'] = len(X_train)

  dfg.to_csv('historico/grid_historico_tfidf_stem.csv', index=False, sep=';')


# -*- coding: utf-8 -*-
"""Treino Regressao Logistica - final.ipynb

Treina com toda base dev - usa os melhores hiperparametos encontrados

## Setup
"""

import nltk
import numpy as np
import pandas as pd
import pickle
import sklearn
import timeit
inicio = timeit.default_timer()

SEED = 42
RODADA = '_R6'

from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

"""## Métodos Auxiliares"""

def executaTFIDF(X_train, y_train, X_validation, ngram_range=(1,1), min_df=1, max_df=1.0, c=1):

  # lowercase=True (default)
  vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
  X_train = vectorizer.fit_transform(X_train)
  X_validation = vectorizer.transform(X_validation)

  # treinando modelo de regressao logistica
  # segundo documentação
  # solver='liblinear' - usar para datasets pequenos e problemas binarios
  # The “balanced” mode uses the values of y to automatically adjust weights
  #    inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
  clf = LogisticRegression(solver='liblinear', multi_class='auto', C=c, class_weight='balanced', random_state=SEED, max_iter=200)
  clf.fit(X_train, y_train)
  y_predicted = clf.predict(X_validation)
  # y_predicted_prob = clf.predict_proba(X_validation)

  return clf, vectorizer, y_predicted

# auxiliares que tratam as métricas
# fonte: https://www.datageeks.com.br/processamento-de-linguagem-natural/

def get_metrics(y_test, y_predicted):

    precision = precision_score(y_test, y_predicted, average='binary', pos_label='ACEITA', zero_division=0)
    recall = recall_score(y_test, y_predicted, average='binary', pos_label='ACEITA', zero_division=0)
    f1 = f1_score(y_test, y_predicted, average='binary', pos_label='ACEITA', zero_division=0)

    return precision, recall, f1

"""## Treinamento"""
FILE_HISTORICO = 'historico/tfidf_final_'+RODADA+'.csv'
col_rotulo = 'rotulo_adaptado2'
historico = []
columns={0:'precision', 1:'recall',
         2:'f1', 3:'base',
         4:'coluna', 5:'tam_treino',
         6:'seed'}

params = [
    {
        'base' : '../../bases/base_limpa_exp_ponto.csv',
        'base_teste': '../../bases/base_teste_bruta.csv',
        'id': 'bruta_pre',
        'coluna': 'texto',
        'ngram' : (3,3),
        'min_df': 1,
        'max_df': 1.0,
        'C': 10,
    },
    {
                'base' : '../../bases/base_limpa_exp_ponto.csv',
                        'base_teste': '../../bases/base_teste_bruta.csv',
                                'id': 'bruta_equi_1g',
                                        'coluna': 'texto',
                                                'ngram' : (1,1),
                                                        'min_df': 1,
                                                        'max_df': 1.0,                                  
                                                                'C': 11,
                                                                    },
    {
                'base' : '../../bases/base_limpa_exp_ponto.csv',
                        'base_teste': '../../bases/base_teste_bruta.csv',
                                'id': 'bruta_equi_2g',
                                        'coluna': 'texto',
                                                'ngram' : (2,2),
                                                        'min_df': 1,
                                                        'max_df': 1.0,
                                                                'C': 10,
                                                                    },
    {
                        'base' : '../../bases/base_limpa_exp_ponto.csv',
                                                'base_teste': '../../bases/base_teste_bruta.csv',
                                               'id': 'bruta_equi_2g_maxdf',
                                                  'coluna': 'texto',
                          'ngram' : (2,2),
                                 'min_df': 1,
                                                 'max_df': 0.1,
                                                                                                                                                                                                                                                                                                                                                        'C': 10,
                                                                                                                                                                                                                                                                                                                                                                                                                            },
    {
                'base' : '../../bases/base_limpa_exp_ponto.csv',
                        'base_teste': '../../bases/base_teste_bruta.csv',
                                'id': 'bruta_semparam',
                                        'coluna': 'texto',
                                                'ngram' : (1,1),
                                                        'min_df': 1,
                                                        'max_df': 1.0,
                                                                'C': 1.0,
                                                                    },
    {
        'base': '../../bases/combinadas/7236_userto_urlno_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv',
        'base_teste': '../../bases/combinadas/base_teste_7236.csv',
        'id': '7236_pre',
        'coluna': 'normalizado',
        'ngram' : (2,3),
        'min_df': 0.07,
        'max_df': 1.0,
        'C': 1.0,
     },
    {
                    'base': '../../bases/combinadas/7236_userto_urlno_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv',
                            'base_teste': '../../bases/combinadas/base_teste_7236.csv',
                                    'id': '7236_semparam',
                                            'coluna': 'normalizado',
                                                    'ngram' : (1,1),
                                                            'min_df': 1,
                                                                    'max_df': 1.0,
                                                                            'C': 1.0,
                                                                                 },
    {
                    'base': '../../bases/combinadas/7236_userto_urlno_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv',
                            'base_teste': '../../bases/combinadas/base_teste_7236.csv',
                                    'id': '7236_f1',
                                            'coluna': 'normalizado',
                                                    'ngram' : (1,1),
                      'min_df': 1,
                'max_df': 1.0,
                 'C': 13,
                                                                                 },
    {
                    'base': '../../bases/combinadas/7236_userto_urlno_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv',
                            'base_teste': '../../bases/combinadas/base_teste_7236.csv',
                                    'id': '7236_f1_maxdf',
                                            'coluna': 'normalizado',
                                                    'ngram' : (1,1),
                                                            'min_df': 1,
                                                                    'max_df': 0.3,
                                                                            'C': 13,
                                                                                 },
    {
        'base': '../../bases/combinadas/8388_userto_urlre_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv',
        'base_teste': '../../bases/combinadas/base_teste_8388.csv',
        'id': '8388',
        'coluna': 'normalizado',
        'ngram' : (2,3),
        'min_df': 0.07,
        'max_df': 1.0,
        'C': 1.0,
     },
    {
        'base': '../../bases/combinadas/6511_userno_urlre_numto_emojno_unino_stopno_enelno_ascno_defno_lowtru.csv',
        'base_teste': '../../bases/combinadas/base_teste_6511.csv',
        'id': '6511_pre',
        'coluna': 'normalizado',
        'ngram' : (3,3),
        'min_df': 1,
        'max_df': 1.0,
        'C': 10,
     },
    {
                    'base': '../../bases/combinadas/6511_userno_urlre_numto_emojno_unino_stopno_enelno_ascno_defno_lowtru.csv',
                            'base_teste': '../../bases/combinadas/base_teste_6511.csv',
                                    'id': '6511_semparam',
                                            'coluna': 'normalizado',
                                                    'ngram' : (1,1),
                                                            'min_df': 1,
                                                                    'max_df': 1.0,
                                                                            'C': 1.0,
                                                                                 },
    {
                    'base': '../../bases/combinadas/6511_userno_urlre_numto_emojno_unino_stopno_enelno_ascno_defno_lowtru.csv',
                            'base_teste': '../../bases/combinadas/base_teste_6511.csv',
                                    'id': '6511_equi',
                                            'coluna': 'normalizado',
                                                    'ngram' : (1,1),
                                                            'min_df': 1,
                                                                    'max_df': 1.0,
                                                                            'C': 10,
                                                                                 },
    {
                    'base': '../../bases/combinadas/6511_userno_urlre_numto_emojno_unino_stopno_enelno_ascno_defno_lowtru.csv',
                            'base_teste': '../../bases/combinadas/base_teste_6511.csv',
                                    'id': '6511_equimaxdf',
                                            'coluna': 'normalizado',
                                                    'ngram' : (1,1),
                                                            'min_df': 1,
                                                                    'max_df': 0.3,
                                                                            'C': 10,
                                                                                 },
]

for param in params:

  if param['id']!="6511_equi":  
      continue
  coluna = param['coluna']
  df_train = pd.read_csv(param['base'], sep=';')
  df_train[coluna] = df_train[coluna].str.lower()
  df_test = pd.read_csv(param['base_teste'], sep=';')
  df_test[coluna] = df_test[coluna].str.lower()

  print(len(df_train), len(df_test))

  # Exclui amostras vazias
  if len(df_train[df_train[coluna].str.strip()==''])>0 or len(df_train[df_train[coluna].str.strip().isna()])>0:
    df_train.drop(df_train[df_train[coluna].str.strip()==''].index, inplace=True)
    df_train.drop(df_train[df_train[coluna].str.strip().isna()].index, inplace=True)
    df_train.reset_index(inplace=True)

  #print(len(df_train), len(df_test))

  X_train = df_train[coluna]
  y_train = df_train[col_rotulo]
  X_test = df_test[coluna]
  y_test = df_test[col_rotulo]

  clf, vectorizer, y_predicted = executaTFIDF(X_train, y_train, X_test, param['ngram'], param['min_df'], param['max_df'], param['C'])

  precision, recall, f1 = get_metrics(y_test, y_predicted)

  historico.append([precision, recall, f1, param['id'], coluna, len(X_train), SEED])

  #model_path = 'modelos/tfidf_model_'+param['id']+'_.pkl'
  #pickle.dump(clf, open(model_path, 'wb'))
  #vectorizer_path = 'modelos/tfidf_vetorizer_'+param['id']+'_.pkl'
  #pickle.dump(vectorizer, open(vectorizer_path, "wb"))
 

"""## Salvando o histórico"""

df = pd.DataFrame.from_dict(historico)
df.rename(columns=columns, inplace=True)

# salva pra não precisar executar novamente
df.to_csv(FILE_HISTORICO, index=False)

fim = timeit.default_timer()
duracao = fim - inicio
tempo = pd.DataFrame.from_dict({'inicio':[inicio],'fim':[fim],'duracao_seg':[duracao]})
FILE_TEMPO = 'tempo/tfidf_final.csv'
tempo.to_csv(FILE_TEMPO, index=False)

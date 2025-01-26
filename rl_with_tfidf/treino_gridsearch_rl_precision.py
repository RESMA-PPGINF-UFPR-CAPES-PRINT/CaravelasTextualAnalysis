# -*- coding: utf-8 -*-
"""Caravelas - 2 - Treino TF-IDF + RL - GridSearch
Faz grid search para achar melhores hiperparametros na base bruta
"""

"""# Setup"""

import nltk
import pandas as pd
import re
import sklearn
import timeit
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

"""# Treino"""

BASE = '../../bases/base_limpa_exp_ponto.csv'
#BASE = '../../bases/combinadas/7236_userto_urlno_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv'
#BASE = '../../bases/combinadas/6511_userno_urlre_numto_emojno_unino_stopno_enelno_ascno_defno_lowtru.csv'
#BASE = '../../bases/combinadas/8388_userto_urlre_numto_emojtr_unino_stopno_enelno_ascye_defle_lowtru.csv'
col_rotulo = 'rotulo_adaptado2'
col_texto = 'texto'
#col_texto = 'normalizado'
FOLDS = 5
TEST_SIZE = 30
RANDOM_STATE = 31
#RANDOM_STATE = 42
#FILE_HISTORICO = 'grid_tfidf_base_combinada8388_23g.csv'
FILE_HISTORICO = 'grid_tfidf_base_bruta_f1_ngram.csv'

df = pd.read_csv(BASE, sep=';')
df['rotulo'] = df[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)
#X_train = df[col_texto].str.lower()
X_train = df[col_texto]
y_train = df['rotulo']

meus_scores = {'recall'   : make_scorer(recall_score, average='binary', pos_label = 1),
               'precision': make_scorer(precision_score, average='binary', pos_label = 1),
               'f1'       : make_scorer(f1_score, average='binary', pos_label = 1)}

#f1 = make_scorer(f1_score, average='binary', pos_label = 1)

# Para usar StratifiedShuffleSplit ao inves do StratifiedKFolds
kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=RANDOM_STATE)


# ngram todos
params = {
    'vectorizer__ngram_range': [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)],
}

# bruta params 3,3
#params = {
#    'vectorizer__ngram_range': [(3,3)],
    #'vectorizer__max_df': [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05], # nao muda nada - nem com C
    #'vectorizer__min_df': [1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], # piora muito -mesmo com C
    #'vectorizer__max_features': [None, 70000, 60000, 50000, 40000, 30000, 20000, 10000], # piora - mesmo com C
    #'model__C': [200, 150, 100, 80, 50, 30, 20, 15, 10, 5, 1], # top 3: 10 5 15
    #'model__C': [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5], # top C 10
    #'model__solver': ['lbfgs','liblinear'], # lbfgs melhor na 3 casa decimal
#}

# 6511 3,3g
#params = {
    #'vectorizer__ngram_range': [(3,3)],
    #'vectorizer__min_df': [1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], # piora muito
    #'vectorizer__max_df': [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05], nao muda nada
    #'vectorizer__max_features': [None, 70000, 60000, 50000, 40000, 30000, 20000, 10000], # piora
    #'model__C': [200, 150, 100, 80, 50, 30, 20, 15, 10, 5, 1], # top 3: 10 5 15
#    'model__C': [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
#}        


# bruta params 2,3g
#params = {
    #'vectorizer__ngram_range': [(2,3)],
    #'vectorizer__max_df': [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05], # top 3 0.10 1.0 0.05
    #'vectorizer__max_df': [1.0, 0.12, 0.11, 0.10, 0.09, 0.08, 0.05], #top 0.10
    #'vectorizer__max_df': [1.0, 0.10],
    #'vectorizer__min_df': [1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], piora
    #'vectorizer__max_features': [None, 150000, 140000, 130000, 120000, 110000, 100000, 90000, 80000, 70000, 60000, 50000, 40000, 30000, 20000, 15000, 10000], # 100k
    #'vectorizer__max_features': [None, 101000, 100500, 100000, 99500, 99000, 98000], # 98k 99k
    #'vectorizer__max_features': [None, 99000, 98000, 97000, 96000, 95000, 94000, 93000, 92000, 91000, 90000], # 98k 99k none na 3 casa decimal
    #'model__C': [200, 150, 100, 80, 50, 30, 20, 15, 10, 5, 1], # piora | com max_df 0.10 top 10 5 20
    #'model__C': [12, 11, 10, 9, 8, 7, 6, 5], #top com max_df 0.10 C 9,10,11,12
    #'model__C': [10, 1],
    #'model__solver': ['lbfgs','liblinear'], # diferena na 4 casa decimal
#}

# 7236 2,3g
#params = {
#    'vectorizer__ngram_range': [(2,3)],
    #'vectorizer__min_df': [1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], # TOP 0.07
#    'vectorizer__min_df':[1,0.07],
    #'vectorizer__max_df': [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05], # MELHORA NA 3 CASA DECIMAL 0.05
    #'vectorizer__max_features': [None, 150000, 140000, 130000, 120000, 110000, 100000, 90000, 80000, 70000, 60000, 50000, 40000, 30000, 20000, 15000, 10000, 5000], piora
#    'model__C': [200, 150, 100, 80, 50, 30, 20, 15, 10, 5, 1], # top 3 100,150,50
    #'model__C': [150,140,130,120,110,103,102,101,100,99,98,97,90,80,70,60,50] #top 90
#    'model__C': [1,90], #nao supera o min_df 
#}

#8388 2,3g
#params = {
#    'vectorizer__ngram_range': [(2,3)],
    #'vectorizer__min_df': [1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], # TOP 0.07
#    'vectorizer__min_df': [1,0.07],
    #'vectorizer__max_df': [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05],  # MELHORA NA 3 CASA DECIMAL 0.05
   # 'vectorizer__max_features': [None, 150000, 140000, 130000, 120000, 110000, 100000, 90000, 80000, 70000, 60000, 50000, 40000, 30000, 20000, 15000, 10000, 5000], nada
#   'model__C': [200, 150, 100, 80, 50, 30, 20, 15, 10, 5, 1], # top 200,150,100 nao supera min_df

#}    


pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('model', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=200))
])
#gs = GridSearchCV(pipe, param_grid=params, cv=kf, scoring=f1, verbose=1, n_jobs=-1)
gs = GridSearchCV(pipe, param_grid=params, cv=kf, scoring=meus_scores, refit='precision', verbose=1, n_jobs=-1)
inicio = timeit.default_timer()
gs.fit(X_train, y_train)
fim = timeit.default_timer()
print(gs.best_score_)
print(gs.best_params_)

dfr = pd.DataFrame(gs.cv_results_)
dfr.to_csv('historico/'+FILE_HISTORICO, index=False)
#print(dfr[['mean_test_precision','param_vectorizer__min_df']].sort_values(by=['mean_test_precision'], ascending=False).head(25))
#print(dfr[['mean_test_precision','param_vectorizer__max_df']].sort_values(by=['mean_test_precision'], ascending=False).head(25))
#print(dfr[['mean_test_precision','param_vectorizer__max_features']].sort_values(by=['mean_test_precision'], ascending=False).head(25))
#print(dfr[['mean_test_precision','param_vectorizer__ngram_range']].sort_values(by=['mean_test_precision'], ascending=False).head(10))
#print(dfr[['mean_test_precision','param_model__C']].sort_values(by=['mean_test_precision'], ascending=False).head(20))
#print(dfr[['mean_test_precision','param_model__C','param_vectorizer__max_df']].sort_values(by=['mean_test_precision'], ascending=False).head(20))
print(dfr[['mean_test_precision','param_model__C','param_vectorizer__min_df']].sort_values(by=['mean_test_precision'], ascending=False).head(25))
#print(dfr[['mean_test_precision','param_model__C','param_vectorizer__max_features']].sort_values(by=['mean_test_precision'], ascending=False).head(25))
#print(dfr[['mean_test_precision','param_model__C','param_model__solver']].sort_values(by=['mean_test_precision'], ascending=False).head(25))

str_inicio = time.strftime("%d/%m/%Y %H:%M:%S:", time.gmtime(inicio))
#print('Inicio:', str_inicio)
str_fim = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(fim))
#print('Fim:', str_fim)
duracao = fim - inicio
#print('Duracao do treino: %f seg' % (duracao))

dft = pd.DataFrame.from_dict({'inicio':[inicio],'fim':[fim],'duracao':[duracao]})
dft.head()

dft.to_csv('historico/tempo_'+FILE_HISTORICO, index=False)

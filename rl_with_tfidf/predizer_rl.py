# -*- coding: utf-8 -*-
"""
PREDIZ USANDO TF-IDF + RL

# Setup
"""

import os
import pandas as pd
import numpy as np
import sklearn
import timeit
import pickle
from sklearn.metrics import confusion_matrix


"""# Avalia o Modelo
"""
coluna = 'texto'
col_rotulo = 'rotulo_adaptado2'

ID = 'bruta'
BASE_TESTE = '../../bases/base_teste_'+ID+'.csv'

ID = '6511'
BASE_TESTE = '../../bases/combinadas/base_teste_'+ID+'.csv'
coluna = 'normalizado'

#model_path = '/home/hfrocha/envRLBERT/treino/modelos/tfidf_model_bruta_.pkl'
#vectorizer_path = '/home/hfrocha/envRLBERT/treino/modelos/tfidf_vetorizer_bruta_.pkl'
model_path = '/home/hfrocha/envRLBERT/treino/modelos/tfidf_model_6511_equi_.pkl'
vectorizer_path = '/home/hfrocha/envRLBERT/treino/modelos/tfidf_vetorizer_6511_equi_.pkl'

model = pickle.load(open(model_path,'rb'))
vectorizer = pickle.load(open(vectorizer_path,'rb'))

df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test['rotulo'] = df_test['rotulo_adaptado2'].apply(lambda r: 1 if r=='ACEITA' else 0)


"""# Predizendo

### Predizendo um Lote
"""
X_test = df_test[coluna]
X_test = vectorizer.transform(X_test)
y_predicted = model.predict(X_test)
y_predicted_prob = model.predict_proba(X_test)

print(y_predicted.shape)
print(y_predicted_prob.shape)

y_proba = []
for p in y_predicted_prob:
    y_proba.append(p[0])

df_test['rotulo_predito'] = y_predicted
df_test['predito'] = df_test['rotulo_predito'].apply(lambda r: 1 if r=='ACEITA' else 0)
df_test['proba'] = y_proba

#FILE_HISTORICO = 'historico/tfidf_base_teste_bruta_prediction.csv'
FILE_HISTORICO = 'historico/tfidf_base_teste_norm_6511_prediction.csv'
df_test.to_csv(FILE_HISTORICO, index=False)

"""### Matriz de Confusão"""
print('\n')

import matplotlib.pyplot as plt

y_true = df_test['rotulo']
y_predicted = df_test['predito']
cm = confusion_matrix(y_true, y_predicted, normalize='pred')
print(cm)
print('\n')

cm = confusion_matrix(y_true, y_predicted)
print(cm)
print('\n')

#sample = df_test.sample(1)
#texto = sample['texto'].item()
#true_label = sample['rotulo'].item()

#predictions = model(tf.constant([texto]))
#score = float(predictions[0])

#print('True Label', true_label)
#print('Label:', (0 if score <=0.5 else 1))
#print('Score: ', round(score, 2))
#print('Prob classe positiva', round((100 * score), 2))
#print('Prob classe negativa', round((100 * (1-score)), 2))



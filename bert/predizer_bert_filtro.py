# -*- coding: utf-8 -*-
"""
AVALIA UM MODELO BERT
APLICA FILTRO AOS DADOS DE TESTE PARA SIMULAR PRODUCAO
# Setup
"""

import os
import pandas as pd
import numpy as np
import sklearn
import timeit

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.metrics import confusion_matrix, precision_score, recall_score


"""# Avalia o Modelo
"""
col_rotulo = 'rotulo_adaptado2'
#col_rotulo = 'rotulo_adaptado1'

ID = 'bruta'
#ID = '5169'
IDX = '_471_R5'
#IDX = '_471_R201'
METRIC = 'precision'
#METRIC = 'f1'
SENTENCE_LENGTH = 344
BATCH_SIZE = 16
BASE_TESTE = '../../bases/base_teste_'+ID+'.csv'
coluna = 'normalizado'
if ID=='bruta':
    #BASE_TESTE = '../../bases/base_teste_'+ID+'.csv'
    BASE_TESTE = '../../bases/base_teste_bruta_filename_motivo.csv'
    coluna = 'texto'

FILE_HISTORICO = 'historico/bert_prediction_'+METRIC+'_'+ID+IDX+'_simulada.csv'
#bestloss5169_213_R1
#bestprecisionbruta_471_R4
path_to_model = '/home/hfrocha/envRLBERT/treino/modelos/best'+METRIC+ID+IDX
print("MODELO", path_to_model)
model = tf.keras.models.load_model(path_to_model, compile=False)

df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test['rotulo'] = df_test[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)
# simula dados de producao
df_test = df_test[df_test['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]
print(len(df_test))

preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
def make_bert_preprocessing_model(sentence_features, seq_length):
    input_segments = [
                    tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
                            for ft in sentence_features
                                ]
    bert_preprocess = hub.load(preprocess_url)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]

    packer = hub.KerasLayer(
                    bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                                    name="packer",
                                        )
    model_inputs = packer(segments)
    return keras.Model(input_segments, model_inputs)

bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]
def preprocess_text(text, seq_length):
    text = tf.convert_to_tensor([text])
    output = bert_preprocess_model([text], seq_length)
    output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
    return output

def preprocess_text_and_image(sample, seq_length):
      text = preprocess_text(sample[coluna], seq_length)
      return text

def dataframe_to_dataset(dataframe):
        columns = [coluna, "rotulo"]
        dataframe = dataframe[columns].copy()
        labels = dataframe.pop("rotulo")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        return ds

auto = tf.data.AUTOTUNE
def prepare_dataset(dataframe, bsize, seq_length):
      ds = dataframe_to_dataset(dataframe)
      ds = ds.map(lambda x, y: (preprocess_text_and_image(x, seq_length), y)).cache()
      return ds.batch(bsize).prefetch(auto)

bert_preprocess_model = make_bert_preprocessing_model(["text_1"], SENTENCE_LENGTH)  

teste_ds = prepare_dataset(df_test, BATCH_SIZE, SENTENCE_LENGTH)
#for x in teste_ds:
#    print(x[0].keys())
#    break


#textos = df_test[coluna]
#predictions = model.predict(textos, verbose=0)
predictions = model.predict(teste_ds, verbose=0)
y_predito = [] # usado na matriz de confusão
for i, prediction in enumerate(predictions):
    score = float(prediction[0])
    pred_label = (0 if score <=0.5 else 1)
    y_predito.append(pred_label)

df_test['proba'] = predictions
df_test['predito'] = y_predito
df_test.to_csv(FILE_HISTORICO, index=False)

import matplotlib.pyplot as plt

y_true = df_test['rotulo']
cm = confusion_matrix(y_true, y_predito, normalize='pred')
print('PRED',cm)
cm = confusion_matrix(y_true, y_predito, normalize='true')
print('TRUE',cm)
cm = confusion_matrix(y_true, y_predito)
print(cm)
precision = precision_score(y_true, y_predito, pos_label=1, zero_division=0)
rec = recall_score(y_true, y_predito, pos_label=1, zero_division=0)
f1 = 2 * (precision*rec / (precision+rec))
print('PRE',precision,'REC', rec, 'f1',f1)

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



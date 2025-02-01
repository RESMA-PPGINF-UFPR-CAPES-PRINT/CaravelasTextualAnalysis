# -*- coding: utf-8 -*-
"""
Cria dataframe com combinações de preprocessamentos
Cria arquivos com tratamentos combinados aplicados ao dados
"""

"""# Setup"""
import emoji
import nltk
import numpy as np
import os
import pandas as pd
import re
import spacy
import unicodedata

nltk.download('punkt') # pacote para tokenizar
nltk.download('rslp') # pacote para stemming
nltk.download('stopwords')


from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

"""# Definição das BASES"""



"""### Menções"""

# trata menções de usuários ex: @intenzetattooink
# esta pengando @scienceoverload.

def tratar_mencoes(texto, opcao=None):
  e = r'\@[a-zA-Z0-9_\.]+'
  if opcao=='remover':
    return re.sub(e, "", texto)
  elif opcao=='tokenizar':
    return re.sub(e, "USUARIO", texto)

  return texto

def remover_mencoes(texto):
  return tratar_mencoes(texto, 'remover')

def tokenizar_mencoes(texto):
  return tratar_mencoes(texto, 'tokenizar')


def tratar_numeros(texto, opcao=None):
  e = r'[0-9]+'
  if opcao=='remover':
    return re.sub(e, "", texto)
  elif opcao=='tokenizar':
    return re.sub(e, "NUMERO", texto)
  return texto

def remover_numeros(texto):
  return tratar_numeros(texto, 'remover')

def tokenizar_numeros(texto):
  return tratar_numeros(texto, 'tokenizar')


"""### URLs"""


def tratar_urls(texto, opcao='remover'):
  e1 = r'http(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\'\/\\\+&amp;%\$#_!]*)?'
  # noticias.terra.com.br
  e2 = r'([0-9a-zA-Z]*)?\.?([0-9a-zA-Z]*)\.com(.br)?'
  if opcao=='remover':
    texto = re.sub(e1, "", texto)
    texto = re.sub(e2, "", texto)
  elif opcao=='tokenizar':
    texto = re.sub(e1, "URL", texto)
    texto = re.sub(e2, "URL", texto)

  return texto

def remover_urls(texto):
  return tratar_urls(texto, 'remover')

def tokenizar_urls(texto):
  return tratar_urls(texto, 'tokenizar')


"""### Stopwords"""

# Retira stopwords usando ER
stopwords = nltk.corpus.stopwords.words('portuguese')

def remove_stopwords(texto):
  pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
  return pattern.sub('', texto)

# metodo para retirar stopwords


"""### Lematização"""

# metodo para lematizar texto
# Troca lexemas (palavras flexionadas) por seus lemas
# Experimentei o CoGrOO e ele parece um pouco melhor do que Spacy
try:
  lemmetizer = spacy.load('pt_core_news_sm')
except:
  print('Não foi possível carregar pt_core_news_sm')
  pass

def lematizar(texto):

  doc = lemmetizer(texto)

  re = ""
  for token in doc:
    if token.pos_ == 'VERB' or token.pos_ == 'NOUN':
      re = re + ' ' + token.lemma_
    else:
      re = re + ' ' + token.text

  return re

"""### Stemização"""

# metodo  para stemizar texto
stemizer = nltk.stem.RSLPStemmer()

def stematizar(texto):

  tokens = nltk.word_tokenize(texto, language='portuguese')

  t = ""
  for token in tokens:
      t = t + ' ' + stemizer.stem(token)

  return t

def deflexionar(texto, opcao=None):

  if opcao=='stemizar':
    return stematizar(texto)
  elif opcao=='lematizar':
    return lematizar(texto)

  return texto


"""### Emojis"""


# Troca o emoji por seu equivalente em texto
# O delimitador é com espaço mesmo

def emoji2text(texto):
  texto = emoji.demojize(texto, delimiters=(' ', ' '), language='pt')
  texto = re.sub(r'_', " ", texto)
  return texto



def tratar_emoji(texto, opcao=None):
  if opcao=='remover':
    return emoji.replace_emoji(texto)
  elif opcao=='tokenizar':
    # o token é com espaço mesmo
    return emoji.replace_emoji(texto, replace=' EMOJI ')
  elif opcao=='traduzir':
    return emoji2text(texto)

  return texto


"""### Normalização com Enelvo"""


import enelvo
from enelvo.normaliser import Normaliser
ig_list='enelvo_ignore_list.txt'
fc_list='enelvo_force_list.txt'
norm = Normaliser(tokenizer='readable', ig_list=ig_list, fc_list=fc_list)
  # norm = Normaliser(tokenizer='readable')


"""### Instagram Fonts"""

# Verifica se o texto NÃO contem uso de Instagram Fonts
# Se verdadeiro o texto NÃO contem
# É bem dificil de determinar isso, pois no final é tudo unicode
# Busca coisas que parecem palavras, mas que não estão escritos usando latim basic, gerando tokens diferentes com o mesma semantica
def is_normalized(t):

  t = re.sub(r'[0-9!,\.\?\/\\#…ºª˙‼⁉´ℹ️₂ㅤ]', " ", t)
  tokens = nltk.word_tokenize(t, language='portuguese')
  normalizado = True

  for token in tokens:
    normC = unicodedata.is_normalized('NFKC', token)
    normD = unicodedata.is_normalized('NFKD', token)
    if normC is False and normD is False:
      normalizado = False
      return normalizado
    elif normC is True and normD is True:
      for l in token:
        n = unicodedata.name(l, 'NOTFOUND')
        if 'NEGATIVE' in n:
          normalizado = False
          return normalizado
        elif 'SMALL CAPITAL' in n:
          normalizado = False
          return normalizado
        elif 'LONG STROKE' in n:
          normalizado = False
          return normalizado

  return normalizado


def tem_instagram_fonts(t):
  norm = is_normalized(t)
  if norm is True:
    return False
  else:
    return True


"""# Combinações

param = {
    'lower': [None,True],
    'mencao': [None,'tokenizar','remover'],
    'url': [None,'tokenizar','remover'],
    'numero': [None,'tokenizar','remover'],
    'emoji': [None,'tokenizar','remover','traduzir'],
    'unicodedata': [None,True],
    'stopwords': [None,True],
    'enelvo': [None,True],
    'ascii': [None,True],
    'deflex': [None,'stemizar','lematizar'],
}

# Nos exp individuais ficou claro que ttos hashtag-relac pioram muito o desempenho
grid = []
id = 0
for element1 in param['lower']:
  for element2 in param['mencao']:
    for element3 in param['url']:
      for element4 in param['numero']:
        for element5 in param['emoji']:
          for element6 in param['unicodedata']:
            for element7 in param['stopwords']:
              for element8 in param['enelvo']:
                for element9 in param['ascii']:
                  for element10 in param['deflex']:
                    l = [element1,element2,element3,element4,element5,element6,element7,element8,element9,element10]
                    c=0
                    for e in l:
                      if e==None:
                        c+=1
                    # 10 None é o raw text
                    # 9 None são os ttos individuais
                    if c>=9:
                      continue
                    nome = 'user'+str(element2)[0:2]+'_url'+str(element3)[0:2]+\
                    '_num'+str(element4)[0:2]+'_emoj'+str(element5)[0:2]+\
                    '_uni'+str(element6)[0:3]+'_stop'+str(element7)[0:3]+\
                    '_enel'+str(element8)[0:3]+'_asc'+str(element9)[0:3]+\
                    '_def'+str(element10)[0:2]+'_low'+str(element1)[0:3]
                    nome = nome.lower().replace('non_','no_').replace('tru_','ye_')
                    # print(nome)
                    grid.append([nome, id, element2, element3, element4, element5,element6,element7,element8,element9,element10, element1])
                    id+=1


dfg = pd.DataFrame.from_dict(grid)
columns = {0:'nome', 1:'id', 2:'mencao', 3:'url', 4:'numero',5:'emoji',
           6:'unicodedata',7:'stopwords',8:'enelvo',9:'ascii',10:'deflex',
           11:'lower'}
dfg.rename(columns=columns, inplace=True)

# Exclui normalizações que não fazerm sentido juntas
dfg.drop(dfg[ (dfg['unicodedata']==True) & (dfg['ascii']==True)].index, inplace=True)
dfg.drop(dfg[ (dfg['lower']==True) & (dfg['deflex']=='stemizar')].index, inplace=True)
dfg.drop(dfg[ (dfg['lower']==True) & (dfg['enelvo']==True)].index, inplace=True)

dfg['precision'] = 0
dfg['precision_stdev'] = np.nan
dfg['recall'] = 0
dfg['recall_stdev'] = np.nan
dfg['f1'] = 0
dfg['f1_stdev'] = np.nan
dfg['tam_dev'] = 0
dfg['tam_treino'] = 0

FILE = '../../bases/combinadas/grid.csv'
dfg.to_csv(FILE, index=False, sep=';')
"""

dfg = pd.read_csv('../../bases/combinadas/grid_stem.csv', sep=';')

"""### Loop
"""
BASE = '../../bases/base_limpa_exp_ponto.csv'
dfl = pd.read_csv(BASE, sep=';')

# tfidf
dfg = dfg[ (dfg['mencao']!='remover') & (dfg['numero']!='remover') & (dfg['url']!='tokenizar') & (dfg['emoji']!='tokenizar') & (dfg['emoji']!='remover')]
#bert
#dfg = dfg[ (dfg['mencao']!='tokenizar') & (dfg['numero']!='tokenizar') & (dfg['url']!='remover') & (dfg['emoji']!='traduzir') & (dfg['emoji']!='remover')]

print('QT',len(dfg))

for params in dfg.itertuples():

  FILE = '../../bases/combinadas/'+str(params.id)+'_'+params.nome+'.csv'

  if os.path.exists(FILE) is True:
    continue

  dftemp = dfl.copy()

  dftemp["normalizado"] = dftemp['texto']
  if params.unicodedata is True:
    dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: unicodedata.normalize('NFKC', t))

  dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: tratar_mencoes(t, params.mencao))
  dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: tratar_urls(t, params.url))
  dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: tratar_numeros(t, params.numero))
  dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: tratar_emoji(t, params.emoji))

  if params.enelvo is True:
    dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: norm.normalise(t))

  if params.stopwords is True:
    dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: remove_stopwords(t))

  dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: deflexionar(t, params.deflex))
  if params.ascii is True:
    dftemp["normalizado"] = dftemp['normalizado'].apply(lambda t: unidecode(t))

  if params.lower is True:
    dftemp["normalizado"] = dftemp['normalizado'].str.lower()


  dftemp.to_csv(FILE, index=False, sep=';')

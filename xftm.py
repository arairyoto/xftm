import os
import sys
import logging
#WordNet
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import spearmanr

#dealing with japanese
mpl.rcParams['font.family'] = 'AppleGothic'

import numpy as np
import codecs
import csv

import re

import util

import math

import requests
import json

# ログ
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s')
logger = logging.getLogger(__name__)

class WSLObject:
    def __init__(self, name, attribute, lang = None):
        self.name = name
        self.attribute = attribute
        self.lang = lang
        if attribute == 'word':
            self.id = name+':'+lang
        elif attribute == 'synset':
            self.id = name
        elif attribute == 'lemma':
            self.id = name.split(':')[0]+':'+lang+':'+name.split(':')[1]



    def __eq__(self, other):
        return self.name == other.name and self.attribute == other.attribute and self.lang == other.lang


class XtendedFastTextMultilingual:
    def __init__(self, folder):
        self.langs = ['eng','jpn','fra']
        self.folder = folder
        logger.info('Start Loading XFTM...')
        self.wv = util.Shared()
        self.wv.loadTxtModel(self.folder+'/'+'words.txt')
        logger.info('    Words DONE')
        self.sv = util.Shared()
        self.sv.loadTxtModel(self.folder+'/naive/'+'synsets.txt')
        logger.info('    Synsets DONE')
        self.lv = util.Shared()
        self.lv.loadTxtModel(self.folder+'/naive/'+'lemmas.txt')
        logger.info('    Lemmas DONE')
        # self.dv = util.Shared()
        # self.dv.loadTextModel(self.folder+'/naive/'+'domains.txt')

        self.words = self.wv.model.keys()
        self.synsets = self.sv.model.keys()
        self.lemmas = self.lv.model.keys()
        logger.info('     FINISH!!')

    def relatedness(self, e1, e2):
        try:
            if e1 in self.wv.model.keys():
                v1 = np.array(self.wv.model[e1])
            elif e1 in self.sv.model.keys():
                v1 = np.array(self.sv.model[e1])
            elif e1 in self.lv.model.keys():
                v1 = np.array(self.lv.model[e1])

            if e2 in self.wv.model.keys():
                v2 = np.array(self.wv.model[e2])
            elif e2 in self.sv.model.keys():
                v2 = np.array(self.sv.model[e2])
            elif e2 in self.lv.model.keys():
                v2 = np.array(self.lv.model[e2])

            return sum(v1*v2)/np.sqrt(sum(v1*v1)*sum(v2*v2))
        except:
            return -1

    def eq(self, e, e_o):
        if e == None:
            return True
        else:
            return e == e_o

    def search_lemmas(self, word=None, lang=None, synset=None):
        result = []
        for lemma in self.lemmas:
            word_o = lemma.split(':')[0]
            lang_o = lemma.split(':')[1]
            synset_o = lemma.split(':')[2]
            if self.eq(word, word_o) and self.eq(lang, lang_o) and self.eq(synset, synset_o):
                result.append(lemma)
            else:
                continue

        return result

    def search_words(self, word=None, lang=None):
        result = []
        for word in self.words:
            word_o = word.split(':')[0]
            lang_o = word.split(':')[1]
            if self.eq(word, word_o) and self.eq(lang, lang_o):
                result.append(word)
            else:
                continue

        return result

    def lemma_vectors(self, word, lang, synset):
        result = {}
        lemmas = self.search_lemmas(word, lang, synset)
        for lemma in lemmas:
            result[lemma] = np.array(self.lv.model[lemma])

        return result


    def most_similar(self, target, target_attributes=['word', 'lemma', 'synset','domain'], target_pos=['n','v','s','a','r'],target_langs = None):
        result = {}
        if target_langs is None:
            langs = self.langs
        else:
            langs = set(target_langs)&set(self.langs)

        if 'synset' in target_attributes:
            for e in self.sv.model.keys():
                pos = e.split(".")[1]
                # Relatedness
                temp = self.relatedness(e,target)
                if not math.isnan(temp):
                    if pos in target_pos:
                        result[e] = temp

        # if 'domain' in target_attributes:
        #     for e in self.get[('domain', None)].model.keys():
        #         o = WSLObject(e, 'domain')
        #         #Relatedness
        #         #temp = self.relatedness(o,target)
        #         #連想指数？？
        #         temp = self.relatedness(o,target)*math.log(self.shortest_path(o,target)/2+1)
        #         if not math.isnan(temp):
        #             result[e] = temp

        if 'lemma' in target_attributes:
            for lang in langs:
                for e in self.search_lemmas(lang=lang):
                    pos = e.split(".")[1]
                    #Relatedness
                    #temp = self.relatedness(o,target)
                    #連想指数？？
                    temp = self.relatedness(e,target)
                    if not math.isnan(temp):
                        if pos in target_pos:
                            result[e] = temp

        if 'word' in target_attributes:
            for lang in langs:
                for e in self.search_words(lang=lang):
                    #Relatedness
                    #temp = self.relatedness(o,target)
                    #連想指数？？
                    temp = self.relatedness(e,target)

                    synsets = wn.synsets(e.split(':')[0], lang=lang)
                    pos_set = set([synset.pos() for synset in synsets])
                    if not math.isnan(temp):
                        if len(set(target_pos) & pos_set) != 0:
                            result[e] = temp
        #ソート
        result = sorted(result.items() , key=lambda x: x[1], reverse=True)
        return result

    def notify2slack(self, text):
        requests.post('https://hooks.slack.com/services/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', data = json.dumps({
            'text': text, # 投稿するテキスト
            'username': u'ResearchBot', # 投稿のユーザー名
            'icon_emoji': u':ghost:', # 投稿のプロフィール画像に入れる絵文字
            'link_names': 1, # メンションを有効にする
        }))

        # access_token = 'xoxp-238888349110-238057234212-248114544272-b9bd0f08cdf71997b77e2d8ceb15c372'
        # CHANNEL_ID = ''
        # with open("FILE_PATH",'rb') as f:
        #     param = {'token':access_token, 'channels':CHANNEL_ID,'title':'タイトル'}
        #     r = requests.post("https://slack.com/api/files.upload", params=param,files={'file':f})


if __name__=='__main__':
    folder = '/Users/arai9814/Desktop/fastText'
    target = '設計:jpn'

    xftm = XtendedFastTextMultilingual(folder)
    result = xftm.most_similar(target)

    f = open("設計.txt","w")
    for v in result:
        f.write(v[0]+","+str(v[1])+"\n")
    f.close()
    xftm.notify2slack('fastText relatedness DONE!!')

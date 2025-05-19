import os, sys, json
import numpy as np

from tqdm.auto import tqdm, trange

from bicm import BipartiteGraph as BiG

import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer



class metode:
    
    
    def __init__(self, texts, row_names=None, alpha=0.01, lang=None):
        # biadjacency list
        self.texts=texts
        # row names
        if row_names is not None:
            self.row_names=row_names
        else:
            self.row_names=np.arange(len(row_names))
        # significance threshold
        self.alpha=alpha
        assert alpha<1 and alpha>0
        # language
        if lang is None:
            self.lang="english"
        else:
            # check that english is among the accepted languages by nltk
            self.lang=lang
        # get the stemmer
        self.stemmer = SnowballStemmer(self.lang, ignore_stopwords=True)
        
        
        # get the biadjacency list
        self.get_bili()
        # get the biadjacency matrix to feed bicm
        self.get_all_tokens()
        self.bili2bima()
    
    def _tests(self, entry):
        stop_words = list(stopwords.words(self.lang))
        bad_char=['©', '–', '‘', '’', '“', '”', "''", "'s",'``']
        super_bad_char=["'", "\\", "/", '+', "^^", ":", '£', '$']
        # I am removing:
        # - stop words;
        # - punctuation
        # - fractional numbers
        _test_0=not (entry in stop_words)
        _test_1=not (entry in bad_char)
        _test_2=not (entry in string.punctuation)
        _test_3=not ('.' in entry)
        _test_4=not (',' in entry)
        _test_5=not entry[0].isnumeric()
        _test_6=not (entry[0] in super_bad_char)
        return _test_0 and _test_1 and _test_2 and _test_3 and _test_4 and _test_5 and _test_6
    
    def get_bili(self):
        self.bili={}
        for i in trange(len(self.texts), leave=True, desc='get biadjacency list'):
            self.bili[self.row_names[i]]=self.text2tokens(self.texts[i])
    
    def text2tokens(self, text):
        word_tokens = [wt.lower() for wt in word_tokenize(text)]
        out=[]
        for w in word_tokens:
            if self._tests(w):
                out.append(self.stemmer.stem(w))
        # calculate the multiplicity of entries in out
        out=np.array(out)
        aux=np.unique(out, return_counts=True)
        return dict(zip(aux[0], aux[1]))
    
    def get_all_tokens(self):
        self.all_tokens=[]
        for key in tqdm(self.bili.keys(), leave=True, desc='get all tokens'):
            for _token in self.bili[key].keys():
                self.all_tokens.append(_token)
        self.all_tokens=np.array(self.all_tokens)

    def bili2bima(self):
        ''' 
        biadjacency list to biadjacency matrix
        '''
        self.bima=np.zeros((len(self.bili.keys()), len(self.all_tokens)), dtype=int)
        for key in tqdm(self.bili.keys(), leave=True):
            for _token in self.bili[key].keys():
                where_token=np.where(self.all_tokens==_token)[0][0]
                self.bima[key, where_token]+=self.bili[key][token]
                
    def metode(self):
        '''
        Maximum Entropy TOpic DEtection
        '''
        # Bipartite Graph of bicm
        self.mygraph=BiG()
        # initialize it with the bipartite matrix, 
        # i.e. the only way to initialize the method for bipartite weighted graphs
        self.mygraph.set_biadjacency_matrix(self.bima)
        # solve BiWCM
        self.mygraph.solve_tool()
        # calculate p-values
        self.mygraph.compute_weighted_pvals_mat()
        self.pval_m=cacca.get_validated_matrix(significance=self.alpha, validation_method='fdr')
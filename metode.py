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

from collections import Counter


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
        
        # bad characters
        self.stop_words = list(stopwords.words(self.lang))
        self.bad_char=['©', '–', '‘', '’', '“', '”', "''", "'s",'``']
        self.super_bad_char=["'", "\\", "/", '+', "^^", ":", '£', '$']
        
        # get the biadjacency list
        #self.get_bili()
        # get the biadjacency matrix to feed bicm
        #self.get_all_tokens()
        #self.bili2bima()
    
    def _tests(self, entry):
        # I am removing:
        # - stop words;
        # - punctuation
        # - fractional numbers
        _test_0=not (entry in self.stop_words)
        _test_1=not (entry in self.bad_char)
        _test_2=not (entry in string.punctuation)
        _test_3=not ('.' in entry)
        _test_4=not (',' in entry)
        #_test_5=not entry.isnumeric()
        _test_5=not entry[0].isnumeric()
        _test_6=not (entry[0] in self.super_bad_char)
        return _test_0 and _test_1 and _test_2 and _test_3 and _test_4 and _test_5# and _test_6
    
    def get_bili(self):
        self.bili={}
        for i in trange(len(self.texts), leave=True, desc='get biadjacency list'):
            self.bili[self.row_names[i]]=self.text2tokens_counter(self.texts[i])
            
            
    def text2tokens_counter(self, text):
        tokens = [self.stemmer.stem(w.lower()) for w in word_tokenize(text) if self._tests(w)]
        return Counter(tokens)
    
    #def text2tokens_defaultdict(self, text):
    #    counts = defaultdict(int)
    #    for w in word_tokenize(text):
    #        if self._tests(w):
    #            counts[self.stemmer.stem(w.lower())] += 1
    #    return counts
    #
    #def text2tokens(self, text):
    #    
    #    word_tokens = [wt.lower() for wt in word_tokenize(text)]
    #    out=[]
    #    out=[self.stemmer.stem(w.lower()) for w in word_tokenize(text) if not (w in self.stop_words) and not (w in self.bad_char) and not (w in string.punctuation) and not ('.' in w) and not (',' in w) and not w.isnumeric()]
    #    aux=np.unique(out, return_counts=True)
    #    return dict(zip(aux[0], aux[1]))
    
    def get_all_tokens(self):
        self.all_tokens=[]
        for key in tqdm(self.bili.keys(), leave=True, desc='get all tokens'):
            for _token in self.bili[key].keys():
                if _token not in self.all_tokens:
                    self.all_tokens.append(_token)
        self.all_tokens.sort()
        self.all_tokens=np.array(self.all_tokens)

    def bili2bima(self):
        ''' 
        biadjacency list to biadjacency matrix
        '''
        if not hasattr(self, 'bili'):
            self.get_bili()
        if not hasattr(self, 'all_tokens'):
            self.get_all_tokens()
        
        self.bima=np.zeros((len(self.bili.keys()), len(self.all_tokens)), dtype=int)
        for i_key, key in enumerate(tqdm(self.bili.keys(), leave=True)):
            for _token in self.all_tokens:
                where_token=np.where(self.all_tokens==_token)[0][0]
                #where_token=self.all_tokens.index(_token)
                self.bima[i_key, where_token]+=self.bili[key][_token]
                
    def validated_bima(self):
        '''
        Maximum Entropy TOpic DEtection
        '''
        if not hasattr(self, 'bima'):
            self.bili2bima()
            
        
        # Bipartite Graph of bicm
        self.mygraph=BiG()
        # initialize it with the bipartite matrix, 
        # i.e. the only way to initialize the method for bipartite weighted graphs
        self.mygraph.set_biadjacency_matrix(self.bima)
        # solve BiWCM
        self.mygraph.solve_tool()
        # calculate p-values
        self.mygraph.compute_weighted_pvals_mat()
        # return the validated matrix
        self.validated_m=self.mygraph.get_validated_matrix(significance=self.alpha, validation_method='fdr')
        
    def sample_me(self):
        if not hasattr(self, 'mygraph'):
            self.validated_bima()
        pass
        
            
    def validated_covol(self):
        pass
        
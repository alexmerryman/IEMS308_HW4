#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:58:53 2018

@author: alexmerryman
"""

# IEMS 308
# HW 4

import glob
import os
import pandas as pd
import numpy as np
import pysolr
import re
import math
import io

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, ne_chunk_sents, sent_tokenize
from nltk import conlltags2tree, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
#nltk.download('all')


solr = pysolr.Solr('http://localhost:8983/solr/hw4_corpus2')


def deleteSolrDocs():
    solr.delete(q='*:*')


#solr start
#solr create -c hw4_corpus2
#post -c hw4_corpus2 corpus/*.txt
def indexCorpus():
    doc_id = 0
    print doc_id
    solr_add_list = []

    corpus_dir = os.path.join(os.getcwd(), "corpus")

    for f in glob.glob(os.path.join(corpus_dir, "*.txt")):

        doc_dict = {
        "id": "doc_" + str(doc_id),
         "title": f
         }
        solr_add_list.append(doc_dict)
        doc_id += 1

    solr.add(solr_add_list)


#deleteSolrDocs()
#indexCorpus()


"""
Take question.
Parse question.
Search corpus.
Rank documents.
Provide answer.
"""


print ' '
def takeQuestion():
    question = raw_input("Please type your question:\n")
    return question

question = takeQuestion()
# Which companies went bankrupt in 2011
# Which companies went bankrupt in 1998
#question = "Which companies went bankrupt in July of 2002?"

question_tokens = word_tokenize(question)
question_tagged = pos_tag(question_tokens)
question_entities = ne_chunk(question_tagged)

wn_lem = WordNetLemmatizer()


"""
Question Parse/Analysis
"""

question_tag_dict = {"who": ["NNP", "NNS", "PERSON", "ORGANIZATION"],
                     "what": ["NN", "NNS", "NNP", "PERSON", "ORGANIZATION"],
                     "which": ["NN", "NNS", "NNP", "ORGANIZATION"],
                     "where": ["NN", "NNS", "NNP"],
                     "how much": "CD",
                     "how many": "CD",
                     "percent": "CD",
                     "when": ["CD", "DATE"]
                     }


question_words = ["who", "what", "where", "when", "why", "which", "how"]

def questionParse(question):
    q_word_index = 0
    tok_index = 0
    for tok in question_tokens:
        if wn_lem.lemmatize(tok) in question_words:
            q_word_index = tok_index

        tok_index += 1

    # reference question word bigram for answer tag
    if wn_lem.lemmatize(question_tokens[q_word_index]).lower() in ["which", "what"]:
        if wn_lem.lemmatize(question_tokens[q_word_index+1]) == "company":
            return ["B-ORGANIZATION", "I-ORGANIZATION"]

        elif wn_lem.lemmatize(question_tokens[q_word_index+1]) in ["person", "people"]:
            return ["B-PERSON", "I-PERSON"]

        else:
            return ["B-NP", "I-NP"]

    elif wn_lem.lemmatize(question_tokens[q_word_index]).lower() in ["who"]:
        return ["B-PERSON", "I-PERSON"]

    else:
        return ["unknown"]



print 'search for answer of type:'
question_tag = questionParse(question)
print question_tag


"""
Question Keywords
----------------------
Extract keywords from the question phrase:
- proper nouns
- noun and adjectival modifiers

Plus hypernyms (WordNet)
"""

question_keywords = []

keyword_pos_tags = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "CD"]

noun_pos = ["NN", "NNS", "NNP", "NNPS", "CD"]
adj_pos = ["JJ", "JJR", "JJS"]

'''
strong keywords (actual words in the question) vs. weak keywords (hypernyms) ???
'''

def questionKeywords(question_tagged, include_hypernyms=0):
    for t in question_tagged:
        if t[1] in keyword_pos_tags and wn_lem.lemmatize(t[0]).lower() not in question_words:
            word = t[0]

            # HYPERNYMS
            if include_hypernyms == 1:
                if t[1] in noun_pos:
                    word_synset = wn.synsets(word, pos=wn.NOUN)

                elif t[1] in adj_pos:
                    word_synset = wn.synsets(word, pos=wn.ADJ)
                else:
                    pass

                if len(word_synset) > 0:
                    word_synset = word_synset[0]
                    hypernyms = word_synset.hypernyms()

                    for h in hypernyms:
                        kword = h.lemma_names()[0]
                        question_keywords.append(str(kword))


            word = wn_lem.lemmatize(t[0])
            question_keywords.append(word)

    return question_keywords


question_keywords = questionKeywords(question_tagged)
print 'search terms:', question_keywords
print ' '



def searchCorpus(search_term, limit_docs=10):
    results = solr.search(search_term, rows=limit_docs)
    total_freq = results.raw_response["response"]["numFound"]

    result_docs = []
    for r in results:
        doc_name = r['resourcename'][0][-14:]
        result_docs.append(doc_name)

    return result_docs, total_freq


def hashMap(kw_list):
    keyword_dict = {}
    keyword_true_freq_dict = {}
    for i in kw_list:
        search = searchCorpus(i)
        keyword_dict[i] = search[0]
        keyword_true_freq_dict[i] = search[1]

    return keyword_dict, keyword_true_freq_dict


def Nfiles(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


def idf(word):
    return math.log(float(N - keyword_true_freq_dict[word] + 0.5)/float(keyword_true_freq_dict[word] + 0.5))


def tf(word, doc):
    doc_tokens = word_tokenize(doc)
    word_freq = 0
    for t in doc_tokens:
        if word == wn_lem.lemmatize(t):
            word_freq += 1

    return word_freq


def qtf(word):
    qtf = 0
    for k in question_keywords:
        if k == word:
            qtf += 1

    return qtf


def doc_len(fpath):
    f = io.open(fpath, 'r', encoding='utf8', errors='ignore')
    s = f.read()
    doc_len = int(len(word_tokenize(s)))

    return doc_len


#print "Calculating length of docs in corpus..."
#doc_len_dict = {}
#corpus_dir = os.path.join(os.getcwd(), "corpus")
#for fname in glob.glob(os.path.join(corpus_dir, "*.txt")):
#    docname = fname[-14:]
#    doc_len_dict[docname] = doc_len(fname)
#
#
#print "Calculating average doc length..."
#avg_dl_list = []
#for d, dl in doc_len_dict.iteritems():
#    avg_dl_list.append(dl)
#
#avg_dl = np.mean(avg_dl_list)
#print avg_dl

avg_dl = 22941.382191780824


def OkapiFormula(fname, word, doc, k1=1.6, k3=0.5, b=0.75):
    idf_o = idf(word)
    tf_o = tf(word, doc)
    qtf_o = qtf(word)
#    doc_len = doc_len_dict[fname]
    doc_len_o = doc_len(fname)

    okapi_tf = ((k1 + 1.0)*tf_o)/((k1*(1.0 - b)+(b*(doc_len_o/avg_dl))) + tf_o)

    weight = idf_o * okapi_tf * (((k3 + 1.0)*qtf_o)/(k3 + qtf_o))

    return weight


corpus_dir = os.path.join(os.getcwd(), "corpus")
N = Nfiles(corpus_dir)

keyword_dict, keyword_true_freq_dict = hashMap(question_keywords)


okapi_dict = {}

for w, retrieved_docs in keyword_dict.iteritems():
    print 'word:\t', w
    print 'freq:\t', keyword_true_freq_dict[w]
    print 'idf:\t', idf(w)
    print 'qtf:\t', qtf(w)

    okapi_dict[w] = []

    print "Document Okapi scores:"
    for fname in retrieved_docs:
        fpath = os.path.join(os.getcwd(), "corpus")
        fpath = os.path.join(fpath, fname)
        f = io.open(fpath, 'r', encoding='utf8', errors='ignore')
        s = f.read()
        okapi = OkapiFormula(fpath, w, s)
        print fname, '    ', okapi
        okapi_dict[w].append((fname, okapi))
    print '----------------------'


okapi_dfs = {}
for kword, scores in okapi_dict.iteritems():
    okapi_dfs[kword] = pd.DataFrame(scores, columns=["file", "okapi_score"])



def retrieveTopDocs(keyword, n_docs=5):
    top_docs = okapi_dfs[keyword].nlargest(n_docs, "okapi_score")["file"].tolist()
    return top_docs


def tf_sent(word, tokenized_sent):
    word_freq = 0
    for t in tokenized_sent:
        if word == wn_lem.lemmatize(t[0]):
            word_freq += 1

    return word_freq


def process_score_doc(kword, doc):
    f = io.open(doc, 'r', errors='ignore')
    s = f.read()

    candidate_sents = []

    orig_sentences = sent_tokenize(s)
    sentences = [word_tokenize(sent) for sent in orig_sentences]
    sentences = [pos_tag(sent) for sent in sentences]

    sent_index = 0
    for sent in sentences:
        NER_sent = ne_chunk(sent)
        iob_tags = tree2conlltags(NER_sent)
        for i in iob_tags:
            if i[2] in question_tag:
                # calculate combined TF-IDF
                combined_tf_idf = 0
                for k in question_keywords:
                    tf_s = tf_sent(k, sent)
                    idf_s = math.log(float(len(sentences) - tf_s + 0.5)/float(tf_s + 0.5))
                    tf_idf_s = tf_s*idf_s
                    combined_tf_idf += tf_idf_s
                candidate_sents.append((orig_sentences[sent_index], combined_tf_idf))

            else:
                pass
        sent_index += 1

    return list(set(candidate_sents))


print "Question-Answer Tag:", question_tag
print "Question Keywords:", question_keywords

overall_answer = "Sorry, I don't know the answer to that."
overall_answer_tfidf = 0
for k in question_keywords:
#    print "------------------------"
#    print "Question Keyword:", k
    top_docs = retrieveTopDocs(k)
#    print "Top Docs:"
#    print top_docs
    fpath = os.path.join(os.getcwd(), "corpus")
    doc_scores_list = []
#    print ' '
    for t in top_docs:
        docpath = os.path.join(fpath, t)
        print "Scoring sentences in document {}".format(t)
        doc_sents_scores = process_score_doc(k, docpath)
        doc_sents_scores_df = pd.DataFrame(doc_sents_scores, columns=["sentence", "tf-idf"])
        doc_scores_list.append(doc_sents_scores_df)

    doc_df = pd.concat(doc_scores_list)
    top_sents = doc_df.nlargest(3, "tf-idf")
    print " "
    print "Top Sentences:"
    print top_sents

    if top_sents.iloc[0]["tf-idf"] > overall_answer_tfidf:
        overall_answer = top_sents.iloc[0]["sentence"]

print " "
print "You asked:"
print question
print " "
print "I think I found the answer:"
print overall_answer





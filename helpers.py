#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import math
import time

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Countprint("BLEU Score: ",bleu.sentence_bleu(reference_translation, candidate_translation_1)) SOS and EOS

    def addSentence(self, sentence, lang_type):
        if lang_type =="ls_source":
            main_words = sourceNormalization(sentence)
            for word in main_words:
                self.addWord(word)
        else:
            for word in sentence.split():
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    #s = s.lower().strip()
    #s = re.split(r'[(|,)]', s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def sourceNormalization(ls_source):
    word_list = []
    words = re.sub(r'[\(|,]', ' ',ls_source)
    words = re.sub('\)', '', words)
    words = re.split(' ', words)
    for word in words:
        word_list.append(word)
    return word_list

def indexesFromSentence(lang, sentence, lang_type):
    sentence = sentence.strip()
    if lang_type=="ls_source":
        indexes = []
        words =  sourceNormalization(sentence)
        for word in words:
           indexes.append(lang.word2index[word])
        return indexes
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, lang_type, device):
    indexes = indexesFromSentence(lang, sentence, lang_type)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], "ls_source", device)
    target_tensor = tensorFromSentence(output_lang, pair[1], "ls_target", device)
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
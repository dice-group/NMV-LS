#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:17:48 2021

@author: asep
"""

import re
import math
import psutil
import time
from io import open
import random
from random import shuffle
import argparse
import matplotlib.pyplot as plt
import os.path as path

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.cuda
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score 
from nltk.translate.chrf_score import sentence_chrf
from tqdm import tqdm
import nltk
nltk.download('wordnet')

use_cuda = torch.cuda.is_available()
print(use_cuda)
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

"""start of sentence tag"""
SOS_token = 0

"""end of sentence tag"""
EOS_token = 1

"""unknown word tag (this is used to handle words that are not in our Vocabulary)"""
UNK_token = 2

def loadDataset(lang1, lang2, dir_data):
    train_pairs = []
    test_pairs = []
    dev_pairs = []
    
    train = readData(dir_data, "train.txt")
    test = readData(dir_data, "test.txt")
    dev = readData(dir_data, "dev.txt")
    
    for data in train:
        pair = []
        pair.append(normalizeString(data[0]))
        pair.append(normalizeString(data[1]))
        train_pairs.append(pair)
    
    for data in test:
        pair = []
        pair.append(normalizeString(data[0]))
        pair.append(normalizeString(data[1]))
        test_pairs.append(pair)
        
    for data in dev:
        pair = []
        pair.append(normalizeString(data[0]))
        pair.append(normalizeString(data[1]))
        dev_pairs.append(pair)
            
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)       
    return input_lang, output_lang, train_pairs, test_pairs, dev_pairs 

def readData(dir_data, filename):
    pairs=[]
    with open(path.join(dir_data, filename), 'r') as f:
        for line in f:
            pair = []
            values = line.split("\t")
            source = values[0]
            target = values[1]
            target = re.sub("\n", "", target)
            pair.append(source)
            pair.append(target)
            pairs.append(pair)
    return pairs

def normalizeString(s):
    s = s.lower().strip()
    return s   

def sourceNormalization(ls_source):
    word_list = []
    words = re.sub(r'[\(|,]', ' ',ls_source)
    words = re.sub('\)', '', words)
    words = re.split(' ', words)
    for word in words:
        word_list.append(word)
    return word_list

class Lang:
    def __init__(self, language):
        self.language_name = language
        self.word_to_index = {"SOS":SOS_token, "EOS":EOS_token, "<UNK>":UNK_token}
        self.word_to_count = {}
        self.index_to_word = {SOS_token: "SOS", EOS_token: "EOS", UNK_token: "<UNK>"}
        self.vocab_size = 3
        self.cutoff_point = -1


    def countSentence(self, sentence, lang_type):
        if lang_type =="ls_source":
            main_words = sourceNormalization(sentence)
            for word in main_words:
                self.countWords(word)
        else:    
            for word in sentence.split(' '):
                self.countWords(word)

    """counts the number of times each word appears in the dataset"""
    def countWords(self, word):
        if word not in self.word_to_count:
            self.word_to_count[word] = 1
        else:
            self.word_to_count[word] += 1

    """if the number of unique words in the dataset is larger than the
    specified max_vocab_size, creates a cutoff point that is used to
    leave infrequent words out of the vocabulary"""
    def createCutoff(self, max_vocab_size):
        word_freqs = list(self.word_to_count.values())
        word_freqs.sort(reverse=True)
        if len(word_freqs) > max_vocab_size:
            self.cutoff_point = word_freqs[max_vocab_size]

    """assigns each unique word in a sentence a unique index"""
    def addSentence(self, sentence, lang_type):
        new_sentence = ''
        if lang_type =="ls_source":
            main_words = sourceNormalization(sentence)
            for word in main_words:
                unk_word = self.addWord(word)
                if not new_sentence:
                    new_sentence =unk_word
                else:
                    new_sentence = new_sentence + ' ' + unk_word
        else:
            for word in sentence.split(' '):
                unk_word = self.addWord(word)
                if not new_sentence:
                    new_sentence =unk_word
                else:
                    new_sentence = new_sentence + ' ' + unk_word
        return new_sentence

    """assigns a word a unique index if not already in vocabulary
    and it appeaars often enough in the dataset
    (self.word_to_count is larger than self.cutoff_point)"""
    def addWord(self, word):
        if self.word_to_count[word] > self.cutoff_point:
            if word not in self.word_to_index:
                self.word_to_index[word] = self.vocab_size
                self.index_to_word[self.vocab_size] = word
                self.vocab_size += 1
            return word
        else:
            return self.index_to_word[2]

"""completely prepares both input and output languages 
and returns cleaned and trimmed train and test pairs"""
def filterPairs(pairs, max_length):
    filtered_pairs=[]
    #print(len(pairs))
    for pair in pairs:
        source = sourceNormalization(pair[0])
        target = pair[1].split(' ')
        if len(source) < max_length and len(target)<max_length:
            filtered_pairs.append(pair)
    return filtered_pairs

def prepareData(lang1, lang2, data_dir, max_vocab_size=50000, 
                reverse=False, trim=0, start_filter=False, perc_train_set=0.9, 
                print_to=None):
    
    input_lang, output_lang, train_pairs, test_pairs, dev_pairs = loadDataset(lang1, lang2, data_dir)
    pairs = train_pairs+test_pairs+dev_pairs
    print("Read %s sentence pairs" % len(pairs))
    
    if print_to:
        with open(print_to,'a') as f:
            f.write("Read %s sentence pairs \n" % len(pairs))
    print('trim', trim)
    if trim != 0:
        pairs = filterPairs(pairs, trim)
        print("Trimmed to %s sentence pairs" % len(pairs))
        if print_to:
            with open(print_to,'a') as f:
                f.write("Read %s sentence pairs \n" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.countSentence(pair[0], 'ls_source')
        output_lang.countSentence(pair[1], 'ls_target')


    input_lang.createCutoff(max_vocab_size)
    output_lang.createCutoff(max_vocab_size)

    pairs = [(input_lang.addSentence(pair[0], "ls_source"),output_lang.addSentence(pair[1], "ls_target")) 
             for pair in pairs]

    #shuffle(pairs)
    
    train_pairs = train_pairs
    test_pairs = test_pairs 
    val_pairs = dev_pairs 

    print("Train pairs: %s" % (len(train_pairs)))
    print("Test pairs: %s" % (len(test_pairs)))
    print("Validation pairs: %s" % (len(dev_pairs)))
    print("Counted Words -> Trimmed Vocabulary Sizes (w/ EOS and SOS tags):")
    print("%s, %s -> %s" % (input_lang.language_name, len(input_lang.word_to_count),
                            input_lang.vocab_size,))
    print("%s, %s -> %s" % (output_lang.language_name, len(output_lang.word_to_count), 
                            output_lang.vocab_size))
    print()

    if print_to:
        with open(print_to,'a') as f:
            f.write("Train pairs: %s" % (len(train_pairs)))
            f.write("Test pairs: %s" % (len(test_pairs)))
            f.write("Validation pairs: %s" % (len(val_pairs)))
            f.write("Counted Words -> Trimmed Vocabulary Sizes (w/ EOS and SOS tags):")
            f.write("%s, %s -> %s" % (input_lang.language_name, 
                                      len(input_lang.word_to_count),
                                      input_lang.vocab_size,))
            f.write("%s, %s -> %s \n" % (output_lang.language_name, len(output_lang.word_to_count), 
                            output_lang.vocab_size))
    print(input_lang.index_to_word)    
    return input_lang, output_lang, train_pairs, test_pairs, val_pairs

"""converts a sentence to one hot encoding vectors - pytorch allows us to just
use the number corresponding to the unique index for that word,
rather than a complete one hot encoding vector for each word"""
def indexesFromSentence(lang, sentence, lang_type):
    indexes = []
    if lang_type=="ls_source":    
        words =  sourceNormalization(sentence)
        for word in words:
            try:
                indexes.append(lang.word_to_index[word])
            except:
                indexes.append(lang.word_to_index["<UNK>"])
    else:
        for word in sentence.split(' '):
            try:
                indexes.append(lang.word_to_index[word])
            except:
                indexes.append(lang.word_to_index["<UNK>"])
    return indexes


def tensorFromSentence(lang, sentence, lang_type):
    indexes = indexesFromSentence(lang, sentence, lang_type)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1)
    if use_cuda:
        return result.cuda()
    else:
        return result
      
"""converts a pair of sentence (input and target) to a pair of tensors"""
def tensorsFromPair(input_lang, output_lang, pair):
    input_variable = tensorFromSentence(input_lang, pair[0], 'ls_source')
    target_variable = tensorFromSentence(output_lang, pair[1], 'ls_target')
    return (input_variable, target_variable)
  

"""converts from tensor of one hot encoding vector indices to sentence"""
def sentenceFromTensor(lang, tensor):
    raw = tensor.data
    words = []
    for num in raw:
        words.append(lang.index_to_word[num.item()])
    return ' '.join(words)

"""seperates data into batches of size batch_size"""
def batchify(data, input_lang, output_lang, batch_size, shuffle_data=True):
    if shuffle_data == True:
        shuffle(data)
    number_of_batches = len(data) // batch_size
    batches = list(range(number_of_batches))
    longest_elements = list(range(number_of_batches))
    print('number of batch', number_of_batches)
    for batch_number in range(number_of_batches):
        #print('batch_number', batch_number)
        longest_input = 0
        longest_target = 0
        input_variables = list(range(batch_size))
        target_variables = list(range(batch_size))
        index = 0      
        for pair in range((batch_number*batch_size),((batch_number+1)*batch_size)):
            input_variables[index], target_variables[index] = tensorsFromPair(input_lang, output_lang, data[pair])
            if len(input_variables[index]) >= longest_input:
                longest_input = len(input_variables[index])
            if len(target_variables[index]) >= longest_target:
                longest_target = len(target_variables[index])
            index += 1
        batches[batch_number] = (input_variables, target_variables)
        longest_elements[batch_number] = (longest_input, longest_target)
    return batches , longest_elements, number_of_batches


"""pads batches to allow for sentences of variable lengths to be computed in parallel"""
def pad_batch(batch):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch[0],padding_value=EOS_token)
    padded_targets = torch.nn.utils.rnn.pad_sequence(batch[1],padding_value=EOS_token)
    return (padded_inputs, padded_targets)

class EncoderRNN(nn.Module):
	def __init__(self,input_size,hidden_size,layers=1,dropout=0.1,
               bidirectional=True):
		super(EncoderRNN, self).__init__()

		if bidirectional:
			self.directions = 2
		else:
			self.directions = 1
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = layers
		self.dropout = dropout
		self.embedder = nn.Embedding(input_size,hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,
                        num_layers=layers,dropout=dropout,
                        bidirectional=bidirectional,batch_first=False)
		self.fc = nn.Linear(hidden_size*self.directions, hidden_size)

	def forward(self, input_data, h_hidden, c_hidden):
		embedded_data = self.embedder(input_data)
		embedded_data = self.dropout(embedded_data)
		hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))

		return hiddens, outputs

	"""creates initial hidden states for encoder corresponding to batch size"""
	def create_init_hiddens(self, batch_size):
		h_hidden = Variable(torch.zeros(self.num_layers*self.directions, 
                                    batch_size, self.hidden_size))
		c_hidden = Variable(torch.zeros(self.num_layers*self.directions, 
                                    batch_size, self.hidden_size))
		if torch.cuda.is_available():
			return h_hidden.cuda(), c_hidden.cuda()
		else:
			return h_hidden, c_hidden

class DecoderAttn(nn.Module):
	def __init__(self, hidden_size, output_size, layers=1, dropout=0.1, bidirectional=True):
		super(DecoderAttn, self).__init__()

		if bidirectional:
			self.directions = 2
		else:
			self.directions = 1
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.num_layers = layers
		self.dropout = dropout
		self.embedder = nn.Embedding(output_size,hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.score_learner = nn.Linear(hidden_size*self.directions, 
                                   hidden_size*self.directions)
		self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,
                        num_layers=layers,dropout=dropout,
                        bidirectional=bidirectional,batch_first=False)
		self.context_combiner = nn.Linear((hidden_size*self.directions)
                                      +(hidden_size*self.directions), hidden_size)
		self.tanh = nn.Tanh()
		self.output = nn.Linear(hidden_size, output_size)
		self.soft = nn.Softmax(dim=1)
		self.log_soft = nn.LogSoftmax(dim=1)


	def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):

		embedded_data = self.embedder(input_data)
		embedded_data = self.dropout(embedded_data)	
		batch_size = embedded_data.shape[1]
		hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))	
		top_hidden = outputs[0].view(self.num_layers,self.directions,
                                 hiddens.shape[1],
                                 self.hidden_size)[self.num_layers-1]
		top_hidden = top_hidden.permute(1,2,0).contiguous().view(batch_size,-1, 1)

		prep_scores = self.score_learner(encoder_hiddens.permute(1,0,2))
		scores = torch.bmm(prep_scores, top_hidden)
		attn_scores = self.soft(scores)
		con_mat = torch.bmm(encoder_hiddens.permute(1,2,0),attn_scores)
		h_tilde = self.tanh(self.context_combiner(torch.cat((con_mat,
                                                         top_hidden),dim=1)
                                              .view(batch_size,-1)))
		pred = self.output(h_tilde)
		pred = self.log_soft(pred)

		
		return pred, outputs

'''Performs training on a single batch of training data. Computing the loss 
according to the passed loss_criterion and back-propagating on this loss.'''

def train_batch(input_batch, target_batch, encoder, decoder, 
                encoder_optimizer, decoder_optimizer, loss_criterion, output_lang, clip):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0
	enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])

	enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)

	decoder_input = Variable(torch.LongTensor(1,input_batch.shape[1]).
                           fill_(output_lang.word_to_index.get("SOS")).cuda()) if use_cuda \
					else Variable(torch.LongTensor(1,input_batch.shape[1]).
                        fill_(output_lang.word_to_index.get("SOS")))

	dec_h_hidden = enc_outputs[0]
	dec_c_hidden = enc_outputs[1]
	
	for i in range(target_batch.shape[0]):
		pred, dec_outputs = decoder(decoder_input, dec_h_hidden, 
                                dec_c_hidden, enc_hiddens)

		decoder_input = target_batch[i].view(1,-1)
		dec_h_hidden = dec_outputs[0]
		dec_c_hidden = dec_outputs[1]
		
		loss += loss_criterion(pred,target_batch[i])


	loss.backward()

	torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
	torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_batch.shape[0]

'''Performs a complete epoch of training through all of the training_batches'''

def train(train_batches, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion, output_lang, clip):

    round_loss = 0
    i = 1
    for batch in tqdm(train_batches):
        #print("Batch ke", i)
        i += 1
        (input_batch, target_batch) = pad_batch(batch)
        batch_loss = train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion, output_lang, clip)
        round_loss += batch_loss

    return round_loss / len(train_batches)

'''Evaluates the loss on a single batch of test data. Computing the loss 
according to the passed loss_criterion. Does not perform back-prop'''

def test_batch(input_batch, target_batch, encoder, decoder, loss_criterion, output_lang):
	
	loss = 0

	#create initial hidde state for encoder
	enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])

	enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)

	decoder_input = Variable(torch.LongTensor(1,input_batch.shape[1]).
                           fill_(output_lang.word_to_index.get("SOS")).cuda()) if use_cuda \
					else Variable(torch.LongTensor(1,input_batch.shape[1]).
                        fill_(output_lang.word_to_index.get("SOS")))
	dec_h_hidden = enc_outputs[0]
	dec_c_hidden = enc_outputs[1]
	
	for i in range(target_batch.shape[0]):
		pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

		topv, topi = pred.topk(1,dim=1)
		ni = topi.view(1,-1)
		
		decoder_input = ni
		dec_h_hidden = dec_outputs[0]
		dec_c_hidden = dec_outputs[1]

		loss += loss_criterion(pred,target_batch[i])
		
	return loss.item() / target_batch.shape[0]

'''Computes the loss value over all of the test_batches'''

def test(test_batches, encoder, decoder, loss_criterion, output_lang):
    print("Testing process")
    with torch.no_grad():
        test_loss = 0

        for batch in tqdm(test_batches):
            (input_batch, target_batch) = pad_batch(batch)
            batch_loss = test_batch(input_batch, target_batch, encoder, decoder, loss_criterion, output_lang)
            test_loss += batch_loss

    return test_loss / len(test_batches)

'''Returns the predicted translation of a given input sentence. Predicted
translation is trimmed to length of cutoff_length argument'''
def evaluated(encoder, decoder, sentence, cutoff_length, input_lang, output_lang):
    #print(encoder)
    with torch.no_grad():
        input_variable = tensorFromSentence(input_lang, sentence, 'ls_source')
        input_variable = input_variable.view(-1,1)
        enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(1)

        enc_hiddens, enc_outputs = encoder(input_variable, enc_h_hidden, enc_c_hidden)

        decoder_input = Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_index.get("SOS")).cuda()) if use_cuda \
						else Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_index.get("SOS")))
        dec_h_hidden = enc_outputs[0]
        dec_c_hidden = enc_outputs[1]

        decoded_words = []
        #print("test", output_sentence)
        for di in range(cutoff_length):
            #print(di)
            ##print(cutoff_length)Dear reviewers,
Thank you for your useful and insightful feedback on our work. Please find the answers to your questions as well as pointers to changes made to the paper in the following.

REVIEW 1

1. ESBM dataset

Although your statements are pertinent, ESBM is the largest available benchmark for intrinsic evaluation of entity summarization and the most used in all state-of-the-art papers. Please take a look at a recent survey [1].  

[1] - Liu, Q., Cheng, G., Gunaratna, K., & Qu, Y. (2021). Entity summarization: State of the art and future challenges. Journal of Web Semantics, 100647.
  
2. NDCG scores. 

Our focus was to compare F-measures between existing works. However, we will provide NDCG scores in the experimental results part of the final version of the paper. 

3. Quantitative and Qualitative results. 

We assumed that by achieving good quantitative results on ESBM which was manually created containing several types of summarization, we reach good qualitative results as well. However, we will take the reviewer’s comment into consideration in future work.
  
4. Enterprise Knowledge Graphs. 

Currently, GATES focuses on generating entity summaries on general-purpose. GATES can be implemented on a specific domain, such as an enterprise knowledge graph. To this end, pre-trained models must be generated from the enterprise knowledge graph (dataset) to support knowledge graph embedding and text embedding, respectively. Subsequently, GATES will be trained using the dataset before it can be used to predict summaries in the related domain. Furthermore, In the conclusion part of our paper (page 13), we plan to extend GATES so that it can be applied on specific domain datasets, such as BioASQ.

5. Tf-idf and other predicate rankings. 

In this work, we only exploited the tf-idf method to calculate weight on predicates. We plan to investigate more ranking methods such as LexRank in future work.    


REVIEW 2
1.  GATES paper is not enough innovative material IMHO. It seems that the approach is a combination of existing techniques.

As we mentioned in the introduction of the GATES paper (page 2), none of the current approaches exploits adjacency information in knowledge graph explicitly. We hence demonstrated that including more structure information into the entity summarization process can lead to better entity summaries. Therefore, the GATES paper showed that the authors of previous state-of-the-art approaches were mistaken when affirming that graph structure would not support improving the summaries. GATES leverages Graph Attention Networks to exploit information from both (1) triple and (2) the topology of knowledge graphs, and finally generate entity summaries. Additionally, to the best of our knowledge, GATES is the first that introduces graph-based neural networks on entity summarization tasks. 

2. The paper has several formal issues. 


We will revise the definition of the KG in the final version of the paper accordingly. 

REVIEW 3

1. Unclear problem definition and doubts on Semantic Web foundations and Entity Summarization task.

We defined the research problem following the literature and previous papers. Also, the problem definition does not seem to be a problem for the other reviewers. However, we will add the information pertaining to how to determine if a summary is good or bad, although it is wide-spread in the community (via F-measure), [1].

Regarding the doubts on CBD, we invite the reviewer to read the book [2]. Also, for understanding the entity summarization task, we also invite the reviewer to read the recent published survey [1]


References:
[1]Liu, Q., Cheng, G., Gunaratna, K., & Qu, Y. (2021). Entity summarization: State of the art and future challenges. Journal of Web Semantics, 100647.
[2] Hitzler, Pascal, Markus Krotzsch, and Sebastian Rudolph. Foundations of semantic web technologies. CRC press, 2009.
            pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
            #print(pred)
            topv, topi = pred.topk(1,dim=1)
            ni = topi.item()
            
            if ni == output_lang.word_to_index.get("EOS"):
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index_to_word[ni])
            decoder_input = Variable(torch.LongTensor(1,1).fill_(ni).cuda()) if use_cuda \
							else Variable(torch.LongTensor(1,1).fill_(ni))
            dec_h_hidden = dec_outputs[0]
            dec_c_hidden = dec_outputs[1]
        #print(decoded_words)
        if "<EOS>" in decoded_words:
                decoded_words.remove("<EOS>")
        output_sentence = ' '.join(decoded_words)
    
        return output_sentence

def evaluate_randomly(encoder, decoder, pairs, create_txt, print_to, input_lang, output_lang, mode, n=2, trim=100):
    if mode=="test":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_encoder = torch.load("model_enc_weights.pt", map_location=map_location)
        encoder.load_state_dict(checkpoint_encoder)
        encoder.to(device)
        
        checkpoint_decoder = torch.load("model_dec_weights.pt", map_location=map_location)
        decoder.load_state_dict(checkpoint_decoder)
        decoder.to(device)
    
    print("evaluate")
    scores=[]
    meteor_scores = []
    chrf_scores = []
    highest_score = 0
    lowest_score = 0
    meteor_highest_score = 0
    meteor_lowest_score = 0
    chrf_highest_score = 0
    chrf_lowest_score = 0
    with open("prediction_output.txt", 'w+') as f_pred:
      f_pred.write("generate prediction output \n")

    with open("target.txt", 'w+') as f_target:
      f_target.write("generate target \n")

    with open("test_output.txt", "w", encoding="utf-8") as f_out:
        for i in tqdm(range(n)):
            #print("Testing progress: {}/{}".format(i, n))
            pair = pairs[i]
            #print('>', pair[0])
            #print('=', pair[1])
            output_sentence = evaluated(encoder, decoder, pair[0], trim, input_lang, output_lang)
            #print('<', output_sentence)
            #print('')    
            if create_txt:
                f = open(print_to, 'a')
                f.write("\n \
    				> %s \n \
    				= %s \n \
    				< %s \n" % (pair[0], pair[1], output_sentence))
            f.close()
            f_out.write("S = {}\n".format(pair[0]))
            f_out.write("T = {}\n".format(pair[1]))
            f_out.write("O = {}\n".format(output_sentence))
            with open("target.txt", 'a') as f_target:
              f_target.write("{}\n".format(pair[1]))
            
            with open("prediction_output.txt", 'a') as f_pred:
              f_pred.write("{}\n".format(output_sentence))
              
            ##print('')
            reference = pair[1].split(' ')
            candidate = output_sentence.split(' ')
            #print('reference', reference)
            #print('candidate', candidate)
            smoothie = SmoothingFunction().method4
            score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
            meteor_score = round(single_meteor_score(pair[1], output_sentence),4)
            chrf_score = sentence_chrf(pair[1], candidate)
            ##print("BLEU Score: ",score)
            f_out.write("BLEU SCORE = {}\n".format(score*100))
            f_out.write("METEOR SCORE = {}\n".format(meteor_score*100))
            f_out.write("ChrF++ SCORE = {}\n".format(chrf_score*100))
            f_out.write("\n")
            scores.append(score)
            meteor_scores.append(meteor_score)
            chrf_scores.append(chrf_score)

            if i==0:
              highest_score = score
              lowest_score = score
              meteor_highest_score = meteor_score
              meteor_lowest_score = meteor_score
              chrf_highest_score = chrf_score
              chrf_lowest_score = chrf_score

            if score > highest_score:
              highest_score=score
            if score < lowest_score:
              lowest_score = score
            
            if meteor_score > meteor_highest_score:
              meteor_highest_score=meteor_score
            if meteor_score < meteor_lowest_score:
              meteor_lowest_score = meteor_score

            if chrf_score > chrf_highest_score:
              chrf_highest_score=chrf_score
            if chrf_score < chrf_lowest_score:
              chrf_lowest_score = chrf_score
            #print("BLEU SCORE = {}\n".format(score))
            #print("METEOR SCORE = {}\n".format(meteor_score))
            #print("METEOR SCORE = {}\n".format(meteor_score))
        avg=(sum(scores)/len(scores))*100
        meteor_scores_avg = (sum(meteor_scores)/len(meteor_scores))*100
        chrf_scores_avg = (sum(chrf_scores)/len(chrf_scores))*100
        highest_score = highest_score*100
        lowest_score = lowest_score*100
        meteor_highest_score = meteor_highest_score*100
        meteor_lowest_score = meteor_lowest_score*100
        chrf_highest_score = chrf_highest_score*100
        chrf_lowest_score = chrf_lowest_score*100
        print("AVG BLEU Score", avg, "Highest", highest_score, "Lowest", lowest_score)
        print("AVG METEOR Score", meteor_scores_avg, "Highest", meteor_highest_score, "Lowest", meteor_lowest_score)
        print("AVG ChfF++ Score", chrf_scores_avg, "Highest", chrf_highest_score, "Lowest", chrf_lowest_score)
        f_out.write("AVG Bleu Score {} highest {} lowest {}\n".format(avg, highest_score, lowest_score))
        f_out.write("AVG METEOR Score {} highest {} lowest {}\n".format(meteor_scores_avg, meteor_highest_score, meteor_lowest_score))
        f_out.write("AVG ChrF++ Score {} highest {} lowest {}\n".format(chrf_scores_avg, chrf_highest_score, chrf_lowest_score))

'''Used to plot the progress of training. Plots the loss value vs. time'''
def showPlot(epochs, losses, fig_name):
    colors = ('red','blue')
    x_axis_label = 'Epochs'
    i = 0
    for key, losses in losses.items():
      if len(losses) > 0:
        plt.plot(epochs, losses, label=key, color=colors[i])
        i += 1
    plt.legend(loc='upper left')
    plt.xlabel(x_axis_label)
    plt.ylabel('Loss')
    plt.title('Training Results')
    plt.savefig(fig_name+'.png')
    plt.close('all')
    
'''prints the current memory consumption'''
def mem():
	if use_cuda:
		mem = torch.cuda.memory_allocated()/1e7
	else:
		mem = psutil.cpu_percent()
	print('Current mem usage:')
	print(mem)
	return "Current mem usage: %s \n" % (mem)

'''converts a time measurement in seconds to hours'''
def asHours(s):
	m = math.floor(s / 60)
	h = math.floor(m / 60)
	s -= m * 60
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)

'''The master function that trains the model. Evlautes progress on the train set
(if present) and also records the progress of training in both a txt file and
a png graph. Also can save the weights of both the Encoder and Decoder 
for future use.'''

def train_and_test(epochs, val_eval_every, plot_every, learning_rate, lr_schedule, train_pairs, val_pairs, input_lang, output_lang, batch_size, val_batch_size, encoder, decoder, loss_criterion, trim, save_weights, create_txt, print_to, output_file_name, criterion, clip):
    times = []
    arEpochs = []
    losses = {'Training set':[], 'Validation set': []}
    print("Batching process")
    print('len test_pairs', len(val_pairs))
    val_batches, longest_seq, n_o_b = batchify(val_pairs, input_lang, 
                                              output_lang, val_batch_size, 
                                              shuffle_data=False)
    start = time.time()
    print('start', start)
    for i in range(1,epochs+1):
        arEpochs.append(i)
        print('epoch', i)
        if i in lr_schedule.keys():
            learning_rate /= lr_schedule.get(i)


        encoder.train()
        decoder.train()

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        batches, longest_seq, n_o_b = batchify(train_pairs, input_lang, 
                                           output_lang, batch_size, 
                                           shuffle_data=True)
        print("train_batches")
        train_loss = train(batches, encoder, decoder, encoder_optimizer, 
                       decoder_optimizer, loss_criterion, output_lang, clip)
		
        now = time.time()
        print("Iter: %s \nLearning Rate: %s \nTime: %s \nTrain Loss: %s \n" % (i, learning_rate, asHours(now-start), train_loss))

        if create_txt:
            with open(print_to, 'a') as f:
                f.write("Iter: %s \nLeaning Rate: %s \nTime: %s \nTrain Loss: %s \n" % (i, learning_rate, asHours(now-start), train_loss))

        if i % val_eval_every == 0:
            if val_pairs:
                val_loss = test(val_batches, encoder, decoder, loss_criterion, output_lang)
                print("Test set loss: %s" % (val_loss))
                if create_txt:
                    with open(print_to, 'a') as f:
                        f.write("Test Loss: %s \n" % (val_loss))
            #evaluate_randomly(encoder, decoder, val_pairs, create_txt, print_to, input_lang, output_lang, "train", 2, trim)

        if i % plot_every == 0:
            times.append((time.time()-start)/60)
            losses['Training set'].append(train_loss)
            if val_pairs:
                losses['Validation set'].append(val_loss)
            showPlot(arEpochs, losses, output_file_name)
            if save_weights:
                torch.save(encoder.state_dict(), output_file_name+'_enc_weights.pt')
                torch.save(decoder.state_dict(), output_file_name+'_dec_weights.pt')


def main(clip, mode, max_length, data_dir, n_epoch, bidirectional):
    create_txt=True
    output_file_name="model"
    
    if create_txt:
    	print_to = output_file_name+'.txt'
    	with open(print_to, 'w+') as f:
    		f.write("Starting Training \n")
    else:
    	print_to = None
    
    """Remove sentences from dataset that are longer than trim (in either language)"""
    if max_length==-1 :
        trim=0
    else:
        trim = max_length
    
    """OUTPUT OPTIONS"""
    
    """denotes how often to evaluate a loss on the test set and print
    sample predictions on the test set.
    if no test set, simply prints sample predictions on the train set."""
    test_eval_every = 1
    
    """denotes how often to plot the loss values of train and test (if applicable)"""
    plot_every = 1
    
    """if true creates a txt file of the output"""
    create_txt = True
    
    """if true saves the encoder and decoder weights to seperate .pt files for later use"""
    save_weights = True
    
    """HYPERPARAMETERS: FEEL FREE TO PLAY WITH THESE TO TRY TO ACHIEVE BETTER RESULTS"""
    
    """signifies whether the Encoder and Decoder should be bidirectional LSTMs or not"""
    bidirectional = bidirectional
    
    """number of layers in both the Encoder and Decoder"""
    layers = 2
    
    """Hidden size of the Encoder and Decoder"""
    hidden_size = 256
    
    """Dropout value for Encoder and Decoder"""
    dropout = 0.1
    
    """Training set batch size"""
    batch_size = 128
    
    """Test set batch size"""
    test_batch_size = 128
    
    """number of epochs (full passes through the training data)"""
    epochs = n_epoch
    
    """Initial learning rate"""
    learning_rate= 0.1
    
    """Learning rate schedule. Signifies by what factor to divide the learning rate
    at a certain epoch. For example {5:10} would divide the learning rate by 10
    before the 5th epoch and {5:10, 10:100} would divide the learning rate by 10
    before the 5th epoch and then again by 100 before the 10th epoch"""
    lr_schedule = {}
    
    """loss criterion, see https://pytorch.org/docs/stable/nn.html for other options"""
    criterion = nn.NLLLoss()
    
    input_lang, output_lang, train_pairs, test_pairs, val_pairs = prepareData('ls_source', 'ls_target', data_dir, trim=trim, print_to=print_to)
    
    if create_txt:
    	with open(print_to, 'a') as f:
    		f.write("\nRandom Train Pair: %s \n\nRandom Test Pair: %s \n\n" % (random.choice(train_pairs),random.choice(test_pairs) if test_pairs else "None"))
    		f.write(mem())
    
    print_to=print_to
    """create the Encoder"""
    encoder = EncoderRNN(input_lang.vocab_size, hidden_size, layers=layers, 
                         dropout=dropout, bidirectional=bidirectional)
    
    """create the Decoder"""
    decoder = DecoderAttn(hidden_size, output_lang.vocab_size, layers=layers, 
                          dropout=dropout, bidirectional=bidirectional)
    
    print('Encoder and Decoder Created')
    if use_cuda:
    	print('Cuda being used')
    	encoder = encoder.cuda()
    	decoder = decoder.cuda()
    
    print('Number of epochs: '+str(epochs))
    
    if create_txt:
    	with open(print_to, 'a') as f:
    		f.write('Encoder and Decoder Created\n')
    		f.write(mem())
    		f.write("Numbebidirectional = Truer of epochs %s \n" % (epochs))
    
    if mode=="train" or mode=="all":
        train_and_test(epochs, test_eval_every, plot_every, learning_rate, lr_schedule, 
                   train_pairs, test_pairs, input_lang, output_lang, batch_size, 
                   test_batch_size, encoder, decoder, criterion, trim, save_weights, create_txt, print_to, output_file_name, criterion, clip)
    
    if mode=="test" or mode=="all":
        evaluate_randomly(encoder, decoder, test_pairs, create_txt, print_to, input_lang, output_lang, "test", len(test_pairs), trim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Machine Translation Link Specification (NMTLS)')
    #parser.add_argument("--filename", type=str, default="", help="Input file")
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--mode', type=str, default="test", help='Mode: train/test')
    parser.add_argument('--max_length', type=int, default=-1, help="Max length")
    parser.add_argument('--data_dir', type=str, default="", help='Directory of dataset')
    parser.add_argument('--n_epoch', type=int, default=10, help="n Epoch")
    parser.add_argument('--bidirectional', type=bool, default=True, help="Bidirectional")
    
    args = parser.parse_args()
    #print(args)
    main(args.clip, args.mode, args.max_length, args.data_dir, args.n_epoch, args.bidirectional)
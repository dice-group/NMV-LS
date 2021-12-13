#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch

from preprocess import prepareData
from models import EncoderRNN, AttnDecoderRNN
from train import trainIters, evaluateRandomly
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
print_every=100
plot_every=1
learning_rate = 0.01
dropout_p=0.1

def main(directory, mode, max_length, model, n_epoch):
    epoch=n_epoch
    input_lang, output_lang, train_pairs, test_pairs, dev_pairs, max_length = prepareData("ls_source", "ls_target", directory, max_length)
    #max_length = 80
    hidden_size = 256
    if model=="gru":
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p, max_length).to(device)
    print(encoder1)
    print(attn_decoder1)    
    if mode=="train" or mode=="all":
        if model=="gru":
            trainIters(encoder1, attn_decoder1, epoch, train_pairs, dev_pairs, test_pairs, input_lang, output_lang, print_every, plot_every, learning_rate, max_length, device, SOS_token, EOS_token, teacher_forcing_ratio)
        plt.show()
    
    if mode=="test" or mode=="all":
        if model=="gru":
            evaluateRandomly(encoder1, attn_decoder1, test_pairs, input_lang, output_lang, max_length, device, SOS_token, EOS_token, n=len(test_pairs))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Link Spesification Verbalization (NNLSV) ')
    parser.add_argument("--directory", type=str, default="", help="Data Directory")
    parser.add_argument("--mode", type=str, default="test", help="Mode operations: train, test, and all")
    parser.add_argument("--max_length", type=int, default="0", help="Msx length")
    parser.add_argument("--model", type=str, default="gru", help="Model: gru/lstm")
    parser.add_argument("--n_epoch", type=int, default="10000", help="N epochs")
    
    
    args = parser.parse_args()
    main(args.directory, args.mode, args.max_length, args.model, args.n_epoch)

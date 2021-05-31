#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:26:47 2021

@author: Asep Fajar Firmansyah
"""
import torch
import random
import time
import math
import torch.nn as nn
from torch import optim
from helpers import tensorsFromPair, timeSince, tensorFromSentence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score 
from nltk.translate.chrf_score import sentence_chrf
import matplotlib.pyplot as plt
from tqdm import tqdm
#import visdom
#viz = visdom.Visdom()

def train(train_input_tensor, train_target_tensor, val_input_tensor, val_target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device, SOS_token, EOS_token, teacher_forcing_ratio):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    train_input_length = train_input_tensor.size(0)
    train_target_length = train_target_tensor.size(0)
    
    val_input_length = val_input_tensor.size(0)
    val_target_length = val_target_tensor.size(0)
    #print(input_length)
    #print("###")
    #print(target_length)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    #Training process
    encoder.train()
    decoder.train()
    loss = 0

    for ei in range(train_input_length):
        #print(ei)
        encoder_output, encoder_hidden = encoder(
            train_input_tensor[ei], encoder_hidden)
        #print(encoder_output[ei].shape)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next inputplt.show()
        for di in range(train_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, train_target_tensor[di])
            decoder_input = train_target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(train_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, train_target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    #Validation process
    encoder.eval()
    decoder.eval()
    val_loss = 0
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    with torch.no_grad():
        for ei in range(val_input_length):
            #print(ei)
            encoder_output, encoder_hidden = encoder(
                val_input_tensor[ei], encoder_hidden)
            #print(encoder_output[ei].shape)
            encoder_outputs[ei] = encoder_output[0, 0]
    
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
    
        use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False
    
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next inputplt.show()
            for di in range(val_target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                val_loss += criterion(decoder_output, val_target_tensor[di])
                decoder_input = val_target_tensor[di]  # Teacher forcing
    
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(val_target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
    
                val_loss += criterion(decoder_output, val_target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
    
        
    train_loss = loss.item() / train_target_length
    validation_loss = val_loss.item()/ val_target_length
    return train_loss, validation_loss

def trainIters(encoder, decoder, n_iters, train_pairs, dev_pairs, test_pairs, input_lang, output_lang, print_every, plot_every, learning_rate, max_length, device, SOS_token, EOS_token, teacher_forcing_ratio):
    
    output_file_name="model"
    print_to = output_file_name+'.txt'
    with open(print_to, 'w+') as f_log:
    	f_log.write("Starting Training \n")

    now = time.time()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_val_loss_total=0
    save_weights=True
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #print(len(train_pairs))
    training_pairs = [tensorsFromPair(random.choice(train_pairs), input_lang, output_lang, device)
                      for i in range(n_iters)]
    
    validation_pairs = [tensorsFromPair(random.choice(train_pairs), input_lang, output_lang, device)
                      for i in range(n_iters)]
    
    criterion = nn.NLLLoss()
    #viz.line([[0.0], [0.0]], [0.], win='{}_loss'.format("train"), opts=dict(title='train loss', legend=['train loss', 'validation loss']))
    arEpochs=[]
    losses = {'Training set':[], 'Validation set': []}

    with open("training_loss_log.csv", 'w+') as f_loss:
      f_loss.write("Train Set \t Validation Set \n")

    for iter in range(1, n_iters + 1):
        arEpochs.append(iter)
        training_pair = training_pairs[iter - 1]
        train_input_tensor = training_pair[0]
        train_target_tensor = training_pair[1]
        
        validation_pair = validation_pairs[iter - 1]
        val_input_tensor = validation_pair[0]
        val_target_tensor = validation_pair[1]

        train_loss, validation_loss = train(train_input_tensor, train_target_tensor, val_input_tensor, val_target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device, SOS_token, EOS_token, teacher_forcing_ratio)
        print_loss_total += train_loss
        plot_loss_total += train_loss
        
        print_val_loss_total += validation_loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_val_loss_avg = print_val_loss_total / print_every
            print_loss_total = 0
            print_val_loss_total=0
            
            print('%s (%d %d%%) %s %.4f %s %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, "traning loss", print_loss_avg, "validation loss", print_val_loss_avg))
            with open(print_to, 'a') as f_log:
              f_log.write("Iter: %s \nLeaning Rate: %s \nTime: %s \nTrain Loss: %s \n" % (iter, learning_rate, asHours(now-start), print_loss_avg))
              f_log.write("Test Loss: %s \n" % (print_val_loss_avg))

        if iter % plot_every == 0:
            losses['Training set'].append(train_loss)
            losses['Validation set'].append(validation_loss)
            showPlot(arEpochs, losses, "train_eval_losses")
            
            with open("training_loss_log.csv", 'a') as f_loss:
              f_loss.write("{} \t {} \n".format(train_loss, validation_loss))
            
            if save_weights:
                torch.save(encoder.state_dict(), 'model_enc_weights.pt')
                torch.save(decoder.state_dict(), 'model_dec_weights.pt')

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length, device, SOS_token, EOS_token):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, 'ls_source', device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

##################################################plt.show()####################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, max_length, device, SOS_token, EOS_token, n=10):
    checkpoint_encoder = torch.load("model_enc_weights.pt")
    encoder.load_state_dict(checkpoint_encoder)
    encoder.to(device)
    
    checkpoint_decoder = torch.load("model_dec_weights.pt")
    decoder.load_state_dict(checkpoint_decoder)
    decoder.to(device)
    
    scores = []
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

    with open("test_output.txt", "w", encoding="utf-8") as f:
        for i in tqdm(range(n)):
            pair = pairs[i]
            ##print('>', pair[0])
            ##print('=', pair[1])
            f.write("S = {}\n".format(pair[0]))
            f.write("T = {}\n".format(pair[1]))
            with open("target.txt", 'a') as f_target:
              f_target.write("{}\n".format(pair[1]))

            output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang, max_length, device, SOS_token, EOS_token)
            if "<EOS>" in output_words:
                output_words.remove("<EOS>")
            output_sentence = ' '.join(output_words)
            #output_sentence = re.sub("<EOS>", "", output_sentence)
            #output_sentence = output_sentence.rstrip()
            ##print('<', output_sentence)
            f.write("O = {}\n".format(output_sentence))
            with open("prediction_output.txt", 'a') as f_pred:
              f_pred.write("{}\n".format(output_sentence))

            ##print('')
            reference = pair[1].split(' ')
            candidate = output_words
            #print('reference', reference)
            #print('candidate', candidate)
            smoothie = SmoothingFunction().method4
            score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
            meteor_score = round(single_meteor_score(pair[1], output_sentence),4)
            chrf_score = sentence_chrf(pair[1], candidate)
            ##print("BLEU Score: ",score)
            f.write("BLEU SCORE = {}\n".format(score*100))
            f.write("METEOR SCORE = {}\n".format(meteor_score*100))
            f.write("ChrF++ SCORE = {}\n".format(chrf_score*100))
            f.write("\n")
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
        f.write("AVG Bleu Score {} highest {} lowest {}\n".format(avg, highest_score, lowest_score))
        f.write("AVG METEOR Score {} highest {} lowest {}\n".format(meteor_scores_avg, meteor_highest_score, meteor_lowest_score))
        f.write("AVG ChrF++ Score {} highest {} lowest {}\n".format(chrf_scores_avg, chrf_highest_score, chrf_lowest_score))

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

def asHours(s):
	m = math.floor(s / 60)
	h = math.floor(m / 60)
	s -= m * 60
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)

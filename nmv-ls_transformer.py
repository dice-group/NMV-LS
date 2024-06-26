#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch.nn.init import xavier_uniform_
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
from torchtext.legacy import data
from preprocess import prepareData
from tqdm import tqdm
from helpers import sourceNormalization
import nltk
nltk.download('wordnet')

# Special Tokens
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

def tokenize_source(text):
    return sourceNormalization(text)

def tokenize_target(text):
    text = text.strip()
    return text.split(' ')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=107, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class MyTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu",source_vocab_length: int = 60000,target_vocab_length: int = 60000, max_len: int = 107) -> None:
        super(MyTransformer, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.target_embedding = nn.Embedding(target_vocab_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.out = nn.Linear(512, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.out(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

def train(train_iter, val_iter, model, optim, num_epochs, BATCH_SIZE, use_gpu=True): 
    output_file_name="model"
    print_to = output_file_name+'.txt'
    with open(print_to, 'w+') as f_log:
    	f_log.write("Starting Training \n")
    
    arEpochs=[]
    losses = {'Training set':[], 'Validation set': []}

    with open("training_loss_log.csv", 'w+') as f_loss:
      f_loss.write("Train Set \t Validation Set \n")


    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(num_epochs)):
        arEpochs.append(epoch)

        train_loss = 0
        valid_loss = 0
        # Train model
        model.train()
        for i, batch in enumerate(train_iter):
            
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg
            #change to shape (bs , max_seq_len)
            src = src.transpose(0,1)
            #change to shape (bs , max_seq_len+1) , Since right shifted
            trg = trg.transpose(0,1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask = (src != 0)
            src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.cuda() if use_gpu else src_mask
            trg_mask = (trg_input != 0)
            trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
            trg_mask = trg_mask.cuda() if use_gpu else trg_mask
            size = trg_input.size(1)
            #print(size)
            np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask.cuda() if use_gpu else np_mask   
            # Forward, backprop, optimizer
            optim.zero_grad()
            preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
            preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
            loss.backward()
            optim.step()
            train_loss += loss.item()/BATCH_SIZE
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_iter):
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg
                #change to shape (bs , max_seq_len)
                src = src.transpose(0,1)
                #change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0,1)
                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)
                src_mask = (src != 0)
                src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
                src_mask = src_mask.cuda() if use_gpu else src_mask
                trg_mask = (trg_input != 0)
                trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
                trg_mask = trg_mask.cuda() if use_gpu else trg_mask
                size = trg_input.size(1)
                #print(size)
                np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))         
                loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
                valid_loss += loss.item()/1
            
        # Log after each epoch
        print(f'''Epoch [{epoch+1}/{num_epochs}] complete. Train Loss: {train_loss/len(train_iter):.3f}. Val Loss: {valid_loss/len(val_iter):.3f}''')
        
        losses['Training set'].append(train_loss/len(train_iter))
        losses['Validation set'].append(valid_loss/len(val_iter))
        showPlot(arEpochs, losses, "train_eval_losses")
        with open("training_loss_log.csv", 'a') as f_loss:
          f_loss.write("{} \t {} \n".format(train_loss/len(train_iter), valid_loss/len(val_iter)))
        #Save best model till now:
        if valid_loss/len(val_iter)<min(valid_losses,default=1e9): 
            print("saving state dict")
            torch.save(model.state_dict(), f"checkpoint_best_epoch.pt")
        
        train_losses.append(train_loss/len(train_iter))
        valid_losses.append(valid_loss/len(val_iter))
    return train_losses,valid_losses

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

def greeedy_decode_sentence(model,sentence, maxlen, SRC, TGT):
    model.eval()
    sentence = SRC.preprocess(sentence)
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 :
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
    sentence = Variable(torch.LongTensor([indexed])).cuda()
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).cuda()
    translated_sentence = ""
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda()
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]
        
        if add_word==EOS_WORD:
            break
        else:
          translated_sentence+=" "+add_word
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        #print(trg)
    return translated_sentence

def evaluateRandomly(model, pairs, maxlen, SRC, TGT):
  with open("prediction_output.txt", 'w+') as f_pred:
    f_pred.write("")

  with open("target.txt", 'w+') as f_target:
    f_target.write("")

  with open("test_output.txt", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(pairs))):
      pair = pairs[i]
      sentence = pair[0]

      f.write("S = {}\n".format(pair[0]))
      f.write("T = {}\n".format(pair[1]))
      with open("target.txt", 'a') as f_target:
        f_target.write("{}\n".format(pair[1]))

      prediction = greeedy_decode_sentence(model,sentence, maxlen, SRC, TGT)

      f.write("O = {}\n".format(prediction))
      with open("prediction_output.txt", 'a') as f_pred:
        f_pred.write("{}\n".format(prediction))

def main(directory, mode, max_length, n_epoch, batch_size):
  SRC = data.Field(tokenize=tokenize_source, pad_token=BLANK_WORD)
  TGT = data.Field(tokenize=tokenize_target, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)
  
  #SOURCE = data.Field(batch_first=True)
  #TARGET = data.Field(sequential=False, unk_token=None)
  train_data, val_data, test_data = data.TabularDataset.splits(  
    path=directory, train='train.txt',
    validation='dev.txt', test='test.txt', format='tsv',
    fields=[('src', SRC), ('trg', TGT)], filter_pred=lambda x: len(vars(x)['src']) <= max_length
    and len(vars(x)['trg']) <= max_length)
  
  MIN_FREQ = 2
  SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
  TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)

  BATCH_SIZE =batch_size
  num_epochs = n_epoch
  # Create iterators to process text in batches of approx. the same length
  train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))
  val_iter = data.BucketIterator(val_data, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))

  source_vocab_length = len(SRC.vocab)
  target_vocab_length = len(TGT.vocab)
 
  model = MyTransformer(source_vocab_length=source_vocab_length,target_vocab_length=target_vocab_length, max_len=max_length)
  optim = torch.optim.Adam(model.parameters(), lr=0.000001)
  #optim = torch.optim.SGD(model.parameters(), lr=0.01)
  model = model.cuda()
  if mode=="train" or mode=="all":
    print("Traning processing ...")
    train_losses,valid_losses = train(train_iter, val_iter, model, optim, num_epochs, BATCH_SIZE)

  if mode=="test" or mode=="all":
    input_lang, output_lang, train_pairs, test_pairs, dev_pairs, max_length = prepareData("ls_source", "ls_target", directory, max_length)
    model.load_state_dict(torch.load(f"checkpoint_best_epoch.pt"))
    print("evaluating processing ...")
    evaluateRandomly(model, test_pairs, max_length, SRC, TGT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Machine Translation Link Specification (NMTLS)')
    #parser.add_argument("--filename", type=str, default="", help="Input file")
    parser.add_argument('--mode', type=str, default="test", help='Mode: train/test')
    parser.add_argument('--max_length', type=int, default=107, help="Max length")
    parser.add_argument('--directory', type=str, default="", help='Directory of dataset')
    parser.add_argument('--n_epoch', type=int, default=30, help="n Epoch")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    
    args = parser.parse_args()
    #print(args)
    main(args.directory, args.mode, args.max_length, args.n_epoch, args.batch_size)
    


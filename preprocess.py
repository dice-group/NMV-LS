#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as path
import argparse
from sklearn.model_selection import train_test_split
import csv
from helpers import Lang, normalizeString, sourceNormalization
import re

def splitDataset(filename):
    data = []
    with open(filename, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            pair = []
            source = row[0]
            target = row[1]
            pair.append(source)
            pair.append(target)
            data.append(pair)
            #print(data)
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    dev, test = train_test_split(test, test_size=0.3334, random_state=42)
    
    return train, test, dev

def filterPairs(pairs, max_length):
    filtered_pairs=[]
    #print(len(pairs))
    for pair in pairs:
        source = sourceNormalization(pair[0])
        target = pair[1].split(' ')
        if len(source) < max_length and len(target)<max_length:
            filtered_pairs.append(pair)
    return filtered_pairs

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
  
def getMaxLength(pairs):
    sources = []
    targets = []
    for pair in pairs:
        source = sourceNormalization(pair[0])
        sources.append(source)
        
        targets.append(pair[1])
    max_source_length = max([len(source) for source in sources])
    max_target_length = [len(target.split(' ')) for target in targets]
    #for target in targets:
    #    print(target.split(' '))
    #    print(len(target.split(' ')))
    max_target_length = max(max_target_length)
    if max_source_length > max_target_length:
        return max_source_length
    else:
        return max_target_length
 
def buildVocabPairs(input_lang, output_lang, lang1, lang2, pairs):
    for pair in pairs:
        #print(pair)
        input_lang.addSentence(pair[0], lang1)
        output_lang.addSentence(pair[1], lang2)
    
    
def prepareData(lang1, lang2, dir_data, max_length):
    
    input_lang, output_lang, train_pairs, test_pairs, dev_pairs = loadDataset("ls_source", "ls_target", dir_data)
    pairs = train_pairs+test_pairs+dev_pairs
    if max_length==0:
        max_length = getMaxLength(pairs)
    print('max length', max_length)
    print("Read %s all pairs" % len(pairs))
    
    train_pairs = filterPairs(train_pairs, max_length)
    test_pairs = filterPairs(test_pairs, max_length)
    dev_pairs = filterPairs(dev_pairs, max_length)
    print("Training pairs", len(train_pairs))
    print("Validation pairs", len(dev_pairs))
    print("Testing pairs", len(test_pairs))
    
    print("Counting words...")
    
    buildVocabPairs(input_lang, output_lang, lang1, lang2, pairs)
        
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(input_lang.index2word)
    return input_lang, output_lang, train_pairs, test_pairs, dev_pairs, max_length    

def main(filename):
    print('Reading the file')
    train, test, dev = splitDataset(filename)
    
    print('Generating file for train, test, and dev dataset')
    directory = path.join("data")
    with open(path.join(directory, "train.txt"), "w", encoding="utf-8") as f:
        for row in train:
            f.write("{}\t{}\n".format(row[0], row[1]))
    with open(path.join(directory, "test.txt"), "w", encoding="utf-8") as f:
        for row in test:
            f.write("{}\t{}\n".format(row[0], row[1]))
    with open(path.join(directory, "dev.txt"), "w", encoding="utf-8") as f:
        for row in dev:
            f.write("{}\t{}\n".format(row[0], row[1]))
    total = len(train+dev+test)
    print("total of data", total)
    percentage_train = (len(train)/total) * 100
    percentage_dev = (len(dev)/total) * 100
    percentage_test = (len(test)/total) * 100
    print('#n training data: {} ({}%)'.format(len(train), percentage_train))
    print('#n testing data: {} ({}%)'.format(len(test), percentage_test))
    print('#n development data: {} ({}%)'.format(len(dev), percentage_dev))
    
    print('Done')

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess graph-based neural LSV ')
    parser.add_argument("--filename", type=str, default="", help="Input file")
    
    args = parser.parse_args()
    main(args.filename)



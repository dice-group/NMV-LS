# NMV-LS: Multilingual Neural Machine Verbalization of Link Specifications for The Explainable Integration of Knowledge Graphs

NMV-LS is a multilingual neural machine verbalization approach for translating complex link specifications into natural language.

## Environment and Dependencies

```
Ubuntu 10.04.2 LTS
python 3.6+
torch 1.7.0
```
## Datasets
There are three different datasets with following size:
1. 107k of pairs (English) 
2. 1m of pairs (English)
3. 73k of pairs (German)

We provide splitted dataset of each dataset in the data folder. Unzip all of zip files, which are each of dataset consists of train, dev, and test sets.

## Installation
Download NVM-LS repository:
```
git clone https://github.com/dice-group/GATES.git
```
Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### Configuration
```
mode: all		# Mode all denotes that you will execute the code on training and testing consecutively. You can choose train or test mode if you want to run separately.
max_length: 107		# The max length of sentence is 107
directory: data/107K/	# Directory denotes where is the path of splitted dataset (train, dev, and test sets)
n_epoch: 100		# n epoch shows how many epochs to train the model
bidirectional: True	# Bidirectional parameter is used for NMV-LS with Bi/LSTM model. If the value of bidirectional is true that shows BiLSTM model is used on training the model.
```
### NMV-LS with GRU
To run NMV-LS with GRU model 
```
$ python main.py --directory data/107K/ --max_length 107 --mode all --n_epoch 100
```

### NMV-LS with Bi/LSTM
To run NMV-LS with BiLSTM model
```
$ python nmv-ls_bilstm.py --directory data/107K/ --max_length 107 --mode all --n_epoch 100 --bidirectional True
```

### NMV-LS with Transformers
To run NMV-LS with Transformer model
```
$ python nmv-ls_transformer.py --directory data/107K/ --max_length 107 --mode all --n_epoch 30
```

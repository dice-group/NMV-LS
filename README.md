# Explainable Integration of Knowledge Graphs using Large Language Models

It is continuous work from the NMV-LS system for translating complex link specifications into natural language that leverages a large language model.

## Environment and Dependencies

```
Ubuntu 10.04.2 LTS
python 3.6+
torch 1.7.0
```
## Datasets for the standard Encoder-Decoder architectures (in data folder)
There are three different datasets with following size:
1. 107k of pairs (English) 
2. 1m of pairs (English)
3. 73k of pairs (German)

We provide splitted dataset of each dataset in the data folder. Unzip all of zip files, which are each of dataset consists of train, dev, and test sets.

## Datasets for few-shot learning scenarios (in datasets folder)
There are four different datasets as follows:
1. LIMES silver
2. Human annotated LIMES silver
3. Human annotated LIMES LS
4. Human annotated SILK LS

## Installation
Download NVM-LS repository:
```
https://github.com/u2018/NMV-LS-T5.git
```
Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### Standard encoder-decoder architecture
#### Configuration
```
mode: all		# Mode all denotes that you will execute the code on training and testing consecutively. You can choose train or test mode if you want to run separately.
max_length: 107		# The max length of sentence is 107
directory: data/107K/	# Directory denotes where is the path of splitted dataset (train, dev, and test sets)
n_epoch: 100		# n epoch shows how many epochs to train the model
bidirectional: True	# Bidirectional parameter is used for NMV-LS with Bi/LSTM model. If the value of bidirectional is true that shows BiLSTM model is used on training the model.
```
#### NMV-LS with GRU
To run NMV-LS with GRU model 
```
$ python main.py --directory data/107K/ --max_length 107 --mode all --n_epoch 100
```

#### NMV-LS with Bi/LSTM
To run NMV-LS with BiLSTM model
```
$ python nmv-ls_bilstm.py --directory data/107K/ --max_length 107 --mode all --n_epoch 100 --bidirectional True
```

#### NMV-LS with Transformers
To run NMV-LS with Transformer model
```
$ python nmv-ls_transformer.py --directory data/107K/ --max_length 107 --mode all --n_epoch 30
```
### Few-shot learning using T5 model
Run [NMVLS_few_shot_learning_using_T5_model.ipynb](https://github.com/u2018/NMV-LS/blob/main/NMVLS_few_shot_learning_using_T5_model.ipynb) on Google colab

## How to Cite
```bibtex
@article{asep2023nmvls,
  author = {Ahmed, Abdullah Fathi and Firmansyah, Asep Fajar and Sherif, Mohamed Ahmed and Mousallem, Diego and Ngomo, Axel-Cyrille Ngonga},
  biburl = {https://www.bibsonomy.org/bibtex/24fd6060de085e75628f880e6a316a098/dice-research},
  title = {Explainable Integration of Knowledge Graphs using Large Language Models},
  url = {https://svn.dice-research.org/open/papers/2023/NLDB_NMVLS/public.pdf},
  year = 2023
}
```

## Contact
If you have any questions or feedbacks, feel free to contact us at asep.fajar.firmansyah@upb.de

# Neural Machine Translation Link Specification (NMTLS)

## Environment and Dependencies

```
Ubuntu 10.04.2 LTS
python 3.6+
torch 1.7.0
```
## Dataset
You can find in the data folder. unzip all of zip files.


## GRU Model

To run the codes for help mode

```
$ python main.py -h

usage: main.py [-h] [--directory DIRECTORY] [--mode MODE]
               [--max_length MAX_LENGTH] [--model MODEL] [--n_epoch N_EPOCH]

Neural Machine Translation Link Specification (NMTLS)

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Data Directory
  --mode MODE           Mode operations: train, test, and all
  --max_length MAX_LENGTH
                        Msx length
  --model MODEL         Model: gru/lstm
  --n_epoch N_EPOCH     N epochs


```

To run the codes for all mode (training and test)

```
$ python main.py --directory data/en_121K/ --max_length 107 --mode all --n_epoch 100

```
## LSTM/BiLSTM Model

To run the codes for help mode

For LSTM, set bidirectional to False, and BiLSTM set bidirectional to True.

```
$ python nmtls-bilstm.py -h

usage: nmtls-bilstm.py [-h] [--clip CLIP] [--mode MODE]
                       [--max_length MAX_LENGTH] [--data_dir DATA_DIR]
                       [--n_epoch N_EPOCH] [--bidirectional BIDIRECTIONAL]

Neural Machine Translation Link Specification (NMTLS)

optional arguments:
  -h, --help            show this help message and exit
  --clip CLIP           gradient clipping
  --mode MODE           Mode: train/test
  --max_length MAX_LENGTH
                        Max length
  --data_dir DATA_DIR   Directory of dataset
  --n_epoch N_EPOCH     n Epoch
  --bidirectional BIDIRECTIONAL
                        Bidirectional

```
To run the codes for all mode (training and test) also
```
python nmtls-bilstm.py --data_dir data/en_121K/ --max_length 107 --mode all --n_epoch 100 --bidirectional True
```

## Transformer Model

To know how to run the code, you can execute a syntax as below:
```
$ python nmtls-transformer.py -h
usage: nmtls-transformer.py [-h] [--mode MODE] [--max_length MAX_LENGTH]
                            [--directory DIRECTORY] [--n_epoch N_EPOCH]
                            [--batch_size BATCH_SIZE]

Neural Machine Translation Link Specification (NMTLS)

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Mode: train/test
  --max_length MAX_LENGTH
                        Max length
  --directory DIRECTORY
                        Directory of dataset
  --n_epoch N_EPOCH     n Epoch
  --batch_size BATCH_SIZE
                        Batch size

```
To run the code in all mode (training and testing)
```
$ python nmtls-transformer.py --directory data/121k/ --max_length 107 --mode all --n_epoch 30 --batch_size=128
```

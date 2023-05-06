# Overview

We conducted a statistical significance test to determine whether the outperformance of a model comparison to other models is statistically significant. We conducted the test with the Wilcoxon signed-rank test, which assumes that the data are not normally distributed, in order to achieve our objective. We calculated error bars using Monte Carlo simulation because the Wilcoxon signed-rank test is a non-parametric test, meaning it does not assume a particular data distribution. In general, the results of the tests indicate that the BiLSTM model significantly outperforms the LSTM model for standard encoder-decoder architectures, regardless of  of whether the datasets are in English or German. In addition, when tested on the LIMES original LS dataset, the model with train set 2 significantly outperforms the model with train set 1 for few-shot learning models. Due to the distinction between GRU and Transformer hyperparameters, statistical tests for these models are not conducted in this test.

## Standard Encoder-Decoder Architecture
### Statistical Significance Tests For English dataset
We conducted a statistical significance test on the models that are trained on the 107K English dataset with epochs parameter set to 100.

| Model                 || Learning Rates | Length | Wilcoxon signed-rank test (p < 0.05)| Remarks|	
| ----------| -----------|----------------|--------|-------------------------------------|--------|
|  	    |            |                |        | p-value                             |        |
| BILSTM    | LSTM       | 0.1            | 107    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|
| BILSTM    | LSTM       | 1              | 107    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|
| BILSTM    | LSTM       | 0.1            | 187    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|
| BILSTM    | LSTM       | 1              | 187    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|

#### Detailed Significance Tests
##### Significance testing for model with BILSTM and LSTM on 107K dataset and parameters [learning rates=0.1, length=107]
1. Wilcoxon signed-rank statistic:  8689676.0
2. p-value:  2.5659419091461536e-167
3. Lower bound of error bars:  19321305.175
4. Upper bound of error bars:  20533512.775

![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-2.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/standard/standard-2.png)

##### Significance testing for model with BILSTM and LSTM on 107K dataset and parameters [learning rates=1, length=107].
The results:
1. Wilcoxon signed-rank statistic:  3438189.0
2. p-value:  0.0
3. Lower bound of error bars:  8564339.1375
4. Upper bound of error bars:  9363701.575

![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-1.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/standard/standard-1.png)

##### Significance testing for model with BILSTM and LSTM on 107K dataset and parameters [learning rates=0.1, length=187].
The results:
1. Wilcoxon signed-rank statistic:  9494241.5
2. p-value:  4.3134106799583765e-98
3. Lower bound of error bars:  16637084.5125
4. Upper bound of error bars:  17756734.0

![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-3.png](https://anonymous.4open.science/r/NMV-LS-CAF5/images/standard/standard-3.png)

##### Significance testing for model with BILSTM and LSTM on 107K dataset and parameters [learning rates=1, length=187].
The results:
1. Wilcoxon signed-rank statistic:  4243211.0
2. p-value:  7.076329161475455e-62
3. Lower bound of error bars:  4980348.3375
4. Upper bound of error bars:  5548023.1875

![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-4.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/standard/standard-4.png)

### Statistical Significance Tests For German dataset
We conducted a statistical significance test on the models that are trained on the 73K German dataset with epochs parameter set to 100.

| Model                 || Learning Rates | Length | Wilcoxon signed-rank test (p < 0.05)| Remarks|	
| ----------| -----------|----------------|--------|-------------------------------------|--------|
|  	    |            |                |        | p-value                             |        |
| BILSTM    | LSTM       | 0.1            | 107    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|
| BILSTM    | LSTM       |   1            | 107    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|
| BILSTM    | LSTM       | 0.1            | 187    | 0.0                                 | The model with BILSTM architecture significantly outperforms the model with LSTM Architecture|


##### Significance testing for model with BILSTM and LSTM with parameters [learning rates=0.1, length=107].
The results:
1. Wilcoxon signed-rank statistic:  5033014.5
2. p-value:  7.667409352246663e-44
3. Lower bound of error bars:  9719445.0625
4. Upper bound of error bars:  10391994.0875

![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-6.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/standard/standard-6.png)

##### Significance testing for model with BILSTM and LSTM with parameters [learning rates=1, length=107].
The results:
1. Wilcoxon signed-rank statistic:  52609.0
2. p-value:  0.0
3. Lower bound of error bars:  70180.05
4. Upper bound of error bars:  107087.15

![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-7.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/standard/standard-7.png)

##### Significance testing for model with BILSTM and LSTM with parameters [learning rates=0.1, length=187].
The results:
1. Wilcoxon signed-rank statistic:  5705914.5
2. p-value:  2.6327760933082888e-17
3. Lower bound of error bars:  9581455.1125
4. Upper bound of error bars:  10271145.3125


![https://github.com/u2018/NMV-LS/blob/main/images/standard/standard-8.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/standard/standard-8.png)

## Few-shot learning models using T5
The models are differentiated by different training datasets, as explained in the paper in the approach section. 

### Experiment Results

| Train set | Test set    | BLUE | BLEU-NLTK | METEOR | ChrF++ | TER |
|-----------|-------------|------|-----------|--------|--------|-----|
|1          |LIMES Original LS    |76.27 |0.76       |0.54    |0.87    |0.15 |
|1          |SILK LS              |34.26 |0.35       |0.26    |0.54    |0.71 |
|2          |LIMES Original LS    |77.91 |0.78       |0.54    |0.89    |0.13 |
|2          |LIMES Manipulated LS |45.76 |0.46       |0.37    |0.68    |0.55 |
|3          |LIMES Manipulated LS |63.64 |0.64       |0.43    |0.80    |0.48 |
|3          |SILK LS              |34.93 |0.35       |0.27    |0.54    |0.67 |
|4          |SILK LS              |36.58 |0.37       |0.34    |0.59    |0.62 |

### Significance Tests.

We performed the significance test to pair models with the same testing datasets. 

| Model                 || Test set    | Wilcoxon signed-rank test (p < 0.05)| Remarks|	
| ----------| -----------| ------------|-------------------------------------|--------|
| Train Set | Train set  |             | p-value                             |        |
| 2         | 1          | LIMES Original LS | 0.05                          | The model of train set 2 significanly outperforms the model of train set 1 |
| 3         | 2          | LIMES Manipulated LS | 0.5                          | The model of train set 3 does not significanly outperforms the model of train set 2 |
| 3         | 1          | SILK LS | 0.5                          | The model of train set 3 does not significanly outperforms the model of train set 1 |
| 4         | 1          | SILK LS | 0.5                          | The model of train set 4 does not significanly outperforms the model of train set 1 |
| 4         | 3          | SILK LS | 0.5                          | The model of train set 4 does not significanly outperforms the model of train set 3 |

On the LIMES original LS testing dataset, the model with train set 2 significantly outperforms the model with train set 1. Meanwhile, models tested on LIMES manipulated and SILK LS datasets do not outperform other models with different train sets significantly.

#### Detailed Significance Tests
##### Significance testing for model train set 2 and 1 on LIMES Original LS
The results:
1. Wilcoxon signed-rank statistic: 5716.0
2. p-value: 0.0498739828408391
3. Lower bound of error bars: 6879.3
4. Upper bound of error bars: 9475.95

![https://github.com/u2018/NMV-LS/blob/main/images/significance-test-1.png](https://anonymous.4open.science/r/NMV-LS-T5-B662/images/significance-test-1.png)

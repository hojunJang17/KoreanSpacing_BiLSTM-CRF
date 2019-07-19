# Korean Spacing (Bidirectional LSTM - CRF)

## Data set

* Sejong Corpus
    * Collection of corpora of modern Korean, International Korean, old Korean and oral folklore literature
    * Link: <https://ithub.korean.go.kr/user/guide/corpus/guide1.do>
* Data
    * No. of sentences in dataset : 1,040,596
    * No. of sentences in train_data : 790,852 (76%)
    * No. of sentences in val_data : 41,624 (4%)
    * No. of sentences in test_data : 208,120 (20%)

## Before training & evaluating

Data size of pickles which were saving data, vocabs were too big to upload in my git repository. 
1. Make "./dataset" directory for saving data and vocabs.
2. Run "build_dataset.py"

## Result

* epochs: 10  |  lstm hidden dim: 64  |  learning rate: 1e-3
    * Train set's f1 score : 95.80%
    * Validation set's f1 score : 95.67%
    * Test set's f1 score : 95.72%

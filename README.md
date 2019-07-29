# Korean Spacing (Bidirectional LSTM - CRF)

## Data set

* Sejong Corpus
    * Collection of corpora of modern Korean, International Korean, old Korean and oral folklore literature
    * Link: <https://ithub.korean.go.kr/user/guide/corpus/guide1.do>
    * Number of sentences in dataset (1,040,596)
        * train_data : 790,852 (76%)
        * val_data : 41,624 (4%)
        * test_data : 208,120 (20%)

## Before training & evaluating

Data size of pickles which were saving data and vocabs were too big to upload in my git repository. 
1. Make "./dataset" directory for saving data and vocabs.
2. Run "build_dataset.py"

## Result

* epochs=10,  lstm_hidden_dim=64,  learning_rate=1e-3
    * Train set's f1 score : 95.80%
    * Validation set's f1 score : 95.67%
    * Test set's f1 score : 95.72%

* epochs=20, lstm_hidden_dim=64, learning_rate=5e-4
    * Train set's f1 score : 96.22%
    * Validation set's f1 score : 96.16%
    * Test set's f1 score : 96.16%

# Data

The dataset and embeddings are not committed to the repo (size + redistribution
restrictions). Place the following files directly in this `data/` directory.

## SemEval-2017 Task 4, Subtask A (Twitter sentiment)

Tab-separated files with the format `tweet_id \t label \t text`, where label is
`positive`, `negative`, or `neutral`:

```
twitter-training-data.txt
twitter-dev-data.txt
twitter-test1.txt
twitter-test2.txt
twitter-test3.txt
```

Available from the [SemEval-2017 Task 4 organisers](https://alt.qcri.org/semeval2017/task4/)
(registration required). Task description: Rosenthal, Farra & Nakov,
*SemEval-2017 Task 4: Sentiment Analysis in Twitter* (2017).

## GloVe embeddings

Download [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) (~822 MB)
from the [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/)
and extract `glove.6B.100d.txt` here:

```bash
curl -LO https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt -d .
```

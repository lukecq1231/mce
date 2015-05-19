Tools for Word Embedding for Contrasting Meaning

We provide an implementation of the Marginal Contrast Embedding (MCE) and the test scripts. Give a thesaurus of antonym and synonym, the tool learns a vecotr for every word in the vocabulary using MCE model. The user should to specify the following:

 - desired vector dimensionality (default is 200)
 - number of negative sample (default is 100)
 - number of threads to use (default is 12)
 - the learning rate (default is 0.05)

The script run.sh trains a MCE model and test the result on "most contrasting word" questions from Graduate Record Examination(GRE). 

The code is based on word2vec (https://code.google.com/p/word2vec/).


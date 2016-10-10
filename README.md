# Tools for Word Embedding for Contrasting Meaning

Implementations of the Marginal Contrast Embedding (MCE) model presented in the paper "Revisiting word embedding for contrasting meaning" by Zhigang Chen, Wei Lin, Qian Chen, Xiaoping Chen, Si Wei, Hui Jiang and Xiaodan Zhu, ACL 2015

We provide an implementation of the Marginal Contrast Embedding (MCE) and the test scripts. Give a thesaurus of antonym and synonym, the tool learns a vecotr for every word in the vocabulary using MCE model. The user should to specify the following:

 - desired vector dimensionality (default is 200)
 - number of negative sample (default is 100)
 - number of threads to use (default is 12)
 - the learning rate (default is 0.05)

The script run.sh trains a MCE model and test the result on "most contrasting word" questions from Graduate Record Examination(GRE). 

The code is based on word2vec (https://code.google.com/p/word2vec/).

For any question or bug with the code, feel free to contact cq1231@mail.ustc.edu.cn

```latex
@InProceedings{Chen-Zhigang:2015:ACL,
  author    = {Chen, Zhigang and Lin, Wei and Chen, Qian and Chen, Xiaoping and Wei, Si and Jiang, Hui and Zhu, Xiaodan},
  title     = {Revisiting Word Embedding for Contrasting Meaning},
  booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and The 7th International Joint Conference of the Asian Federation of Natural Language Processing (ACL-IJCNLP 2015)},
  month     = {July},
  year      = {2015},
  address   = {Beijing, China},
  publisher = {ACL}
}
```
Homepage of Qian Chen, http://home.ustc.edu.cn/~cq1231/

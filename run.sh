make
time ./word2vec -train ./data/AntSyn.txt -output vectors.txt -size 200 -window 8 -negative 100 -sample 1e-4 -threads 12 -binary 0 -iter 20 -alpha 0.05 -min-count 0 -read-ant ./data/WordNet.Roget21st.Antonym.Hash.txt -read-syn ./data/WordNet.Roget21st.Synonym.Hash.txt
if [ ! -e data/testset.txt ]; then
    wget http://saifmohammad.com/WebDocs/LC-data/testset.txt -O data/testset.txt
fi
perl cal_log.pl

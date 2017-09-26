#1
you should run GetCorpus.py to make a corpus for training by convenience
eg:
python GetCorpus.py

#2
Then you can train your w2v model using TrainWord2Vec.py
eg:
python TrainWord2Vec.py

#3
you should run DecodeRes2UTF8.py to standard origin res to utf-8
eg:
decoder = DecodeRes2UTF8.DecodeRes()
decoder.decodeNeg2UTF8()
decoder.decodePos2UTF8()

#4
Finally, you can run W2V_SVM.py to make SA. We use w2v and SVM here.
During your training, you'll use CleanData.py which is about cleaning data.
If you are interested in it, you can see details by your self
eg:
python W2V_SVM.py

# coding: utf-8

# ## sRNN learning positive and negative sentences in {SP,SL}{2,4,8}

# ### import, etc

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import math
from IPython import display
import numpy as np
import glob
import re
import os
from collections import Counter
import types
import time
import sys

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
from chainer  import Variable
from chainer import serializers

#I took following 3 lines from sRNN code.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from IPython.core.display import Javascript
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import pandas as pd

#get_ipython().magic('matplotlib inline') this is commented out

#cuda part commented out
#cuda.check_cuda_available()
#cuda.get_device(0).use()
#xp = cuda.cupy
xp = np

# In[2]:

# javascript part commented out
#get_ipython().run_cell_magic('javascript', '', 'var nb = IPython.notebook;\nvar kernel = IPython.notebook.kernel;\nvar command = "NOTEBOOK_NAME = \'" +nb.notebook_name + "\'";\nkernel.execute(command);')


# ### Read PN files
# * Use 1000 positives and 1000 negatives. (made the two numbers same.)

# In[8]:
language_class_name = "S[PL]/S[PL][248]"
data ={'tr':{}, 'te1':{}, 'te2':{}}
label ={'tr':{}, 'te1':{}, 'te2':{}}
pat_name = {'tr':'Train', 'te1':'Test1', 'te2':'Test2'}
datasets = glob.glob(language_class_name+'/*')
print datasets
for dataset_name in datasets:
    for key in pat_name.keys():
        for filename in glob.glob(dataset_name+'/*'):
            if   os.path.basename(filename).find(pat_name[key]) >=0:
                print '"%s"'%filename, "is read as the", key, "set."
                pd = np.array([ l.strip().split() for l in open(filename)])
                positive = pd[:,1] == np.string_('TRUE')
                negative = pd[:,1] == np.string_('FALSE')
                if np.sum(positive) + np.sum(negative) != len(positive):
                    print "READ ERROR!"
                    break
                data[key][dataset_name] = pd[:,0]
                label[key][dataset_name]= positive

# In[9]:

def get_dataset(dataset_name):
    global data, label, pat_name
    _d_tr = data['tr'][dataset_name]
    Sigma = sorted(set([a for l in _d_tr for a in l] ))
    to_int = { a: i+1 for i,a in enumerate(Sigma) }
    _d_ret = {}
    _l_ret = {}
    for key in pat_name.keys():
        _d = data[key][dataset_name]
        _d = [ np.int32([to_int[a] for a in l]) for l in _d ]
        _d = np.array(_d)
        _l = np.int32(label[key][dataset_name])
        _d_ret[key] = _d
        _l_ret[key] = _l
    return Sigma, _d_ret, _l_ret

''' test for debug '''
print data['te1'].keys()
Sigma, _data, _label = get_dataset(data['te1'].keys()[0])
print Sigma, _data['te1'], _label['te1']

# ### Method that returns the ids  which are sorted by the length of each sentence in 'data' and are shuffled  in a blockwise manner.
# * Sort 'data' by the length of each sentence.
#  * If two sentences have almost the same length (e.g., +-1), either randomly goes first.
# * 'Data' is splited into blocks with 'bsize' length.
# * Shuffle the blocks.

# In[10]:

def get_shuffled_ids(data, bsize):
    sorted_ids = np.argsort([len(l)+np.random.uniform(-1.0,1.0) for l in data])
    blocked_sorted_ids = [ sorted_ids[i:i+bsize] for i in xrange(0,len(data),bsize) ]
    np.random.shuffle( blocked_sorted_ids )
    return blocked_sorted_ids

''' test for debug '''
temp = [[i]*(i%4+1) for i in range(0,8)]
print temp
print get_shuffled_ids(temp,2)

# ### Make a 'sentence batch' as the 2D array.

# In[11]:

def make_box(sentence_block):
    lens = [ len(s) for s in sentence_block ]
    w = max(lens)
    h = len(sentence_block)
    box = np.zeros((h,w), dtype=np.int32)
    for i, s in enumerate(sentence_block):
        box[i,-len(s):] = s
    return box.T

''' test for debug '''
shuffled_ids = get_shuffled_ids(_data['tr'],32)
print make_box(_data['tr'][shuffled_ids[0]])

print " ", ' '.join(np.array(['n','p'])[_label['tr'][shuffled_ids[0]]])

# ### Define the sRNN model

# In[12]:
"""Simple Recurrent Neural Network implementation"""
class sRNN(chainer.Chain):
    def __init__(self, num_Sigma, dim=100):
        super(sRNN, self).__init__()

        self.add_link('embed', L.EmbedID(num_Sigma, dim, ignore_label=0    ) )
        self.add_link('l1' , L.Linear ( dim, dim ) )
        self.add_link('r1' , L.Linear ( dim, dim ) )
        self.add_link('l2' , L.Linear ( dim, 2 ) )

        self.acc     = np.zeros(2)
        self.timer   = []
        self.prev_time = 0.0

    def __call__(self, box, label = None, learning = True ):
        if label is None:
            learning = False
        recurrent_h = None

        for x in box:
            z = Variable( xp.asarray(x), volatile = not learning)
            z = self.embed(z)
            if recurrent_h is None:
                recurrent_h = F.tanh(self.l1(z))
            else:
                recurrent_h = F.tanh(self.l1(z) + self.r1(recurrent_h))
        y = self.l2(recurrent_h)
        if label is None:
            y = F.softmax(y).data.get()
            return y
        else:
            t = xp.asarray(label)
            self.acc += ( F.accuracy(y, t).data*len(box), len(box) )
            loss  = F.softmax_cross_entropy(y, t)
            return loss

# ### Lerning loop

# In[14]:

dataset_names = np.array(sorted(data['tr'].keys()))
print dataset_names

# In[ ]:
import copy
for dataset_name in dataset_names:
    for dim in [10, 30, 100]:
        _ret = get_dataset(dataset_name)
        Sigma, _d, _l = _ret[0], copy.copy(_ret[1]), copy.copy(_ret[2])
        model = sRNN(len(Sigma)+1, dim=dim)
        #commented out gpu
        #model.to_gpu()

        # Set up an optimizer
        optimizer = optimizers.Adam()
        #optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer.setup(model)
        optimizer.clip_grads(1.0)

        tr = ~( np.arange(len(_d['tr']))%10 == 0 )
        data_tr, label_tr = _d['tr'][tr], _l['tr'][tr]
        _d['te0'], _l['te0'] = _d['tr'][~tr], _l['tr'][~tr]

        model.zerograds()
        bsize = 128
        language = dataset_name.split('/')[1]
        datasize = dataset_name.split('/')[-1]
        #of_basename = os.path.splitext(NOTEBOOK_NAME)[0]+"_%s_%s_v%d"%(language,datasize,dim)
        of_basename = "lang_name"+"_%s_%s_v%d"%(language,datasize,dim)
        print language, datasize, dim, of_basename
        of = open( of_basename+".log","w", buffering=1)
        #of = sys.stdout
        for epoc in range(100):
            model.acc = np.zeros(2)
            shuffled_ids = get_shuffled_ids(data_tr,bsize)
            for bid in shuffled_ids:
                box = make_box(data_tr[bid])
                loss = model(box, label_tr[bid])
                model.zerograds() # cleargrads()
                loss.backward()
                optimizer.update()
            print >>of, str(epoc).zfill(2),"training: %.4f"%(model.acc[0]/model.acc[1]),

            for te_name in ['te0','te1','te2']:
                model.acc = np.zeros(2)
                for i in xrange(0, len(_d[te_name]), bsize):
                    box = make_box( _d[te_name][i:i+bsize] )
                    loss = model(box, _l[te_name][i:i+bsize], learning = False)
                print >>of, te_name, ": %.4f"%(model.acc[0]/model.acc[1]),

            print >> of, ''

            if epoc%25 == 24:
                serializers.save_npz( of_basename +'_'+ str(epoc+1) + '.model', model)

        of.close()

# ### Generate learning curves

# In[49]:

path = "fig1_test_both/"
#logpath = "log/"
LSTMName = "lang_name"
if not os.path.exists(path): os.mkdir(path)
for language in ("SL2","SL4","SL8","SP2","SP4","SP8"):
    for datasize in ("1k","10k","100k"):
        for dim in (10,30,100):
            filename = LSTMName+"_%s_%s_v%d.log"%(language,datasize,dim)
            lines = open(filename).readlines()
            a = np.asarray([ l.strip().split() for l in lines])[:,[0,2,5,8, 11]]
            #print a
            a = a.astype(np.float)
            x1, x2, x3, x4, y =  a[:,1],a[:,2],a[:,3],a[:,4],a[:,0]
            x1a =  np.mean( [np.roll(x1, i) for i in range(-5,6) ], axis=0 )
            x2a =  np.mean( [np.roll(x2, i) for i in range(-5,6) ], axis=0 )
            x3a =  np.mean( [np.roll(x3, i) for i in range(-5,6) ], axis=0 )
            x4a =  np.mean( [np.roll(x4, i) for i in range(-5,6) ], axis=0 )
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(y[5:-5],  x2a[5:-5], c = 'yellow', lw = 20 )
            ax.plot(y[5:-5],  x1a[5:-5], c = 'lightgreen', lw = 20 )
            ax.plot(y[5:-5],  x3a[5:-5], c = 'skyblue', lw = 20 )
            ax.plot(y[5:-5],  x4a[5:-5], c = 'lightpink', lw = 20 )
            ax.plot(y, x2, c = 'orange', label="held-out data")
            ax.plot(y, x1, c = 'green', label="traing data")
            ax.plot(y, x3, c = 'blue', label="test1 data")
            ax.plot(y, x4, c = 'deeppink', label="test2 data")
            filename = path+LSTMName+"_%s_%s_v%d.png"%(language,datasize,dim)

            fig.suptitle('Learning Curve for %s, %s and v%d '%(language,datasize, dim), fontsize=20)
            ax.set_ylabel("Correct Classification Rate", fontsize=18)
            ax.set_xlabel("Epochs", fontsize=18)
            ax.set_ylim([0.5, 1.0])
            plt.legend(fontsize=18)
            plt.tick_params(labelsize=18)
            plt.savefig(filename)
            plt.show()

# subreg_deeplearning
Subregular Complexity and Deep Learning

The aim of this project is to show that the judicial use of formal language theory and grammatical inference are invaluable tools in understanding how deep neural networks can and cannot represent and learn long-term dependencies in temporal sequences. We conducted learning experiments with two types of Recurrent Neural Networks (RNNs) on six formal languages drawn from the Strictly Local (SL) and Strictly Piecewise (SP) classes. The networks were Simple RNNs (s-RNNs) and Long Short-Term Memory RNNs (LSTMs) of varying sizes. The SL and SP classes are among the simplest in a mathematically well-understood hierarchy of subregular classes. They encode local and long-term dependencies, respectively. The grammatical inference algorithm Regular Positive and Negative Inference (RPNI) provided a baseline. According to earlier research, the LSTM architecture should be capable of learning long-term dependencies and should outperform s-RNNs. The results of these experiments challenge this narrative. First, the LSTMs’ performance was generally worse in the SP experiments than in the SL ones. Second, the s-RNNs out-performed the LSTMs on the most complex SP experiment and performed comparably to them on the others.

Here, on this Github page, we want to share our datasets and the RNNs codes so that anyone interested in this kind of research can replicate and go beyond our research. Following this aim, we try to explain how we generated our train and test datasets, what kinds of softwares and tool kits we have used, and what are the results of this research.

Generating the Train and Test Datasets

In this study, six formal target languages (three from the SL class and three from the SP class) were defined in order for training and testing purposes. These formal languages are drawn from well-understood subclasses of the regular languages which form a complexity hierarchy.
For each language, three training sets were prepared, and for each training set two test sets were prepared, for a total of 36 test sets. These target languages were implemented as finite-state machines using foma (https://code.google.com/archive/p/foma/), a publicy available, open-source platform (Hulden, 2009). In the Experimental Data folder, the code for generating training and test data can be found. We also shared the actual training and test datasets we had generated, in the same folder.

Neural Network Architectures and the RPNI

For the LSTMs and s-RNNs, we constructed simple networks to test the capability of the networks themselves. We implemented the RNNs with Chainer (http://chainer.org), and the RPNI was implemented using Matlab and the gi-toolbox https://code.google.com/archive/p/gitoolbox/ (Akram et al., 2010). In the Learning Algorithms folder, the code for LSTM, sRNN with ADAM, sRNN with SGD, and the codes we used to run RPNI can be found.

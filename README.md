# subreg_deeplearning
Subregular Complexity and Deep Learning

The aim of this project is to show that the judicial use of formal language theory and grammatical inference are invaluable tools in understanding how deep neural networks can and cannot represent and learn long-term dependencies in temporal sequences.

Learning experiments were conducted with two types of Recurrent Neural Networks (RNNs) on six formal languages drawn from the Strictly Local (SL) and Strictly Piecewise (SP) classes. The networks were Simple RNNs (s-RNNs) and Long Short-Term Memory RNNs (LSTMs) of varying sizes. The SL and SP classes are among the simplest in a mathematically well-understood hierarchy of subregular classes. They encode local and long-term dependencies, respectively. The grammatical inference algorithm Regular Positive and Negative Inference (RPNI) provided a baseline.

Here we share;
* code for generating training and test data
* the actual training and test data we generated (I guess this uses a lot of space?)
* the code we used to run LSTMs (analysis 3)
* the code we used to run sRNNs
* the code for sRNN with SGD
* the code we used to run RPNI
* result summaries

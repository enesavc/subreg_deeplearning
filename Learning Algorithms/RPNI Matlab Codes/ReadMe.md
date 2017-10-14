Description

Please copy and paste RPNI.m and RPNI_OUTPUT.m files into the folder that was created with gi-toolbox https://code.google.com/archive/
p/gitoolbox (Akram et al., 2010).

RPNI starts from creating PTA according to positive samples. Initialize RED state set and
BLUE state set as following:
	(i). set start state as RED state and add it to RED set.
	(ii). add all the successors of start state to BLUE set.
After initialization, RPNI starts to extract each BLUE state and merge it for each RED state.

After each Merge, we examine the new merged DFA if it accepts any negative sample,
	(i).if it does, abandon such DFA and promote currently manipulated BLUE state and continue another Merge.
	(ii).if it does not accpet any negative sample, update DFA and add all the red states' successors which are not BLUE states to BLUE set. Until there's no BLUE state for Merge.

Data Input Format

In order to make your program runnable, you should follow the format of input sample. * RPNI require both positive and negative sample,

The input data should look like this:

8 2

1 3 a a a

1 4 a a b a

1 3 b b a

1 5 b b a b a

0 1 a

0 2 b b

0 3 a a b

0 3 a b a


The header line indicates the total number of samples (here is 8) and the number of different alphabets (here is 2).
Then each line represents a sample, the first bit of which indicate if it's positive sample (1) or negative (0), 
the second bit indicates the length of the sample and following is the real sample string.

A Simple Example

After downloading the toolbox, extract to any place you want and switch to the root folder in Matlab.
Assume that you have a sample file "input.txt" formated as above for RPNI, open the command line window in Matlab and type:

[training, group, positive, negative] = ReadSamples('Data/input.txt');

dfa = RPNI(positive, negative);

Now you got a DFA learned from sample "input.txt".

Our RPNI_OUTPUT function translates the output of RPNI algorithm to a foma readable .att file.
Then, we load the .att file to foma by opening the foma. When the dfa is loaded to the stack, just trim the machine in order to get the minimized dfa.
Finally when we run the foma test code, which tests the minimized dfa on our target test sets (test1 and test2), accuracy of the minimized dfa that is the output of RPNI can be shown on target test sets.

% [Authors]: Hasan Ibne Akram, Huang Xiao
% [Institute]: Munich University of Technology
% [Web]: http://code.google.com/p/gitoolbox/
% [Emails]: hasan.akram@sec.in.tum.de, huang.xiao@mytum.de
% Copyright ? 2010
% 
% ****** This is a beta version ******
% [DISCLAIMER OF WARRANTY]
% This source code is provided "as is" and without warranties
% as to performance or merchantability. The author and/or 
% distributors of this source code may have made statements 
% about this source code. Any such statements do not constitute 
% warranties and shall not be relied on by the user in deciding
% whether to use this source code.
% 
% This source code is provided without any express or implied
% warranties whatsoever. Because of the diversity of conditions
% and hardware under which this source code may be used, no
% warranty of fitness for a particular purpose is offered. The 
% user is advised to test the source code thoroughly before relying
% on it. The user must assume the entire risk of using the source code.
% 
% -------------------------------------------------
% [Description]
% The RPNI algorithm
% RPNI starts from creating PTA according to positive samples. Initialize
% RED state set and BLUE state set as following:
%   1. set start state as RED state and add it to RED set.
%   2. add all the successors of start state to BLUE set.
% After initialization, RPNI starts to extract each BLUE state and merge it
% for each RED state. After each Merge, we examine the new merged DFA if it
% accepts any negative sample, 
%   1. if it does, abandon such DFA and promote currently manipulated BLUE 
%      state and continue another Merge.
%   2. if it does not accpet any negative sample, update DFA and add all the
%      red states' successors which are not BLUE states to BLUE set.
% Until there's no BLUE state for Merge.
% see also RPNI_MERGE, RPNI_PROMOTE, RPNI_FOLD, RPNI_COMPATIBLE, BUILD_PTA,
% IsStringAccepted, ReadSamples

function dfa = RPNI(positive, negative)
    max_recursion_depth(2560)
    printf('Bulding PTA....');
    dfa = BUILD_PTA(positive);
    
    save('pta.mat', 'dfa');
   
    % adding the red states
    dfa.RED = [dfa.RED, dfa.FiniteSetOfStates(1)];
    
    % adding the blue states
    for i = 1:length(dfa.Alphabets)
        temp_blue = GetTransitionState(dfa, dfa.FiniteSetOfStates(1), dfa.Alphabets(i));
        if(temp_blue~=0) % 0 means no transition            
            dfa.BLUE = [dfa.BLUE, temp_blue];
        end
    end
    
    printf('Running RPNI on PTA....');
    while (~isempty(dfa.BLUE))
        % sorting blue in order to choose a blue state in lexical order
        % dfa.BLUE = sort(dfa.BLUE, 2, 'descend');
        dfa.BLUE = sort(dfa.BLUE);
        q_b = dfa.BLUE(1);
        
        % deleting the first element of BLUE
        dfa.BLUE = [dfa.BLUE(2:length(dfa.BLUE))];
        promote = 1;
        for i = 1:length(dfa.RED)
           dfa_merged = RPNI_MERGE(dfa, dfa.RED(i), q_b);
           if(RPNI_COMPATIBLE(dfa_merged, negative))             
               dfa = dfa_merged;
               dfa
               printf('merge accepted')
               dfa.RED(i)
               q_b
               dfa = AddNewBlueStates(dfa);
               promote = 0;
               break;
           end    
        end
        if (promote==1)
            dfa = RPNI_PROMOTE(q_b, dfa);            
        end 
        
    end
    RPNI_OUTPUT(dfa)
end

%[training, group, positive, negative] = ReadSamples('NSData/X.txt'); 
%dfa = RPNI(positive, negative);
function RPNI_OUTPUT(dfa)
g = dfa;

M=g.TransitionMatrix;
FS=dfa.FinalAcceptStates;

Alphabets = {'a','b','c','d'};
N=length(Alphabets);
B=zeros(size(M,2)*size(M,1),2);  
for k=1:size(M,1)
  for i=1:size(M,2)
  B(i+size(M,2)*(k-1),:)=[k M(k,i)];
  end
end

B(B==1)=-1; B(B==0)=1; B(B==-1)=0;


fid = fopen('SLP','wt');
for k=1:size(M,2)*size(M,1)
   fprintf(fid,'%d\t%d\t%s\t%s\n',B(k,:),Alphabets{mod(k+N-1,N)+1},Alphabets{mod(k+N-1,N)+1});
end
for m=1:length(FS)
  fprintf(fid,'%d \n',FS(m));  
end

fclose(fid);
end
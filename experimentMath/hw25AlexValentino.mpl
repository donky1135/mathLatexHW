#C25.txt: Zero-Knowledge Proofs. April 18, 2024
Help:=proc(): print(`LD(p), RG(n,p), RHG(n,p), ZKP1(n,G,pi,Opt)`): end:
with(combinat):
#LD(p): inputs a pos. rational number from 0 to 1 and  outputs true with prob. p
LD:=proc(p) local i:i:=rand(1..denom(p))(): if i<=numer(p) then true:else false:fi:end:

#RG(n,p): inputs a pos. integer n and outputs a simple graph on n vertices where the
#prob of an edge is p (independetly)
RG:=proc(n,p) local i,j,G:
G:={}:
for i from 1 to n do
 for j from i+1 to n do
  if LD(p) then
    G:=G union {{i,j}}:
  fi:
 od:
od:
G:  
end:


#RHG(n,p): inputs a pos. integer n and outputs a simple  HAMILTONIAN graph on n vertices 
#where the
#prob of an edge is p (independetly) together with the Hamiltonian path
#The output is a pair [G,permutation] where the permutation of {1,...,n}
#is the Hamiltonian cycle that you know and you don't want anyone else to know
#BUT you do want to convince them that you DO know
RHG:=proc(n,p) local i,j,G,pi:
pi:=randperm(n):
G:={seq({pi[i],pi[i+1]},i=1..n-1), {pi[n],pi[1]}}:

for i from 1 to n do
 for j from i+1 to n do
  if LD(p) then
    G:=G union {{i,j}}:
  fi:
 od:
od:
[G, pi]:  
end:

#ZKP1(n,G,pi,Opt): Does ONE ROUND of the Blum protocol. Inputs a graph n,G,
#a Hamiltonian pi, and Opt=1 or 2 Opt=1 you show all the n+binomial(n,2) boxes
#Opt2, you show the contents of the n boxes correponsing to the mapping of
#your Hamiltonian
ZKP1:=proc(n,G,pi,Opt) local B1,B2,i,j,sig:
sig:=randperm(n):
for i from 1 to n do
 B1[i]:=sig[i]:
od:

for i from 1 to n do
 for j from i+1 to n do
   if member ({i,j},G) then
     B2[{sig[i],sig[j]}]:=1:
   else
    B2[{sig[i],sig[j]}]:=0:
   fi:
od:
od:

if Opt=1 then
 [op(B1),op(B2)]:
else
 {seq(B2[{sig[pi[i]],sig[pi[i+1]]}],i=1..n-1),B2[{sig[pi[n]],sig[pi[1]]}]}:
fi:
end:







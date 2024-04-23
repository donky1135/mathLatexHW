LD := proc(p) local i: i := rand(1..denom(p))(): if i <= numer(p) then true: else false: fi: end:

#takes in a number of vertices and a probability and outputs a graph based on randomly connecting nodes w/ prob p
RG := proc(n,p) local i,j, G: 

G := {}:
for i from 1 to n do 
    for j from i+1 to n do  
        if LD(p) then 
            G := G union {{i,j}}:
        fi:
    od:
od:
G:
end:

#makes a random graph w/ a premade hamiltonian path, outputs the random graph and the permutation 

RHG := proc(n,p) local i,j, G, pi: 
pi := randperm(n):
G := {seq({pi[i],pi[i+1]}, i=1..n-1), {pi[n], pi[1]}}
G := {}:
for i from 1 to n do 
    for j from i+1 to n do  
        if LD(p) then 
            G := G union {{i,j}}:
        fi:
    od:
od:
[G,pi]:
end:
#ZKP1(n,G,pu,Opt): Does ONE ROUND of the Blum protocol.  Inputs a graph n,G,
#a Hamiltonian permutation pi, and Opt=1 or 1, Opt=1 you show all of hte n + nC2 boxes
#Opt2, you show the contents of the n boxes corresponding to the mapping of your hamiltonian
ZKP1 := proc(G,pi,Opt):

#C24.txt: April 15, 2024
Help:=proc(): print(`MakeRSAkey1(K) , MakeRSAkey(K,M) `): 
print(`AliceToBob(M,Nbob,eBob,Nalice,dAlice,H)`):
print(`BobReadAlice(MS,Nbob,dBob,Nalice,eAlice,H)`):
end:
with(numtheory):
#MakeRSAkey1(K): makes a triple [N,e,d] where N,e, are public and d private
#one try
MakeRSAkey1:=proc(K) local P,Q,ra,N,e,d:
ra:=rand(10^K..10^(K+1)):
P:=nextprime(ra()):
Q:=nextprime(ra()):
if P=Q then
 RETURN(FAIL):
fi:
N:=P*Q:
e:=rand(1..(P-1)*(Q-1))():
if gcd(e,(P-1)*(Q-1))<>1 then
  RETURN(FAIL):
fi:
d:=e&^(-1) mod (P-1)*(Q-1):
[N,e,d]:
end:
#MakeRSkey(K,M): tries to make an RSA key by trying M times
MakeRSAkey:=proc(K,M)  local m,rsa:

for m from 1 to M do
rsa:=MakeRSAkey1(K):
 if rsa<>FAIL then
   RETURN(rsa):
 fi:
od:
FAIL:
end:

#Hash function: x&^3 mod H
#AliceToBob(M,Nbob,eBob,Nalice,dAlice,H):|
#Inputs Alice's message M, the [Nbob,eBob] public part of
#Bob's RSA key, the Nalice (public) and dAlice (private) of Alice's key
#and H (public Hash function x&^3 mod H
#output a pair the encrypted message, followed by the much shorter
#signature
AliceToBob:=proc(M,Nbob,eBob,Nalice,dAlice,H) local x,M1,S:
M1:=M&^eBob mod Nbob:
x:=M&^3 mod H:
S:=x&^dAlice mod Nalice:
[M1,S]:
end:

#BobReadAlice(MS,Nbob,dBob,Nalice,eAlice,H)
#inputs a signed message MS=[M1,S] sent from Alice,
#Nbob,dBob,Nalice, eAlice, and H outputs
#the deciphered message from Alice and also checks
#that it came from Alice and not from a bad guy
#pretending to be Alice
BobReadAlice:=proc(MS,Nbob,dBob,Nalice,eAlice,H) local M1,S,M,X,X1:
M1:=MS[1]:
S:=MS[2]:
M:=M1&^dBob mod Nbob:
X:=M&^3 mod H:
X1:=S&^eAlice mod Nalice:
if X<>X1 then
 print(`You are not Alice, I will call the police`):
fi:
M:
end:



#C18.txt, March 25, 2024
Help:=proc(): print(` RW(k), BinToIn(L) , InToBin(n,k), BinFun(F,L) ,  Feistel(LR,F)`): 
print(`revFeistel(LR,F)`):
end:

#RW(k): a random binary word of length k
RW:=proc(k) local ra,i:
ra:=rand(0..1):
[seq(ra(),i=1..k)]:
end:

#BinToIn(L): inputs a binary word L and outputs the integer n whose binary representation
#it is
BinToIn:=proc(L) local k,i:
k:=nops(L):
#[1,0,1]: 1*2^2+0*2^1+1*2^0
add(L[i]*2^(k-i)  , i=1..k):
end:

#InToBin(n,k): inputs a non-neg.  integer n<2^k into its binary representation with
#0's in front if neccessary so that it is of length k
InToBin:=proc(n,k) local i:
#option remember:
if ( n>=2^k or n<0)  or not (type(n,integer)) then
   RETURN(FAIL):
fi:

if k=1 then
 if n=0 then
    RETURN([0]):
 else
   RETURN([1]):
 fi:
fi:

if n<2^(k-1) then
  RETURN([0,op(InToBin(n,k-1))]):
else
RETURN([1,op(InToBin(n-2^(k-1),k-1))]):
fi:

end:

#BinFun(F,L): conmverts a function on the integers (mod 2^k) to a function
#on binary words
BinFun:=proc(F,L) local k:
k:=nops(L):
 InToBin(F(BinToIn(L)) mod 2^k,k):
end: 

#Feistel(LR,F): The Feistel transform that takes [L,R]->[R+F(L) mod 2, L]
#For example Feistel([1,1,0,1],n->n^5+n);
Feistel:=proc(LR,F) local k,L,R:
k:=nops(LR):
 if k mod 2<>0 then
   RETURN(FAIL):
 fi:
L:=[op(1..k/2,LR)]:
R:=[op(k/2+1..k,LR)]:
[op(R+ BinFun(F,L) mod 2)   ,  op(L)  ]:
end:


#revFeistel(LR,F): The reverse Feistel transform that takes [L,R]->[R,L+F(R)]
#For example Feistel([1,1,0,1],n->n^5+n);
revFeistel:=proc(LR,F) local k,L,R:
k:=nops(LR):
 if k mod 2<>0 then
   RETURN(FAIL):
 fi:
L:=[op(1..k/2,LR)]:
R:=[op(k/2+1..k,LR)]:
[op(R),      op(L+ BinFun(F,R ) mod 2)    ]:
end:



FindRP := proc(d):
    nextprime(rand(10^(d-1)..10^d-1)()):
end:

isPrim := proc(P,g) local i: #Is g primitive mod P?
    for i from 1 to P-2 do
        if g&^i mod P = 1 then 
            RETURN(false):
        fi:
    od:
    RETURN(true):
end:

FindPrim := proc(P) local g: #given a prime P, find a primitive root g
    for g from 2 to P-2 do
        if isPrim(P,g) then 
            RETURN(g):
        fi:
    od:

    FAIL:
end:

# FindSGP(d) it finds a Sophie Germain prime with d digits

FindSGP := proc(d,K) local Q,i,ra:

    ra := rand(10^(d-1)..10^d-1):

    for i from 1 to K do
        Q := nextprime(ra()):
        if isprime(2*Q + 1) then 
            RETURN(2*Q+1):
        fi:
    od:

end:

FindPrimiSGP := proc(P) local Q:
    Q := (P-1)/2:

    if not (isprime(P) and isprime(Q)) then 
        RETURN(FAIL):
    fi:

    a := rand(2..P-2)():
    if a^&Q mod P = P-1 then 
        RETURN(a):
    else 
        RETURN(P-a)
    fi:
end:

AlicceToBob := proc(Pg) local a:
    a := rand(3..P-3)():
    [a,g&^a mod P]:
end:

BobToAlice := proc(P,g) local b:
    B;= rand(3..P-3)():
    [b,g&^b mod P]:
end:
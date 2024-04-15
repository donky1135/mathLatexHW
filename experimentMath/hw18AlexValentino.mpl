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
    print(n):
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



#hw begins:

#1

MakeCubic := proc(L,n):
    BinToIn([op(1..5,L)])+BinToIn([op(6..10,L)])*n+BinToIn([op(11..15,L)])*n^2+BinToIn([op(16..20,L)])*n^3:
end:

#2 

EncodeDES := proc(M,K,r) local i, cubics,n, E:
    cubics := [seq(MakeCubic([op((1+20*i)..(20+20*i),K)],n),i=0..(r-1))]:
    print(cubics);
    E := M:
    for i from 1 to r do
        E := Feistel(E,n->cubics[i]):
    od:
    E:
end:

#3

DecodeDES := proc(M,K,r) local i, cubics, n, D:
    cubics := [seq(MakeCubic(op((1+20*i)..(20+20*i),K),n),i=0..(r-1))]:
    D := M:
    for i from 1 to r do
        D := revFeistel(D,n->cubics[r-i+1]):
    od:
    D:
end:
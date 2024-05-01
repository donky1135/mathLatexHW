Help := proc(): print(`Fqn(q,n), RV(q,n), LD(q,u,v)`) end:

with(LinearAlgebra):

#Alphabet {0,1,...q-1}, Fqn(q,n): {0,1,...,q-1}^n
FqnP:=proc(q,n) local S,a,v:
option remember:
if n=0 then
 RETURN({[]}):
fi:

S:=FqnP(q,n-1):

{seq(seq([op(v),a- (q-1)/2],a=0..q-1), v in S)}:

end:

Fqn:=proc(q,n) local S,a,v:
option remember:
if n=0 then
 RETURN({[]}):
fi:

S:=Fqn(q,n-1):

{seq(seq([op(v),a],a=0..q-1), v in S)}:

end:

LtoCP:=proc(q,M) local n,k,C,c,i,M1:
option remember:
k:=nops(M):
n:=nops(M[1]):
if k=1 then
 RETURN({seq(i*M[1] mod q,i=-1*(q-1)/2..(q-1)/2) }):
fi:
M1:=M[1..k-1]:
C:=LtoCP(q,M1):
{seq(seq(c+i*M[k] mod q,i=-1*(q-1)/2..(q-1)/2),c in C)}:
end:

LtoC:=proc(q,M) local n,k,C,c,i,M1:
option remember:
k:=nops(M):
n:=nops(M[1]):
if k=1 then
 RETURN({seq(i*M[1] mod q,i=0..q-1) }):
fi:
M1:=M[1..k-1]:
C:=LtoC(q,M1):
{seq(seq(c+i*M[k] mod q,i=0..q-1),c in C)}:
end:

#RV(q,n): A random word of length n in {0,1,..,q-1}
RV:=proc(q,n) local ra,i:
ra:=rand(0..q-1):
[seq( ra(), i=1..n)]:
end:

#q-Lee distance between two code words u,v
LD := proc(q,u,v) local n,i:
    n := nops(u):
    if nops(v) <> n then 
        RETURN(FAIL):
    fi:
    add(seq(min(abs(u[i] - v[i]),q-abs(u[i] - v[i])) mod q, i=1..n)) mod q:
end:

dot := proc(q,u,v) local n,i:
    n := nops(u):
    if nops(v) <> n then 
        RETURN(FAIL):
    fi:
    add(seq(u[i]*v[i], i=1..n)) mod q:
end:

parCheck := proc(q,u,setWords):
    result := {}:
    for word in setWords do
        if dot(q,u,word) = 0 then 
            result := result union {word}:
        fi:
    od:
    result:
end:

weightCheck := proc(q,setWords):
    result := {}:
    n := nops(setWords[1]):
    for word in setWords do
        if LD(q,[0$n],word) = 3 then 
            result := result union {word}:
        fi:
    od:
    result:
end:

OZ := proc(n):
    id := [seq([0$(i-1), 1, 0$(n-i)], i=1..n)]:
    idMore := []:
    negPerms := Fqn(2,n):
    for negs in negPerms do
        sample := []:
        for i from 1 to n do
            sample := [op(sample), (-1)^(negs[i])*id[i]]:
        od:
        idMore := [op(idMore), sample]:
    od:
    {seq(op(permute(ids)), ids in idMore)}:
end:

groupOp := proc(p,n):
    OZn := OZ(n):
    leeP3 := weightCheck(p,parCheck(p,[seq(i, i=1..n)], FqnP(p,n))):
    codeCopies := {}:
    for perm in OZn do
        temp := {}:
        for word in leeP3 do
            temp := {convert(Multiply(Matrix(perm), Vector(word)), list)} union temp:
        od:
        codeCopies := codeCopies union {temp}:
    od:
    codeCopies:
end:

#groupOp

genMatrix := proc(n):
    mat := []:
    for i from 1 to n do
        if i mod 2 = 0 then 
            mat := [op(mat), [i, 0$(n - 2 - (i)/2), 1,1, 0$(-1+i/2)]]
        else
            mat := [op(mat), [i, 0$(n - i/2 - 1), 2, 0$((i-1)/2)]]:
        fi:
    od:
    mat:
end:
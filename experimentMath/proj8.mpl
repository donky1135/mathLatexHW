Help := proc(): print(`Fqn(q,n), RV(q,n), LD(q,u,v)`) end:

#Alphabet {0,1,...q-1}, Fqn(q,n): {0,1,...,q-1}^n
Fqn:=proc(q,n) local S,a,v:
option remember:
if n=0 then
 RETURN({[]}):
fi:

S:=Fqn(q,n-1):

{seq(seq([op(v),a],a=0..q-1), v in S)}:

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
    add(seq(min(abs(u[i] - v[i]),q-abs(u[i] - v[i])) mod q, i=1..n)):
end:

dot := proc(q,u,v) local n,i:
    n := nops(u):
    if nops(v) <> n then 
        RETURN(FAIL):
    fi:
    add(seq(u[i]*v[i], i=1..n)) mod q:
end:


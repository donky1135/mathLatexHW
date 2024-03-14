#March 7, 2024 C15.txt
Help:=proc(): print(`S(x,y), aS(a), OurPF(R,t),  aSr(a) , `):end:

#S(x,y): maps a pair of lists of numbers of the same length, n, say,
#and outputs a list of numbers of length 2*n
#[x[1]+....+x[n],x[1]*y[1]+...+x[n]*y[n],...., x[1]*y[1]^(2*n-1)+.. x[n]*y[n]^(2*n-1)]
S:=proc(x,y) local n,i,r:
if not (type(x,list) and type(y,list) and nops(x)=nops(y)) then
 RETURN(FAIL):
fi:
n:=nops(x):
[ seq(add(x[i]*y[i]^r,i=1..n),r=0..2*n-1)]:
end:

#aS(s): reverse S(x,y) using Maple's solve command.
#inputs a list of length 2*n and outputs lists x,y of
#length n, such that S(x,y)=a
aS:=proc(a) local n,x,y,X,Y,i,eq,var,r:
if not( type(a,list) and nops(a) mod 2=0) then
 RETURN(FAIL):
fi:
n:=nops(a)/2:
eq:={seq(add(x[i]*y[i]^r,i=1..n)=a[r+1],r=0..2*n-1)}:
var:={seq(x[i],i=1..n),seq(y[i],i=1..n)}:
var:=solve(eq,var)[1]:
[[seq(subs(var,x[i]),i=1..n)],[seq(subs(var,y[i]),i=1..n)]]:
end:


#corrected after class
#OurPF(R,t): inputs a rational function of t outputs the
#partial fraction decomopition over the complex numbers of the form
#[a1,r1],..., [an,rn] where 1/r1,..., 1/rn are the complex roots of the
#bottom
OurPF:=proc(R,t) local n,Top,Bot,zer,Y,i:
Top:=numer(R):
Bot:=denom(R):
zer:=[solve(Bot,t)]:
n:=nops(zer):
Y:=sort([seq(1/zer[i],i=1..nops(zer))]):
[[seq(-Y[i]*subs(t=1/Y[i],Top)/subs(t=1/Y[i],diff(Bot,t)),i=1..n)],Y]:
end:


#aSr(a): solves the system of Ramanujan using his method with partial fractions
#outputs the rational function whose partial fraction would solve the system
#modified after class
aSr:=proc(a) local n,A,B,i,Top,Bot,f,eq,var,t:
if not( type(a,list) and nops(a) mod 2=0) then
 RETURN(FAIL):
fi:
n:=nops(a)/2:

Top:=add(A[i+1]*t^i,i=0..n-1):

Bot:=1+add(B[i]*t^i,i=1..n):

#Top/Bot=add(a[i]*t^(i-1),i=1..2*n):

f:=expand(Top-Bot*add(a[i]*t^(i-1),i=1..2*n)):
eq:={seq(coeff(f,t,i),i=0..2*n-1)}:
var:={seq(A[i],i=1..n),seq(B[i],i=1..n)}:
var:=solve(eq,var):
OurPF(normal(subs(var,Top)/subs(var,Bot)),t):
end:




#beginning of homework 15
# 1:
# Observe that (1-y_i*t)*A(t)/B(t) = C_i + (1-y_i*t)\sum_{j = 1,j \neq i}^n \frac{C_i}{(1-y_j*t)}
# Therefore if one takes the limit as t -> 1/y_i for (1-y_i*t)*A(t)/B(t) = C_i
# as the sum (1-y_i*t)\sum_{j = 1,j \neq i}^n \frac{C_i}{(1-y_j*t)} would go to zero,
# all roots of B(t) are distinct.  Furthermore, if one is to take the original 
# expression and compute the limit as t -> 1/y_i one would get the indeterminate form 
# 0/0, therefore we must apply l'Hoptial's rule, yielding 
# the fraction (A'(t)(1-y_i*t) - y_i * A(t))/B'(t).  
# Furthermore since the roots of B(t) are distinct then 
# it is implied that B'(1/y_i) \neq 0.  Therefore 
# we can solve the limit by substitution yielding 
# C_i = -y_i * A(1/y_i)/B'(1/y_i)

# 2: 

a := [2,3,16,31,103,235,674,1669,4526,11595]:
attempts := [[-3/5,-1],[(18 + sqrt(5))/10,(3 + sqrt(5))/2],[(18 - sqrt(5))/10,(3 - sqrt(5))/2],[-(8+sqrt(5))/(2*sqrt(5)), (sqrt(5)-1)/2],[(8-sqrt(5))/(2*sqrt(5)),-(sqrt(5)+1)/2]]:
result := aSr(a):
eps := 0.0001:
for i from 1 to nops(a)/2 do
    for j from 1 to nops(a)/2 do
        if evalb(evalf(abs(attempts[i][1] - result[1][j])) < eps and 
        abs(evalf(attempts[i][2] - result[2][j])) < eps) then 
            print(`aSr found : `, attempts[i]);
        fi:
    od
od:

#the corresponding output is: 
(*
aSr found : , [-3/5, -1]

                                                              1/2         1/2
                                                             5           5
                                        aSr found : , [9/5 + ----, 3/2 + ----]
                                                              10          2

                                                              1/2         1/2
                                                             5           5
                                        aSr found : , [9/5 - ----, 3/2 - ----]
                                                              10          2

                                                            1/2   1/2   1/2
                                                      (8 + 5   ) 5     5
                                     aSr found : , [- ---------------, ---- - 1/2]
                                                            10          2

                                                          1/2   1/2     1/2
                                                    (8 - 5   ) 5       5
                                     aSr found : , [---------------, - ---- - 1/2]
                                                          10            2
*)
#therefore the paper's example has been verified.

#partial 3:
Sq:=proc(x,y,q) local n,i,r:
if not (type(x,list) and type(y,list) and nops(x)=nops(y)) then
 RETURN(FAIL):
fi:
n:=nops(x):
[ seq(add(x[i]*y[i]^r,i=1..n),r=0..2*n-1)] mod q:
end:

qOurPF:=proc(R,t,q) local n,Top,Bot,zer,Y,i:
Top:=numer(R):
Bot:=denom(R):
zer:=[]:
for i from 0 to q-1 do 
    if subs(t=i,Bot) mod q = 0 then
        zer := [nops(zero), q]:
    fi:
od:


n:=nops(zer):
Y:=sort([seq(zer[i]&^(-1) mod q,i=1..nops(zer))]):
[[seq(-Y[i]*subs(t=Y[i]&^(-1) mod q,Top)*(subs(t=Y[i]&^(-1) mod q,diff(Bot,t)),i=1..n))&^(-1) mod q],Y]:
end:
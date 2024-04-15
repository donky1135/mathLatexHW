
#C20.txt: April 1, 2024
Help:=proc(): print(`RP(a,b), DoesLie(a,b,P), AddE(a,b,P,Q) `):end:
#Elliptic curve: y^2=x^3+a*x+b 
RP:=proc(a,b) local x: x:=rand(0..200)()/100: [x,sqrt(x^3+a*x+b)]:end:

DoesLie:=proc(a,b,P): P[2]^2-P[1]^3-a*P[1]-b:end:

AddE:=proc(a,b,P,Q) local x,y,s,eq,sol,R,xR:
 s:=(P[2]-Q[2])/(P[1]-Q[1]):
  y:=P[2]+(x-P[1])*s:
sol:={solve(y^2-x^3-a*x-b,x)};


if not (nops(sol)=3 and member(P[1],sol) and member(Q[1],sol)) then
  print(`Something bad happened`):
  RETURN(FAIL):
fi:

sol:=sol minus {P[1],Q[1]}:
xR:=sol[1]:
[xR,-sqrt(xR^3+a*xR+b)]:

end: 

#beginning of HW 20
#1
Diag := proc(a,b):

    plot1 := plot(sqrt(x^3 + a*x + b), x = 0 .. 2, color = red):
    plot1 := plot(-1*sqrt(x^3 + a*x + b), x = 0 .. 2, color = red):
    A := RP(a,b):
    B := RP(a,b):
    AnB := AddE(a,b,pt1,pt2):
    C := [AnB[1],-1*AnB[2]]:
    plot3 := plot([A,B,AnB,C]):
    display(plot1,plot2,plot3);
end:

#2 
IntSol := proc(a,b,K):
    S := {}:
    for x from -K to K do
        if type(sqrt(x^3 + a*x + b),integer) then
            S := S union {[x,sqrt(x^3 + a*x + b)],[x,-1*sqrt(x^3 + a*x + b)]}:
        fi:
    od:
    S:
end:

#3
SolChain := proc(a,b,S1,S2,K):
    L := [S1,S2]:

    for i from 1 to K do
        L := [op(L),AddE(a,b,L[-2],L[-1])]:
    od:

    L:
end:
#this method can attain an infinite number of solutions, as the successive points could potentially 
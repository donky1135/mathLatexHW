<<<<<<< Updated upstream
#C21.txt, 4/4/24

Help:=proc(): print(`   AE(a,b,P,Q), EC(a,b,p), AEp(a,b,P,Q,p), Mul(g,A,a,b,p), Sqrt(n,p,K) `):end:

#old code

Help20:=proc(): print(`RP(a,b), DoesLie(a,b,P), AddE(a,b,P,Q) `):end:
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

###end of C20.txt

AE:=proc(a,b,P,Q) local x,y,s,R,xR,yR:
if P=infinity then
  RETURN(Q):
fi:
if Q=infinity then
  RETURN(P):
fi:

if P[1]<>Q[1] then
 s:=(P[2]-Q[2])/(P[1]-Q[1]):
 xR:=s^2-P[1]-Q[1]:
 yR:=s*(P[1]-xR)-P[2]:
 RETURN([xR,yR]):
 else
  if P[1]=Q[1] and P[2]<>Q[2] then
   RETURN(infinity):
   elif P[1]=Q[1] and P[2]=0 and Q[2]=0 then
    RETURN(infinity):
    elif P=Q then
      s:=(3*P[1]^2+a)/(2*P[2]):
      xR:=(s^2-2*P[1]):
      yR:=s*(P[1]-xR)-P[2]:
      RETURN([xR,yR]):
     else
      RETURN(FAIL):
    fi:
  fi:
end:

 #EC(a,b,p): all the points (x,y) such that y^2=x^3+a*x+b mod p 
EC:=proc(a,b,p) local x,y,S:
S:={}:
 for  x from 0 to p-1 do
  for y from 0 to p-1 do
   if (y^2-x^3-a*x-b) mod p=0 then
     S:=S union {[x,y]}:
   fi:
 od:
od: 
S union {infinity}:
end:

#AEp(a,b,P,Q,p): addiditon in the elliptic ruve y^2=x^3+a*x+b mod p
 AEp:=proc(a,b,P,Q,p) local x,y,s,R,xR,yR:
if P=infinity then
  RETURN(Q):
fi:
if Q=infinity then
  RETURN(P):
fi:

if P[1]<>Q[1] then
 s:=(P[2]-Q[2])/(P[1]-Q[1]) mod p:
 xR:=s^2-P[1]-Q[1] mod p:
 yR:=s*(P[1]-xR)-P[2] mod p:
 RETURN([xR,yR]):
 else
  if P[1]=Q[1] and P[2]<>Q[2] then
   RETURN(infinity):
   elif P[1]=Q[1] and P[2]=0 and Q[2]=0 then
    RETURN(infinity):
    elif P=Q then
      s:=(3*P[1]^2+a)/(2*P[2]) mod p:
      xR:=(s^2-2*P[1]) mod p:
      yR:=s*(P[1]-xR)-P[2] mod p:
      RETURN([xR,yR]):
     else
      RETURN(FAIL):
    fi:
  fi:
end:    

#Mul(g,A,a,b,p): inputs a member,g, of the elliptic curve y^2=x^3+a*x+b mod p
#and outputs g+g+...+g (A times)
Mul:=proc(g,A,a,b,p) local B:
if A=1 then
  RETURN(g):
elif A mod 2=0 then
 B:=Mul(g,A/2,a,b,p):
 RETURN(AEp(a,b,B,B,p)):
else
 RETURN(AEp(a,b,Mul(g,A-1,a,b,p),g,p)):
fi:

end:
 

#Sqrt(n,p,K): the x such that x^2=n mod p
Sqrt:=proc(n,p,K) local a,i,ra,x:
ra:=rand(1..p-1):

if n&^((p-1)/2) mod p=p-1 then
   RETURN(FAIL):
fi:

for i from 1 to K do
 a:=ra():
if (a^2-n mod p)&^((p-1)/2) mod p=p-1 then
   x:=expand((a+sqrt(a^2-n))^((p+1)/2)):
   
  RETURN(op(1,x) mod p):
  fi:
od:
FAIL:
end:



 

#hw begin
#1 #2
#3 
RandPt := proc(a,b,p):
   
    for i from 1 to p do
        xCan := rand(0..(p-1))():
        yCan := Sqrt(xCan^3 +  a*xCan + b, p, p):
        if yCan <> FAIL then 
            RETURN([xCan,yCan]):
        fi:
    od:
    RETURN(FAIL):
end:
#4
AliceToBobEC := proc(a,b,p,g):
    A := rand(0..(p-1)):
    [A,Mul(A,a,b,g,p)]:
end:

BobToAliceEC := proc(a,b,p,g):
    B := rand(0..(p-1)):
    [A,Mul(B,a,b,g,p)]:
end:

#5 
Hasse := proc(a,b,K):
    p := 3:
    minP := [nops(EC(a,b,P)),p]:
    maxP := minP:
    p := 5:
    while p <= K do
        canNum := nops(EC(a,b,P)):
        if canNum < minP[1] then
            minP := [canNum,p]:
        fi:
        if canNum > maxP[1] then
            maxP := [canNum,p]:
        fi:
        p := nextprime(p):
    od:
    [minP,maxP]:
=======
#C21.txt, 4/4/24

Help:=proc(): print(`   AE(a,b,P,Q), EC(a,b,p), AEp(a,b,P,Q,p), Mul(g,A,a,b,p), Sqrt(n,p,K) `):end:

#old code

Help20:=proc(): print(`RP(a,b), DoesLie(a,b,P), AddE(a,b,P,Q) `):end:
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

###end of C20.txt

AE:=proc(a,b,P,Q) local x,y,s,R,xR,yR:
if P=infinity then
  RETURN(Q):
fi:
if Q=infinity then
  RETURN(P):
fi:

if P[1]<>Q[1] then
 s:=(P[2]-Q[2])/(P[1]-Q[1]):
 xR:=s^2-P[1]-Q[1]:
 yR:=s*(P[1]-xR)-P[2]:
 RETURN([xR,yR]):
 else
  if P[1]=Q[1] and P[2]<>Q[2] then
   RETURN(infinity):
   elif P[1]=Q[1] and P[2]=0 and Q[2]=0 then
    RETURN(infinity):
    elif P=Q then
      s:=(3*P[1]^2+a)/(2*P[2]):
      xR:=(s^2-2*P[1]):
      yR:=s*(P[1]-xR)-P[2]:
      RETURN([xR,yR]):
     else
      RETURN(FAIL):
    fi:
  fi:
end:

 #EC(a,b,p): all the points (x,y) such that y^2=x^3+a*x+b mod p 
EC:=proc(a,b,p) local x,y,S:
S:={}:
 for  x from 0 to p-1 do
  for y from 0 to p-1 do
   if (y^2-x^3-a*x-b) mod p=0 then
     S:=S union {[x,y]}:
   fi:
 od:
od: 
S union {infinity}:
end:

#AEp(a,b,P,Q,p): addiditon in the elliptic ruve y^2=x^3+a*x+b mod p
 AEp:=proc(a,b,P,Q,p) local x,y,s,R,xR,yR:
if P=infinity then
  RETURN(Q):
fi:
if Q=infinity then
  RETURN(P):
fi:

if P[1]<>Q[1] then
 s:=(P[2]-Q[2])/(P[1]-Q[1]) mod p:
 xR:=s^2-P[1]-Q[1] mod p:
 yR:=s*(P[1]-xR)-P[2] mod p:
 RETURN([xR,yR]):
 else
  if P[1]=Q[1] and P[2]<>Q[2] then
   RETURN(infinity):
   elif P[1]=Q[1] and P[2]=0 and Q[2]=0 then
    RETURN(infinity):
    elif P=Q then
      s:=(3*P[1]^2+a)/(2*P[2]) mod p:
      xR:=(s^2-2*P[1]) mod p:
      yR:=s*(P[1]-xR)-P[2] mod p:
      RETURN([xR,yR]):
     else
      RETURN(FAIL):
    fi:
  fi:
end:    

#Mul(g,A,a,b,p): inputs a member,g, of the elliptic curve y^2=x^3+a*x+b mod p
#and outputs g+g+...+g (A times)
Mul:=proc(g,A,a,b,p) local B:
if A=1 then
  RETURN(g):
elif A mod 2=0 then
 B:=Mul(g,A/2,a,b,p):
 RETURN(AEp(a,b,B,B,p)):
else
 RETURN(AEp(a,b,Mul(g,A-1,a,b,p),g,p)):
fi:

end:
 

#Sqrt(n,p,K): the x such that x^2=n mod p
Sqrt:=proc(n,p,K) local a,i,ra,x:
ra:=rand(1..p-1):

if n&^((p-1)/2) mod p=p-1 then
   RETURN(FAIL):
fi:

for i from 1 to K do
 a:=ra():
if (a^2-n mod p)&^((p-1)/2) mod p=p-1 then
   x:=expand((a+sqrt(a^2-n))^((p+1)/2)):
   
  RETURN(op(1,x) mod p):
  fi:
od:
FAIL:
end:



 

#hw begin
#1 #2
#3 
RandPt := proc(a,b,p):
   
    for i from 1 to p do
        xCan := rand(0..(p-1))():
        yCan := Sqrt(xCan^3 +  a*xCan + b, p, p):
        if yCan <> FAIL then 
            RETURN([xCan,yCan]):
        fi:
    od:
    RETURN(FAIL):
end:
#4
AliceToBobEC := proc(a,b,p,g):
    A := rand(0..(p-1)):
    [A,Mul(A,a,b,g,p)]:
end:

BobToAliceEC := proc(a,b,p,g):
    B := rand(0..(p-1)):
    [A,Mul(B,a,b,g,p)]:
end:

#5 
Hasse := proc(a,b,K):
    p := 3:
    minP := [nops(EC(a,b,P)),p]:
    maxP := minP:
    p := 5:
    while p <= K do
        canNum := nops(EC(a,b,P)):
        if canNum < minP[1] then
            minP := [canNum,p]:
        fi:
        if canNum > maxP[1] then
            maxP := [canNum,p]:
        fi:
        p := nextprime(p):
    od:
    [minP,maxP]:
>>>>>>> Stashed changes
end:
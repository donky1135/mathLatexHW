<<<<<<< Updated upstream
#C19.txt, March 28, 2024; Public Key Cryptography; Diffie-Hellman
Help:=proc(): print(`FindRP(d), IsPrimi(g,P), FindPrimi(P), FindSGP(d,K)`):
print(`FindPrimiSGP(P), AliceToBob(P,g), BobToALice(P,g) `):
end:

#FindRD(d): finds a random prime with d digits
FindRP:=proc(d):
nextprime(rand(10^(d-1)..10^d-1)()):
end:

#IsPrim(g,P): Is g primitive mod P? {g,g^2,..., g^(p-2)} are all different
#if g^i mod p=1 doesn't happen until i=p-1
IsPrim:=proc(g,P) local i:
for i from 1 to P-2 do
 if g&^i mod P=1 then
    RETURN(false):
 fi: 
od:
true:
end:

#FindPrimi(P): given a prime P, finds a primitive g by brute force
FindPrimi:=proc(P) local g:
for g from 2 to P-2 do
  if IsPrim(g,P) then
    RETURN(g):
   fi:
od:

FAIL: 
end:

#FindSGP(d,K): finds a random Sophie Germain prime such that (P-1)/2 has d digits
#trying K times or return FAIL
FindSGP:=proc(d,K) local Q,i,ra:
ra:=rand(10^(d-1)..10^d-1):

for i from 1 to K do
 Q:=nextprime(ra()):
 if isprime(2*Q+1) then
    RETURN(2*Q+1):
 fi:
od:
FAIL:
end:

#FindPrimiSGP(P): finds a random primitive element mod P if P is a Sophie Germain prime
FindPrimiSGP:=proc(P) local Q,a:
Q:=(P-1)/2:
if not (isprime(P) and isprime(Q)) then
  RETURN(FAIL):
fi:

a:=rand(2..P-2)():
if a&^Q mod P=P-1 then
   RETURN(a):
 else
  RETURN(P-a):
fi:
end:

#AliceToBob(P,g): alice picks her secret integer a from 2 to P-2 does not
#tell anyone (not even Bob) and sends Bob g^a mod P
AliceToBob:=proc(P,g) local a:
a:=rand(3..P-3)():
[a,g&^a mod P]:
end:


#BobToAice(P,g): Bob picks his secret integer b from 2 to P-2 does not
#tell anyone (not even Alice) and sends Alice g^b mod P
BobToAlice:=proc(P,g) local b:
b:=rand(3..P-3)():
[b,g&^b mod P]:
end:




#actual HW

p := proc(n):
    sumValTot := 0:
    for m from 1 to 2^n do
        sumValIntern := 0:
        sumValIntern := 1+ add(seq(floor(((1+(j-1)!)/j) - floor(((j-1)!/j))), j=2..m)):
        sumValIntern := floor(n/sumValIntern):
        sumValIntern := floor((sumValIntern)**(1/n)):
        sumValTot := sumValIntern + sumValTot:
    od:
    return (1 + sumValTot):
end:

t := proc(n):
    sumValIntern := 1+add(seq(floor((n+2)/i - floor((n+1)/i)), i=2..(n+1))):
    return 2 + n*floor(1/sumValIntern):
end:

PI := proc(n):
    floor(add(seq(evalf((sin(Pi*(k-1)!/(2*k)))^2), k=1..n)))
end:

#5

DL := proc(x,P,g):
    for a from 1 to (P-1) do
        if x = g&^a mod P then 
            RETURN(a):
        fi:
    od:
    RETURN(FAIL):
end:

=======
#C19.txt, March 28, 2024; Public Key Cryptography; Diffie-Hellman
Help:=proc(): print(`FindRP(d), IsPrimi(g,P), FindPrimi(P), FindSGP(d,K)`):
print(`FindPrimiSGP(P), AliceToBob(P,g), BobToALice(P,g) `):
end:

#FindRD(d): finds a random prime with d digits
FindRP:=proc(d):
nextprime(rand(10^(d-1)..10^d-1)()):
end:

#IsPrim(g,P): Is g primitive mod P? {g,g^2,..., g^(p-2)} are all different
#if g^i mod p=1 doesn't happen until i=p-1
IsPrim:=proc(g,P) local i:
for i from 1 to P-2 do
 if g&^i mod P=1 then
    RETURN(false):
 fi: 
od:
true:
end:

#FindPrimi(P): given a prime P, finds a primitive g by brute force
FindPrimi:=proc(P) local g:
for g from 2 to P-2 do
  if IsPrim(g,P) then
    RETURN(g):
   fi:
od:

FAIL: 
end:

#FindSGP(d,K): finds a random Sophie Germain prime such that (P-1)/2 has d digits
#trying K times or return FAIL
FindSGP:=proc(d,K) local Q,i,ra:
ra:=rand(10^(d-1)..10^d-1):

for i from 1 to K do
 Q:=nextprime(ra()):
 if isprime(2*Q+1) then
    RETURN(2*Q+1):
 fi:
od:
FAIL:
end:

#FindPrimiSGP(P): finds a random primitive element mod P if P is a Sophie Germain prime
FindPrimiSGP:=proc(P) local Q,a:
Q:=(P-1)/2:
if not (isprime(P) and isprime(Q)) then
  RETURN(FAIL):
fi:

a:=rand(2..P-2)():
if a&^Q mod P=P-1 then
   RETURN(a):
 else
  RETURN(P-a):
fi:
end:

#AliceToBob(P,g): alice picks her secret integer a from 2 to P-2 does not
#tell anyone (not even Bob) and sends Bob g^a mod P
AliceToBob:=proc(P,g) local a:
a:=rand(3..P-3)():
[a,g&^a mod P]:
end:


#BobToAice(P,g): Bob picks his secret integer b from 2 to P-2 does not
#tell anyone (not even Alice) and sends Alice g^b mod P
BobToAlice:=proc(P,g) local b:
b:=rand(3..P-3)():
[b,g&^b mod P]:
end:




#actual HW

p := proc(n):
    sumValTot := 0:
    for m from 1 to 2^n do
        sumValIntern := 0:
        sumValIntern := 1+ add(seq(floor(((1+(j-1)!)/j) - floor(((j-1)!/j))), j=2..m)):
        sumValIntern := floor(n/sumValIntern):
        sumValIntern := floor((sumValIntern)**(1/n)):
        sumValTot := sumValIntern + sumValTot:
    od:
    return (1 + sumValTot):
end:

t := proc(n):
    sumValIntern := 1+add(seq(floor((n+2)/i - floor((n+1)/i)), i=2..(n+1))):
    return 2 + n*floor(1/sumValIntern):
end:

PI := proc(n):
    floor(add(seq(evalf((sin(Pi*(k-1)!/(2*k)))^2), k=1..n)))
end:

#5

DL := proc(x,P,g):
    for a from 1 to (P-1) do
        if x = g&^a mod P then 
            RETURN(a):
        fi:
    od:
    RETURN(FAIL):
end:

>>>>>>> Stashed changes
#This is very slow, don't try testing for 10 digit primes
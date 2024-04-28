#C27.txt April 25, 2024
Help:=proc(): print(`RGC3(n), ZKP3(n,E,C,Opt)`): end:
with(combinat):
#RGC3(n): inputs a pos. integer and outputs the triple (n,E,C)
#where E is the set of edges and C is a proper coloring
RGC3:=proc(n) local C,i,j,ra,ra1,E:
ra:=rand(1..3):
C:=[seq(ra(),i=1..n)]:
ra1:=rand(0..1):
E:={}:
for i from 1 to n do
 for j from i+1 to n do
   if C[i]<>C[j] then
     if ra1()=1 then
       E:=E union {{i,j}}:
     fi:
   fi:
 od:
od:
n,E,C:
end:

ZKP3:=proc(n,E,C,Opt)  local B1,B2,i,j,c,sig,B1iV, B1iC, B1jV, B1jC, B1v,B1c,S:
sig:=randperm(3*n):
B1:=[seq(seq([c,i],c=1..3),i=1..n)]:
B1:=[seq(B1[sig[i]],i=1..nops(B1))]:
for i from 1 to 3*n do
 for j from 1 to 3*n do
  B1iC:=B1[i][1]:
  B1iV:=B1[i][2]:
  B1jC:=B1[j][1]:
  B1jV:=B1[j][2]:
  if C[B1iV]=B1iC and C[B1jV]=B1jC and member({B1iV,B1jV},E) then
    B2[i,j]:=1:
   else
   B2[i,j]:=0:
 fi:
od:
od:
B2:=[seq([seq(B2[i,j],j=1..3*n)],i=1..3*n)]:

if Opt=1 then
 B1v:=[seq(B1[i][2],i=1..3*n)]:
  Verify1(n,E,B1v,B2):

else
 B1c:=[seq(B1[i][1],i=1..3*n)]:
S:={}:
for i from 1 to 3*n do
 #for j from 1 to 3*n do
 for j from i+1 to 3*n do
   if B1[i][1]=B1[j][1] then
    S:=S union {[[i,j],B2[i][j]]}:
   fi:
 od:
od:
RETURN(Verify2(B1c,S)):
fi:
end:

#Verify1(n,E,B2,B1v): verifies that B2 and B1v are consistent with the graph
Verify1:=proc(n,E,B2,B1v) local i,j:
 for i from 1 to 3*n do
   for j from 1 to 3*n do
  if not member({B1v[i],B1v[j]},E) and B2[i][j]=1 then
    print(`Liar!`):
   RETURN(false):
 fi:
od:
od:
true:
end:

Verify2:=proc(B1c,S) local s:
for s in S do
 if s[2]<>0 then
   RETURN(false):
 fi:
if B1c[s[1][1]]<>B1c[s[1][2]] then
  RETURN(false):
fi:
od:
true:
end:


#1 
#2
MetaZKP3 := proc(n,E,C,K):
    goodGraph := true:
    for i from 1 to K do
        choice := rand(0.0..1.0)():
        if choice <= 0.5 then 
            if not ZKP3(n,E,C,1) then 
                goodGraph := false
            fi:
        else 
            if not ZKP3(n,E,C,2) then 
                goodGraph := false
            fi:
        fi:
    od:

    goodGraph:
end:

# I ran the program on a 10,20,30, 40, node graph, and they all checked out.
# If I run randperm on the coloring array C, I got false! I tested this 10 times, 
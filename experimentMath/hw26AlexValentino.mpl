#C26.txt, April 22, 2024
Help:=proc(): print(` FT(SetL), TM(SetL) `):end:
read `ENGLISH.txt`:



#FT(SetL): The frequency table of all words whose length belobgs to SetL
#For example FT({3,4}) returns the randked frequency table for words of
#length 3 and 4
FT:=proc(SetL) local ALPH,T1,W,DB,C,B,FL,L,A,FL1,S1:
ALPH:=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]:
DB:=ENG():
C:=0:
for L in ALPH do
 T1[L]:=0:
od:

for A in SetL do
 for W in DB[A] do
  for B from 1 to nops(W) do
    T1[W[B]]:=T1[W[B]]+1:
    C:=C+1:
  od:
 od:
od:
FL:=evalf([seq(T1[ALPH[A]]/C,A=1..nops(ALPH))]):
FL1:=sort(FL,`>`):
for A from 1 to nops(ALPH) do
 S1[FL[A]]:=ALPH[A]:
 od:

FL1,[seq(S1[FL1[A] ],A=1..nops(ALPH))], [seq([FL1[A],S1[FL1[A]] ],A=1..nops(ALPH))] :

end:


#TM(SetL): The frequency table of all words whose length belobgs to SetL
#For example FT({3,4}) returns the randked frequency table for words of
#length 3 and 4
#AND the Transiton matrix 28x28 the first "letter" is ST and the last is EN
#NOT YET FINISHED
TM:=proc(SetL) local ALPH,T1,W,DB,C,B,FL,L,A,FL1,S1,T2,L1,L2,ST,EN,T3,ALPH1:
print(`Not yet finished`):
ALPH:=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]:
DB:=ENG():
C:=0:
for L in ALPH do
 T1[L]:=0:
od:

for L1 from 1 to nops(ALPH) do
 T2[ST,ALPH[L1]]:=0:
  T2[ALPH[L1],EN]:=0:
 for L2 from 1 to nops(ALPH) do
  T2[ALPH[L1],ALPH[L2]]:=0:
  od:
od:

for A in SetL do
 for W in DB[A] do
  for B from 1 to nops(W) do
    T1[W[B]]:=T1[W[B]]+1:

   C:=C+1:

    if B<nops(W) then
     T2[W[B],W[B+1]]:=T2[W[B],W[B+1]]+1:
     else
      T2[W[B],EN]:=T2[W[B],EN]+1:
     fi:

    if B=1 then
    T2[ST,W[1]]:=T2[ST,W[1]]+1:
    fi:
   
  od:
 od:
od:
FL:=evalf([seq(T1[ALPH[A]]/C,A=1..nops(ALPH))]):
FL1:=sort(FL,`>`):
for A from 1 to nops(ALPH) do
 S1[FL[A]]:=ALPH[A]:
 od:


for L1 from 1 to nops(ALPH) do

T3[ST,ALPH[L1]]:=evalf(T2[ST,ALPH[L1]]/C):
T3[ALPH[L1],EN]:=evalf(T2[ALPH[L1],EN]/C):
od:

for L1 from 1 to nops(ALPH) do
for L2 from 1 to nops(ALPH) do
 T3[ALPH[L1],ALPH[L2]]:=evalf(T2[ALPH[L1],ALPH[L2]]/C):
od:
od:
 

FL1,[seq(S1[FL1[A] ],A=1..nops(ALPH))], [seq([FL1[A],S1[FL1[A]] ],A=1..nops(ALPH))] , op(T3):

end:


#1 
#2 
prob2 := proc() local I, J, L, W, Tcon, Tacc:
    ALPH:=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]:
    for I in ALPH do
        for J in ALPH do
            Tcon[I,J] := 0:
        od:
    od:
    DB:=ENG():
    for I from 2 to 28 do
        for W in DB[I] do
            for L from 2 to I do
                Tcon[W[L-1], W[L]] := Tcon[W[L-1], W[L]] + 1:
            od:
        od:
    od:
    
    Tacc := {}:

    for I in ALPH do
        for J in ALPH do
            if Tcon[I,J] <> 0 then 
                Tacc := Tacc union {{I,J}}:
            fi:
        od:
    od:

    Tacc minus {seq({A}, A in ALPH)}, Tcon:

end:
#after computing the size of Tacc divided by 676 I get that ~45% of all pairs of letters 
#occur in the given dictionary 

prob3 := proc() local I, J, L, W, Tcon, Tacc:
    ALPH:=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]:
    for I in ALPH do
        for J in ALPH do
            Tcon[I,J] := 0:
        od:
    od:
    DB:=ENG():
    for I from 2 to 28 do
        for W in DB[I] do
            for L from 2 to I do
                Tcon[W[L-1], W[L]] := Tcon[W[L-1], W[L]] + 1:
            od:
        od:
    od:
    
    Tacc := {}:

    for I in ALPH do
        for J in ALPH do
            if Tcon[I,J] > 10 then 
                Tacc := Tacc union {{I,J}}:
            fi:
        od:
    od:

    Tacc minus {seq({A}, A in ALPH)}, Tcon:

end:

#after computing the size of Tacc divided by 676 I get that ~39% of all pairs of letters occur 
#in the given dictionary more then 10 times 

prob4 := proc() local I, J, L, W, Tcon, Tacc:
    ALPH:=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]:
    for I in ALPH do
        for J in ALPH do
            for K in ALPH do
                Tcon[I,J,K] := 0:
            od:
        od:
    od:
    DB:=ENG():
    for I from 3 to 28 do
        for W in DB[I] do
            for L from 3 to I do
                Tcon[W[L-2],W[L-1], W[L]] := Tcon[W[L-2],W[L-1], W[L]] + 1:
            od:
        od:
    od:
    
    Tacc := {}:

    for I in ALPH do
        for J in ALPH do
            for K in ALPH do
                if Tcon[I,J] <> 0 then 
                    Tacc := Tacc union {{I,J,K}}:
                fi:
            od:
        od:
    od:

    Tacc minus {seq(seq({A,B}, B in ALPH), A in ALPH)}, Tcon:

end:

#it appears that 14% of all letter triples occur in this dictionary
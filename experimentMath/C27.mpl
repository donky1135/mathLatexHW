
#RGC3 inputs a postitive integer and outputs the triple (n,E,C) E is the edge set and C is a coloring 
RGC3 := proc(n) local C,i,j, ra, ra1, E:
    ra := rand(1..3):
    C := [seq(ra(), i=1..n)]:

    ra1 := rand(0..1):
    E:={}:
    for i from 1 to n do
        for j from i+1 to n do
            if C[i] <> C[j] then 
                if ra1()=1 then 
                    E := E union {{i,j}}:
                fi:
            fi:
        od:
    od:
    n, E, C:
end:

ZKP3 := proc(n,E,C,Opt) local B1,B2,i,j,c,sig:
    sig = randperm(3*n):
    B1 := [seq([seq([c,i],c=1..3)], i=1..n)]:
    B1 := [seq(B1[sig[i]], i=1..nops(B1))]:
    for i from 1 to 3*n do 
        for j from 1 to 3*n do
        B1iC:= B1[i][1]:
        B1iV := B1[i][2]:
        
end:
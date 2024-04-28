
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

ZKP3 := proc(n.E,C,Opt) local B1,B2:

end:
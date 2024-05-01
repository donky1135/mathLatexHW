Fqn:=proc(q,n) local S,a,v:
option remember:
if n=0 then
 RETURN({[]}):
fi:

S:=Fqn(q,n-1):

{seq(seq([op(v),a],a=0..q-1), v in S)}:

end:


GS := proc(w1,w2,N):
    validWords := []:
    for i from 1 to N do
        validIWords  := {}:
        if i < nops(w1) or i < nops(w2) then 
            validWords := [op(validWords), validIWords]:
        else
            iWords := Fqn(2,i):
            for word in iWords do
                w1count := 0:
                w2count := 0:
                # print(word):
                if nops(w1) <= i then
                for k from 1 to i - nops(w1) + 1 do
                    # print(evalb(w1 = word[k..k-1 + nops(w1)])):
                    if w1 = word[k..k-1 + nops(w1)] then
                        w1count++:
                    fi:
                od: 
                fi:
                # print(w1count):
                if nops(w2) <= i then 
                for k from 1 to i - nops(w2) + 1 do
                    # print(evalb(w2 = word[k..k-1 + nops(w2)])):
                    if w2 = word[k..k-1 + nops(w2)] then
                        w2count++:
                    fi:
                od:
                fi:
                # print(w2count):
                if w1count > w2count then 
                    validIWords := {word} union validIWords:
                fi:
            od:
            validWords := [op(validWords), validIWords]:
        fi:
    od:
    validWords:
end:
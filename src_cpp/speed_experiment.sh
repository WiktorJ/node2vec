#!/usr/bin/env bash

graph=lesmis-fix.edgelist
emb=ls.emb
flags="-dry -v -ow"
p=0.5
q=2
r=32
l=80
build/node2vec/node2vec -i:../graph/${graph}  -o:../emb/${emb} -l:${l} -p:${p} -q:${q}  -r:${r} ${flags} 
build/node2vec_ms/node2vec_ms -i:../graph/${graph}  -o:../emb/${emb} -l:${l} -p:${p} -q:${q}  -r:${r} ${flags}
build/node2vec_ms_bias/node2vec_ms_bias -i:../graph/${graph}  -o:../emb/${emb} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.2 ${flags}
build/node2vec_ms_bias/node2vec_ms_bias -i:../graph/${graph}  -o:../emb/${emb} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.4 ${flags}
build/node2vec_ms_bias/node2vec_ms_bias -i:../graph/${graph}  -o:../emb/${emb} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.6 ${flags}
build/node2vec_ms_bias/node2vec_ms_bias -i:../graph/${graph}  -o:../emb/${emb} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.8 ${flags}

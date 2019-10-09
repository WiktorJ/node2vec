#!/usr/bin/env bash

graph=lesmis-fix.edgelist
emb=ls.emb
flags="-dry -ow"
p=0.5
q=2
r=32
l=80
paths="-i:../graph/lesmis-fix.edgelist -o:../emb/ls.em"
declare -a arr=("-i:../graph/lesmis-fix.edgelist -o:../emb/ls.em", "-i:../graph/roadsNet-PA-fix.txt -o:../emb/ls.em")
for paths in "${arr[@]}"
do
    echo paths
    echo "base"
    build/node2vec/node2vec ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} ${flags}
    echo "MS"
    build/node2vec_ms/node2vec_ms ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} ${flags}
    echo "Biased 0.2"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.2 ${flags}
    echo "Biased 0.4"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.4 ${flags}
    echo "Biased 0.6"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.6 ${flags}
    echo "Biased 0.8"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.8 ${flags}
    echo "------------------------------------------------"
done
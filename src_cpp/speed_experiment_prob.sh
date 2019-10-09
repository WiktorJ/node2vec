#!/usr/bin/env bash

graph=lesmis-fix.edgelist
emb=ls.emb
flags="-dry -ow"
p=0.5
q=2
r=32
l=80
declare -a arr=("-i:../graph/facebook_combined.edgelist -o:../emb/ls.em", "-i:../graph/roadNet-PA-fix.txt -o:../emb/r.em", "-i:../graph/twitter_combined-fix.txt -o:../emb/t.em",  "-i:../graph/com-youtube-fix.ungraph.txt -o:../emb/y.em")
for paths in "${arr[@]}"
do
    echo "Biased 0.1"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.2 ${flags}
    echo "Biased 0.2"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.2 ${flags}
    echo "Biased 0.3"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.3 ${flags}
    echo "Biased 0.4"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.4 ${flags}
    echo "Biased 0.5"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.5 ${flags}
    echo "Biased 0.6"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.6 ${flags}
    echo "Biased 0.7"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.7 ${flags}
    echo "Biased 0.8"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.8 ${flags}
    echo "Biased 0.9"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rb:0.9 ${flags}
    echo "------------------------------------------------"
    echo "------------------------------------------------"
    echo "------------------------------------------------"
done
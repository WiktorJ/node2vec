#include <chrono>
#include "stdafx.h"
#include "n2v.h"
#include "vector"

void node2vec(PWNet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, int64> &WalksVV,
              TIntFltVH &EmbeddingsHV) {
    auto start_time = std::chrono::high_resolution_clock::now();
    //Preprocess transition probabilities
    PreprocessTransitionProbs(InNet, ParamP, ParamQ, Verbose);
    TIntV NIdsV;
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        NIdsV.Add(NI.GetId());
    }
    //Generate random walks
    int64 AllWalks = (int64) NumWalks * NIdsV.Len();
//    int64 AllWalks = (int64) NIdsV.Len();
    WalksVV = TVVec<TInt, int64>(AllWalks, WalkLen);
    TRnd Rnd(time(NULL));
    int64 WalksDone = 0;

    for (int i = 0; i < NIdsV.Len(); ++i) {
        std::vector<int64> start_nodes;
        start_nodes.push_back(NIdsV[i]);
        int64 current_walk_number = i * WalkLen;
        SimulateWalk(InNet, WalksVV, start_nodes, WalkLen, NumWalks, Rnd, current_walk_number);
    }


    auto walk_end_time = std::chrono::high_resolution_clock::now();
    if (!OutputWalks) {
        LearnEmbeddings(WalksVV, Dimensions, WinSize, Iter, Verbose, EmbeddingsHV);
        auto learn_end_time = std::chrono::high_resolution_clock::now();
    }
    printf("\rWalk time: %lld ms",
           std::chrono::duration_cast<std::chrono::milliseconds>(walk_end_time - start_time).count());
}

void node2vec(PWNet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              TIntFltVH &EmbeddingsHV) {
    TVVec<TInt, int64> WalksVV;
    bool OutputWalks = 0;
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
             Iter, Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}


void node2vec(const PNGraph &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, int64> &WalksVV,
              TIntFltVH &EmbeddingsHV) {
    PWNet NewNet = PWNet::New();
    for (TNGraph::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
        if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
        if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
        NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), 1.0);
    }
    node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
             Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

void node2vec(const PNGraph &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              TIntFltVH &EmbeddingsHV) {
    TVVec<TInt, int64> WalksVV;
    bool OutputWalks = 0;
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
             Iter, Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

void node2vec(const PNEANet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, int64> &WalksVV,
              TIntFltVH &EmbeddingsHV) {
    PWNet NewNet = PWNet::New();
    for (TNEANet::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
        if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
        if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
        NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), InNet->GetFltAttrDatE(EI, "weight"));
    }
    node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
             Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

void node2vec(const PNEANet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              TIntFltVH &EmbeddingsHV) {
    TVVec<TInt, int64> WalksVV;
    bool OutputWalks = 0;
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
             Iter, Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}


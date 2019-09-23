#include "stdafx.h"
#include "Snap.h"
#include "biasedrandomwalk.h"
#include <unordered_map>
#include <tuple>
#include <numeric>

//Preprocess alias sampling method
void GetNodeAlias(TFltV &PTblV, TIntVFltVPr &NTTable) {
    int64 N = PTblV.Len();
    TIntV &KTbl = NTTable.Val1;
    TFltV &UTbl = NTTable.Val2;
    for (int64 i = 0; i < N; i++) {
        KTbl[i] = 0;
        UTbl[i] = 0;
    }
    TIntV UnderV;
    TIntV OverV;
    for (int64 i = 0; i < N; i++) {
        UTbl[i] = PTblV[i] * N;
        if (UTbl[i] < 1) {
            UnderV.Add(i);
        } else {
            OverV.Add(i);
        }
    }
    while (UnderV.Len() > 0 && OverV.Len() > 0) {
        int64 Small = UnderV.Last();
        int64 Large = OverV.Last();
        UnderV.DelLast();
        OverV.DelLast();
        KTbl[Small] = Large;
        UTbl[Large] = UTbl[Large] + UTbl[Small] - 1;
        if (UTbl[Large] < 1) {
            UnderV.Add(Large);
        } else {
            OverV.Add(Large);
        }
    }
    while (UnderV.Len() > 0) {
        int64 curr = UnderV.Last();
        UnderV.DelLast();
        UTbl[curr] = 1;
    }
    while (OverV.Len() > 0) {
        int64 curr = OverV.Last();
        OverV.DelLast();
        UTbl[curr] = 1;
    }

}

//Get random element using alias sampling method
int64 AliasDrawInt(TIntVFltVPr &NTTable, TRnd &Rnd) {
    int64 N = NTTable.GetVal1().Len();
    TInt X = static_cast<int64>(Rnd.GetUniDev() * N);
    double Y = Rnd.GetUniDev();
    return Y < NTTable.GetVal2()[X] ? X : NTTable.GetVal1()[X];
}

void PreprocessNode(PWNet &InNet, const double &ParamP, const double &ParamQ,
                    TWNet::TNodeI NI, int64 &NCnt, const bool &Verbose) {
    if (Verbose && NCnt % 100 == 0) {
        printf("\rPreprocessing progress: %.2lf%% ", (double) NCnt * 100 / (double) (InNet->GetNodes()));
        fflush(stdout);
    }
    //for node t
    THash<TInt, TBool> NbrH;                                    //Neighbors of t
    for (int64 i = 0; i < NI.GetOutDeg(); i++) {
        NbrH.AddKey(NI.GetNbrNId(i));
    }
    for (int64 i = 0; i < NI.GetOutDeg(); i++) {
        TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));      //for each node v
        double Psum = 0;
        TFltV PTable;                              //Probability distribution table
        for (int64 j = 0; j < CurrI.GetOutDeg(); j++) {           //for each node x
            int64 FId = CurrI.GetNbrNId(j);
            TFlt Weight;
            if (!(InNet->GetEDat(CurrI.GetId(), FId, Weight))) { continue; }
            if (FId == NI.GetId()) {
                PTable.Add(Weight / ParamP);
                Psum += Weight / ParamP;
            } else if (NbrH.IsKey(FId)) {
                PTable.Add(Weight);
                Psum += Weight;
            } else {
                PTable.Add(Weight / ParamQ);
                Psum += Weight / ParamQ;
            }
        }
        //Normalizing table
        for (int64 j = 0; j < CurrI.GetOutDeg(); j++) {
            PTable[j] /= Psum;
        }
        GetNodeAlias(PTable, CurrI.GetDat().GetDat(NI.GetId()));
    }
    NCnt++;
}

//Preprocess transition probabilities for each path t->v->x
void PreprocessTransitionProbs(PWNet &InNet, const double &ParamP, const double &ParamQ, const bool &Verbose) {
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        InNet->SetNDat(NI.GetId(), TIntIntVFltVPrH());
    }
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        for (int64 i = 0; i <
                          NI.GetOutDeg(); i++) {                    //allocating space in advance to avoid issues with multithreading
            TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
            CurrI.GetDat().AddDat(NI.GetId(), TPair<TIntV, TFltV>(TIntV(CurrI.GetOutDeg()), TFltV(CurrI.GetOutDeg())));
        }
    }
    int64 NCnt = 0;
    TIntV NIds;
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        NIds.Add(NI.GetId());
    }
#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < NIds.Len(); i++) {
        PreprocessNode(InNet, ParamP, ParamQ, InNet->GetNI(NIds[i]), NCnt, Verbose);
    }
    if (Verbose) { printf("\n"); }
}

struct pair_hash {
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const {
        const auto first_hash = std::hash<T1>{}(pair.first);
        return first_hash ^ (std::hash<T2>{}(pair.second) + 0x9e3779b9 + (first_hash << 6) + (first_hash >> 2));
    }
};

std::vector<int64> draw_edge(PWNet &InNet,
                             int64 current_node,
                             int64 previous_node,
                             TRnd &Rnd,
                             int64 steps_number) {
    std::vector<int64> drawn;
    drawn.reserve(steps_number);
    for (auto i = 0; i < steps_number; i++) {
        drawn.push_back(InNet->GetNI(current_node).GetNbrNId(
                AliasDrawInt(InNet->GetNDat(current_node).GetDat(previous_node), Rnd)));
    }
    return drawn;
}

void update_step(std::vector<int64> &drawn,
                 int64 current_node,
                 TVVec<TInt, int64> &WalksVV,
                 std::unordered_map<std::pair<int64, int64>, std::vector<int64>, pair_hash> &visit_next,
                 std::vector<int64> &indexes,
                 int64 walk_length,
                 int64 current_length) {

    for (auto next_node: drawn) {
        auto index = indexes.back();
        if (current_length < walk_length - 1) {
            visit_next[std::make_pair(current_node, next_node)].push_back(index);
        }
        indexes.pop_back();
        WalksVV.PutXY(index, current_length, next_node);
    }

}


//Simulates a random walk
void SimulateWalk(PWNet &InNet,
                  TVVec<TInt, int64> &WalksVV,
                  std::vector<int64> StartNIds,
                  const int &WalkLen,
                  const int &NumWalk,
                  TRnd &Rnd,
                  int64 current_walk_number) {
    std::unordered_map<std::pair<int64, int64>, std::vector<int64>, pair_hash> visit;
    std::unordered_map<std::pair<int64, int64>, std::vector<int64>, pair_hash> visit_next;
    for (auto &node: StartNIds) {
        auto node_iterator = InNet->GetNI(node);
        if (node_iterator.GetOutDeg() != 0) {
            for (auto i = 0; i < NumWalk; i++) {
                auto current_node =node_iterator.GetNbrNId(Rnd.GetUniDevInt(node_iterator.GetOutDeg()));
                auto pair = std::make_pair(node, current_node);
                visit[pair].push_back(current_walk_number); // New entry is created if doesn't exist
                WalksVV.PutXY(current_walk_number, 0, node);
                WalksVV.PutXY(current_walk_number, 1, current_node);
                current_walk_number++;
            }
        } else {
            for (auto i = 0; i < NumWalk; i++) {
                WalksVV.PutXY(current_walk_number++, 0, node);
            }
        }
    }
    int64 current_length = 2;
    while (!visit.empty()) {
        for (auto it = visit.cbegin(), next_it = it; it != visit.cend(); it = next_it) {
            auto previous_node = it->first.first;
            auto current_node = it->first.second;
            auto indexes = it->second;
            auto node_iterator = InNet->GetNI(current_node);
            if (node_iterator.GetOutDeg() != 0) {
                auto steps_number = indexes.size();
                std::vector<int64> drawn;
                drawn.reserve(steps_number);
                for (auto i = 0; i < steps_number; i++) {
                    drawn.push_back(node_iterator.GetNbrNId(
                            AliasDrawInt(InNet->GetNDat(current_node).GetDat(previous_node), Rnd)));
                }
                update_step(drawn, current_node, WalksVV, visit_next, indexes, WalkLen, current_length);
            }
            ++next_it;
            visit.erase(it);
        }
        current_length++;
        std::swap(visit, visit_next);
    }

}


//int64 PredictMemoryRequirements(PWNet &InNet) {
//    int64 MemNeeded = 0;
//    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
//        for (int64 i = 0; i < NI.GetOutDeg(); i++) {
//            TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
//            MemNeeded += CurrI.GetOutDeg() * (sizeof(TInt) + sizeof(TFlt));
//        }
//    }
//    return MemNeeded;
//}

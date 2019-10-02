#ifndef RAND_WALK_H
#define RAND_WALK_H

#include <base.h>
#include "network.h"
#include "vector"
#include "unordered_map"
#include "map"
#include <boost/multiprecision/cpp_int.hpp>
#include <immintrin.h>
typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

///Preprocesses transition probabilities for random walks. Has to be called once before SimulateWalk calls
void PreprocessTransitionProbs(PWNet &InNet, const double &ParamP, const double &ParamQ, const bool &verbose);

///Simulates one walk and writes it into Walk vector
void SimulateWalk(PWNet &InNet,
                  TVVec<TInt, uint64> &WalksVV,
                  const std::vector<uint64> &StartNId,
                  const int &WalkLen,
                  const int &NumWalk,
                  TRnd &Rnd,
                  uint64 current_walk_offset,
                  std::vector<boost::multiprecision::uint512_t> &previous_nodes,
                  std::vector<boost::multiprecision::uint512_t> &current_nodes,
                  std::vector<uint64> &saved_step,
                  std::map<int64, int64> &stats,
                  const double &reuse_prob);


///Simulates one walk and writes it into Walk vector
void SimulateWalkReducedBias(PWNet &InNet,
                             TVVec<TInt, uint64> &WalksVV,
                             const std::vector<uint64> &StartNId,
                             const int &WalkLen,
                             const int &NumWalk,
                             TRnd &Rnd,
                             uint64 current_walk_offset,
                             std::vector<boost::multiprecision::uint512_t> &previous_nodes,
                             std::vector<boost::multiprecision::uint512_t> &current_nodes,
                             std::vector<uint64> &saved_step,
                             std::vector<bool> &is_dist_1,
                             std::map<int64, int64> &stats,
                             const double &reuse_prob);

//Predicts approximate memory required for preprocessing the graph
uint64 PredictMemoryRequirements(PWNet &InNet);

#endif //RAND_WALK_H

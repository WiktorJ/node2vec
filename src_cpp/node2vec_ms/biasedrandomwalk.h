#ifndef RAND_WALK_H
#define RAND_WALK_H
#include <base.h>
#include "network.h"
#include "vector"
typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

///Preprocesses transition probabilities for random walks. Has to be called once before SimulateWalk calls
void PreprocessTransitionProbs(PWNet& InNet, const double& ParamP, const double& ParamQ, const bool& verbose);

///Simulates one walk and writes it into Walk vector
TIntV SimulateWalk(PWNet& InNet, std::vector<int64> StartNId, const int& WalkLen, const int& NumWalk, TRnd& Rnd);

//Predicts approximate memory required for preprocessing the graph
int64 PredictMemoryRequirements(PWNet& InNet);

#endif //RAND_WALK_H

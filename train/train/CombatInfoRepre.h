#include <BWAPI.h>
#include <BWAPI/Client.h>
#include <BWTA.h>
#include <vector>

#define DIM_DIRECTION			(8)
#define SIGHT_RANGE				(256)



std::vector<double> GetUnitInfoRepresent(BWAPI::Unit own);
std::vector<double> GetNNInput(std::vector<double> curUnitInfoRepr, std::vector<double> lastUnitInfoRepr, size_t lastAct);
double DistanceNormalized(double d);
double DistanceTerrainNormalized(double d);
double GetUnitDistance(BWAPI::Unit own, BWAPI::Unit target);
double GetTerrainDistance(BWAPI::Unit own, size_t direction);
size_t Direction2Index(int x, int y);
size_t Direction2Index(BWAPI::Position v);
BWAPI::Point<double> Index2Direction(size_t i);
BWAPI::Unit GetAttackEnemyUnit(BWAPI::Unit own);
double RewardAttack(BWAPI::Unit own, double last_own_hitpoint, int last_cool_down);
double RewardDestroy(BWAPI::Unit own);
double RewardMove(BWAPI::Unit own, std::vector<double>& last_state, const size_t& last_action);
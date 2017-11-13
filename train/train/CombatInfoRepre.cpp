#include "CombatInfoRepre.h"
#include <cmath>

#define _USE_MATH_DEFINES
#include <math.h>


std::vector<double> GetUnitInfoRepresent(BWAPI::Unit own)
{
	// [0]
	double cool_down = own->getGroundWeaponCooldown(); 
	double cool_down_info = cool_down / own->getType().groundWeapon().damageCooldown(); 
	// [1]
	double hitpoint = own->getHitPoints(); 
	double hitpoint_info = hitpoint / own->getType().maxHitPoints(); 
	// [2-9]
	std::vector<double> owns_sum_info(DIM_DIRECTION, 0.0);
	for (BWAPI::Unit ou : BWAPI::Broodwar->self()->getUnits())
	{
		if (ou->getID() == own->getID() || !ou->exists())
			continue;
		size_t ind = Direction2Index(ou->getPosition() - own->getPosition());
		owns_sum_info[ind] += DistanceNormalized(GetUnitDistance(own, ou));
	}
	// [10-17]
	std::vector<double> owns_max_info(DIM_DIRECTION, 0.0); 
	for (BWAPI::Unit ou : BWAPI::Broodwar->self()->getUnits())
	{
		if (ou->getID() == own->getID() || !ou->exists())
			continue;
		size_t ind = Direction2Index(ou->getPosition() - own->getPosition());
		owns_max_info[ind] = std::max(owns_max_info[ind], DistanceNormalized(GetUnitDistance(own, ou)));
	}
	// [18-25]
	std::vector<double> enemys_sum_info(DIM_DIRECTION, 0.0);
	for (BWAPI::Unit eu : BWAPI::Broodwar->enemy()->getUnits())
	{
		if (!eu->exists())
			continue;
		size_t ind = Direction2Index(eu->getPosition() - own->getPosition());
		enemys_sum_info[ind] += DistanceNormalized(GetUnitDistance(own, eu));
	}
	// [26-33]
	std::vector<double> enemys_max_info(DIM_DIRECTION, 0.0);
	for (BWAPI::Unit eu : BWAPI::Broodwar->enemy()->getUnits())
	{
		if (!eu->exists())
			continue;
		size_t ind = Direction2Index(eu->getPosition() - own->getPosition());
		enemys_max_info[ind] = std::max(enemys_max_info[ind], DistanceNormalized(GetUnitDistance(own, eu)));
	}
	// [34-41]
	std::vector<double> terrains_info(DIM_DIRECTION, 0.0);
	for (size_t i = 0; i < DIM_DIRECTION; i++)
		terrains_info[i] = DistanceTerrainNormalized(GetTerrainDistance(own, i)); 
	// 0 cooldown, 1 hitpoint, 2-9 own_sum_info, 10-17 own_max_info, 18-25 enemy_sum_info, 26-33 enemy_max_info, 34-41 terrain_info
	std::vector<double> res(1,cool_down_info);
	res.push_back(hitpoint_info);
	res.insert(res.end(), owns_sum_info.begin(), owns_sum_info.end());
	res.insert(res.end(), owns_max_info.begin(), owns_max_info.end());
	res.insert(res.end(), enemys_sum_info.begin(), enemys_sum_info.end());
	res.insert(res.end(), enemys_max_info.begin(), enemys_max_info.end());
	res.insert(res.end(), terrains_info.begin(), terrains_info.end());
	return res;
}

std::vector<double> GetNNInput(std::vector<double> curUnitInfoRepr, std::vector<double> lastUnitInfoRepr, size_t lastAct)
{
	std::vector<double> res;
	res.insert(res.end(), curUnitInfoRepr.begin(), curUnitInfoRepr.end());
	res.insert(res.end(), lastUnitInfoRepr.begin(), lastUnitInfoRepr.end());
	res.insert(res.end(), DIM_DIRECTION + 1, 0.0);
	*(res.end() - (DIM_DIRECTION + 1 - lastAct)) = 1; 
	return res;
}

double DistanceNormalized(double d)
{
	double res; 
	if (d < 0)
		res = -9999;
	else if (d > SIGHT_RANGE)
		res = 0.05;
	else
		res = 1.0 - 0.95 / SIGHT_RANGE*d; 
	return res; 
}

double DistanceTerrainNormalized(double d)
{
	double res;
	if (d < 0)
		res = -9999;
	else if (d > SIGHT_RANGE)
		res = 0.0;
	else
		res = 1.0 - 1.0 / SIGHT_RANGE*d;
	return res;
}

double GetUnitDistance(BWAPI::Unit own, BWAPI::Unit target)
{
	return own->getDistance(target);
}

double GetTerrainDistance(BWAPI::Unit own, size_t direction)
{
	double d = 0;
	BWAPI::Point<double> dir = Index2Direction(direction);
	while (d < SIGHT_RANGE)
	{
		BWAPI::WalkPosition wpos(own->getPosition() + BWAPI::Position(d*dir.x, d*dir.y)); 
		if (!BWAPI::Broodwar->isWalkable(wpos))
			return d;
		else
			d += 20;
	}
	return d;
}

size_t Direction2Index(int x, int y)
{
	double a = atan2(y, x);
	a = 2.0*M_PI + a + M_PI / DIM_DIRECTION;
	a = fmod(a, 2.0*M_PI);
	return int(a / (2.0*M_PI / DIM_DIRECTION));
}

size_t Direction2Index(BWAPI::Position v)
{
	return Direction2Index(v.x, v.y);
}

BWAPI::Point<double> Index2Direction(size_t i)
{
	double a = 2.0*M_PI / DIM_DIRECTION*i;
	return BWAPI::Point<double>(cos(a), sin(a)); 
}

/** select attack enemy target
* now we only consider the lowest hitpoint enemy
* TODO: more efficient attack mode */
BWAPI::Unit GetAttackEnemyUnit(BWAPI::Unit own)
{
	// availabel enemys
	BWAPI::Unitset& enemysBeAttackedGround = own->getUnitsInWeaponRange(own->getType().groundWeapon(), BWAPI::Filter::IsEnemy && BWAPI::Filter::Exists);
	BWAPI::Unitset& enemysBeAttackedAir = own->getUnitsInWeaponRange(own->getType().airWeapon(), BWAPI::Filter::IsEnemy && BWAPI::Filter::Exists);
	if (enemysBeAttackedGround.size() <= 0 && enemysBeAttackedAir.size() <= 0)
		return nullptr;

	BWAPI::Unit bestEnemy = nullptr;
	double minHitPoints = 10000;
	for (auto eu : enemysBeAttackedGround)
	{
		if (minHitPoints > eu->getHitPoints() + eu->getShields())
		{
			minHitPoints = eu->getHitPoints() + eu->getShields();
			bestEnemy = eu;
		}
	}
	for (auto eu : enemysBeAttackedAir)
	{
		if (minHitPoints > eu->getHitPoints())
		{
			minHitPoints = eu->getHitPoints();
			bestEnemy = eu;
		}
	}
	return bestEnemy;
}

double RewardAttack(BWAPI::Unit own, double last_own_hitpoint, int last_cool_down)
{
	double r = 0;
	if (last_cool_down < own->getGroundWeaponCooldown())
		r += own->getType().groundWeapon().damageAmount()*own->getType().groundWeapon().damageFactor();  // 12
	if (own->getHitPoints() + own->getShields() < last_own_hitpoint)
		r -= 2.5*(last_own_hitpoint - own->getHitPoints() - own->getShields());
	    //r -= (last_own_hitpoint - own->getHitPoints() - own->getShields());
	return r / 10; 
}

double RewardDestroy(BWAPI::Unit own)
{
	return -10; 
}

double RewardMove(BWAPI::Unit own, std::vector<double>& last_state, const size_t& last_action)
{
	if (BWAPI::Broodwar->enemy()->getUnits().empty())
	{
		if (!own->getUnitsInRadius(SIGHT_RANGE, BWAPI::Filter::IsOwned).empty())
			return 0;
		std::vector<double> last_own_max_info(last_state.begin() + 10, last_state.begin() + 10 + DIM_DIRECTION);
		if (last_action == DIM_DIRECTION || last_own_max_info[last_action] < (-0.5 + 0.01))
			return -0.5; // -0.1;
		else
			return 0;
	}
	else
	{
		if (own->getUnitsInRadius(own->getType().groundWeapon().maxRange(), BWAPI::Filter::IsEnemy).empty())
		{
			std::vector<double> last_enemy_max_info(last_state.begin() + 26, last_state.begin() + 26 + DIM_DIRECTION);
			if (last_action == DIM_DIRECTION || last_enemy_max_info[last_action] < (-0.5 + 0.01))
				return -0.5; // -0.1;
			else
				return 0; 
		}
	}
	return 0;
}
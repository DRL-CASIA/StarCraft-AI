#include <BWAPI.h>
#include <BWAPI/Client.h>
#include <BWTA.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <algorithm>
#include <vector>
#include <ctime>
#include <fstream>
#include <iterator> 
#include <map>
#include "CombatNN.h"
#include "CombatRL.h"
#include "CombatInfoRepre.h"

using namespace BWAPI;

void DrawAction(const std::map<int, size_t>& set_last_actions, const std::map<int, BWAPI::PositionOrUnit>& set_last_action_targets);
void drawExtendedInterface();

bool flag_draw;

/** reconnect if connection is broken */
void reconnect()
{
  while(!BWAPIClient.connect())
  {
    std::this_thread::sleep_for(std::chrono::milliseconds{ 1000 });
  }
}

std::vector<double> operator-(const std::vector<double>& a, const double& b)
{
	std::vector<double> res(a);
	for (size_t i = 0; i < res.size(); i++)
		res[i] -= b;
	return res;
}

/** main function entry */
int main(int argc, const char* argv[])
{
	srand((unsigned)time(0));
	int dim_input = 42 * 2 + 9;
	int dim_hidden = 100;
	int dim_output = DIM_DIRECTION+1;
	CombatNN net(dim_input, dim_hidden, dim_output);
	net.Reset();
	//
	double gamma = 0.9;
	double lambda = 0.8;
	double alpha = 0.001; 
	double epsilon = 0.0;
	//
	int gameSpeed = 200; 
	// record episodes
	int episode_num = 0;
	flag_draw = true;
	// 
	char file_nn_str[100];
	sprintf_s(file_nn_str, "starcraft_combat_nn_%d_%d_%d.db", dim_input, dim_hidden, dim_output); 
	net.Load(file_nn_str);
	// 
	std::string file_reward_str = "starcraft_combat_reward.txt"; 
	std::ofstream os_reward(file_reward_str, std::ofstream::out | std::ofstream::app); 
	if (!os_reward.is_open())
		std::ferror(NULL); 

	std::string file_outcome_str = "starcraft_combat_outcome.txt";
	std::ofstream os_outcome(file_outcome_str, std::ofstream::out | std::ofstream::app);
	if (!os_outcome.is_open())
		std::ferror(NULL);

	// record win/lose episode
	int record_win = 0;
	int record_lose = 0;

	std::cout << "Connecting..." << std::endl;;
	reconnect();
	while(true)
	{
		std::cout << "waiting to enter match" << std::endl;
		while ( !Broodwar->isInGame() )
		{
			BWAPI::BWAPIClient.update();
			if (!BWAPI::BWAPIClient.isConnected())
			{
			std::cout << "Reconnecting..." << std::endl;;
			reconnect();
			}
		}
		std::cout << "starting match!" << std::endl;
		Broodwar->sendText("Hello world!");
		Broodwar << "The map is " << Broodwar->mapName() << std::endl;
		// Enable some cheat flags
		Broodwar->enableFlag(Flag::UserInput);
		// Uncomment to enable complete map information
		//Broodwar->enableFlag(Flag::CompleteMapInformation);
		Broodwar->setCommandOptimizationLevel(0); //0--3
		episode_num++;
		int lastFrameCount = -1;

		Broodwar->setLocalSpeed(gameSpeed);
		// last step own actions and targets
		std::map<int, size_t> set_last_actions; 
		std::map<int, BWAPI::PositionOrUnit> set_last_action_targets; 
		// last step own weapon cool down statue
		std::map<int, int> set_last_cool_downs; 
		// last step own hitpoints
		std::map<int, double> set_last_own_hitpoints; 
		// last step own Q value
		std::map<int, double> set_last_Qs;
		// last step temperal difference
		std::map<int, double> set_temperal_diffs;
		// count how may own units live
		int num_agent = 0;
		// store rl agent 
		std::map<int, CombatRL> set_rl_agents; 
		// record agents sum of rewards
		std::map<int, double> set_record_rewards;
		// record agents life time
		std::map<int, int> set_record_steps; 
		// record agents sum of Q value
		std::map<int, double> set_Q_values;
		// last step own units state and action
		std::map<int, std::vector<double> > set_last_states; 

		while(Broodwar->isInGame())
		{
			for(auto &e : Broodwar->getEvents())
			{
				switch(e.getType())
				{
				// OnFrame case
				case EventType::MatchFrame:
				{
					if (BWAPI::Broodwar->getFrameCount() > 10000)
					{
						BWAPI::Broodwar->restartGame();
						break;
					}
					BWAPI::Broodwar->drawTextScreen(10, 10, "%d -th frame", BWAPI::Broodwar->getFrameCount());
					drawExtendedInterface();
					DrawAction(set_last_actions, set_last_action_targets);
					BWAPI::Bulletset bs = BWAPI::Broodwar->getBullets();
					for (auto& b : BWAPI::Broodwar->getBullets())
					{
						if (!b->exists())
							continue; 
						if (b->getSource() == nullptr || !b->getSource()->exists())
							continue; 
						if (b->getTarget() == nullptr || !b->getTarget()->exists())
							continue; 
						BWAPI::Broodwar->drawLineMap(b->getSource()->getPosition(), b->getTarget()->getPosition(), BWAPI::Colors::Red); 
					}
					if (lastFrameCount >= 0 && Broodwar->getFrameCount() - lastFrameCount < 10)
						continue;
					BWAPI::Broodwar->drawBoxScreen(10, 10, 70, 25, BWAPI::Colors::Green);
					// generate, execute, and draw own units actions
					lastFrameCount = Broodwar->getFrameCount();
					// clear record data
					num_agent = 0;
					// let's do new step
					for (auto &u : Broodwar->self()->getUnits())
					{
						num_agent++;
						if (set_record_steps.count(u->getID()) <= 0)
							set_record_steps[u->getID()] = 0;
						set_record_steps[u->getID()]++;
						std::vector<double> state = GetUnitInfoRepresent(u) - 0.5;
						if (set_last_states.count(u->getID()) <= 0)
							set_last_states.insert(std::pair<int, std::vector<double> >(u->getID(), state));
						if (set_last_actions.count(u->getID()) <= 0)
							set_last_actions.insert(std::pair<int, size_t>(u->getID(), DIM_DIRECTION));
						std::vector<double> nn_input(state);
						nn_input.insert(nn_input.end(), set_last_states[u->getID()].begin(), set_last_states[u->getID()].end());
						std::vector<double> action_input(DIM_DIRECTION + 1, 0.0);
						action_input[set_last_actions[u->getID()]] = 1;
						nn_input.insert(nn_input.end(), action_input.begin(), action_input.end());
						if (set_rl_agents.count(u->getID()) <= 0)
							set_rl_agents.insert(std::pair<int, CombatRL>(u->getID(), CombatRL(&net)));
						size_t action = set_rl_agents[u->getID()].GetAction(nn_input, epsilon);
						// if no move direction, hold position
						if (action == DIM_DIRECTION)
						{
							BWAPI::Unit target = GetAttackEnemyUnit(u); 
							if (target != nullptr 
								&& set_last_actions[u->getID()] == DIM_DIRECTION
								&& set_last_action_targets.count(u->getID()) > 0
								&& set_last_action_targets[u->getID()].isUnit()
								&& set_last_action_targets[u->getID()].getUnit()->getID() == target->getID())
							{ }
							else
							{
								if (target != nullptr)
								{
									u->stop();
									u->attack(target, true);
									set_last_action_targets[u->getID()] = target;
								}
								else
								{
									u->holdPosition(); 
									set_last_action_targets[u->getID()] = BWAPI::Position(-1, -1); 
								}
							}
						}
						// move to certain direction
						else
						{
							BWAPI::Point<double> dir = Index2Direction(action);
//							target = u->getPosition() + BWAPI::Position(dir.x * 48, dir.y * 48);
//							BWAPI::Position target = u->getPosition() + BWAPI::Position(dir.x * 56, dir.y * 56);
							BWAPI::Position target = u->getPosition() + BWAPI::Position(dir.x * 64, dir.y * 64);
//							BWAPI::Position target = u->getPosition() + BWAPI::Position(dir.x * 120, dir.y * 120);
							u->rightClick(target);
							set_last_action_targets[u->getID()] = target; 
						}
						set_last_actions[u->getID()] = action; 
						set_last_states[u->getID()] = state; 
						set_last_cool_downs[u->getID()] = u->getGroundWeaponCooldown(); 
						set_last_own_hitpoints[u->getID()] = u->getHitPoints() + u->getShields();
						set_last_Qs[u->getID()] = set_rl_agents[u->getID()].GetQValue(nn_input, action);
						set_rl_agents[u->getID()].UpdateEligibility(action, lambda, gamma); 
					}
					break;
				}
				case EventType::MatchEnd:
					if (e.isWinner())
					{
						Broodwar << "I won the game" << std::endl;
						record_win++;
					}
					else
					{
						Broodwar << "I lost the game" << std::endl;
						record_lose++;
					}
					break;
				case EventType::SendText:
					if (e.getText() == "s")
						flag_draw = !flag_draw;
					else if (e.getText() == "+")
					{
						gameSpeed -= 10;
						Broodwar->setLocalSpeed(gameSpeed);
					}
					else if (e.getText() == "-")
					{
						gameSpeed += 10;
						Broodwar->setLocalSpeed(gameSpeed);
					}
					else if (e.getText() == "++")
					{
						gameSpeed = 0;
						Broodwar->setLocalSpeed(gameSpeed);
					}
					else if (e.getText() == "--")
					{
						gameSpeed = 42;
						Broodwar->setLocalSpeed(gameSpeed);
					}
					else
						Broodwar << "You typed \"" << e.getText() << "\"!" << std::endl;
					break;
				case EventType::ReceiveText:
					Broodwar << e.getPlayer()->getName() << " said \"" << e.getText() << "\"" << std::endl;
					break;
				case EventType::PlayerLeft:
					Broodwar << e.getPlayer()->getName() << " left the game." << std::endl;
					break;
				case EventType::NukeDetect:
					break;
				case EventType::UnitCreate:
					break;
				case EventType::UnitDestroy:			
					break;
				case EventType::UnitMorph:
					break;
				case EventType::UnitShow:
					break;
				case EventType::UnitHide:
					break;
				case EventType::UnitRenegade:
					break;
				case EventType::SaveGame:
					Broodwar->sendText("The game was saved to \"%s\".", e.getText().c_str());
					break;
				}
			}

			Broodwar->drawTextScreen(300,0,"FPS: %f",Broodwar->getAverageFPS());
			 
			BWAPI::BWAPIClient.update();
			if (!BWAPI::BWAPIClient.isConnected())
			{
			std::cout << "Reconnecting..." << std::endl;
			reconnect();
			}
		}
		

	}
	std::cout << "Press ENTER to continue..." << std::endl;
	std::cin.ignore();
	return 0;
}


void DrawAction(const std::map<int, size_t>& set_last_actions, const std::map<int, BWAPI::PositionOrUnit>& set_last_action_targets)
{
	for (auto& u : BWAPI::Broodwar->self()->getUnits())
	{
		if (!u->exists())
			continue;
		if (set_last_actions.count(u->getID()) <= 0 || set_last_action_targets.count(u->getID()) <= 0)
			continue;
		if (set_last_actions.find(u->getID())->second == DIM_DIRECTION)
		{
			BWAPI::Broodwar->drawCircleMap(u->getPosition(), 10, BWAPI::Colors::Blue);
			const BWAPI::PositionOrUnit& target = set_last_action_targets.find(u->getID())->second;
			if (target.isUnit() && target.getUnit()->exists())
				BWAPI::Broodwar->drawLineMap(u->getPosition(), target.getUnit()->getPosition(), BWAPI::Colors::Orange); 
		}
		else
			BWAPI::Broodwar->drawLineMap(u->getPosition(), set_last_action_targets.find(u->getID())->second.getPosition(), BWAPI::Colors::White); 
	}

}

/** draw health bar */
void drawExtendedInterface()
{
	int verticalOffset = -10;

	// draw enemy units
	for (auto & ui : Broodwar->enemy()->getUnits())
	{
		BWAPI::UnitType type(ui->getType());
		int hitPoints = ui->getHitPoints();
		int shields = ui->getShields();

		const BWAPI::Position & pos = ui->getPosition();

		int left = pos.x - type.dimensionLeft();
		int right = pos.x + type.dimensionRight();
		int top = pos.y - type.dimensionUp();
		int bottom = pos.y + type.dimensionDown();

		if (!BWAPI::Broodwar->isVisible(BWAPI::TilePosition(ui->getPosition())))
		{
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, top), BWAPI::Position(right, bottom), BWAPI::Colors::Grey, false);
			BWAPI::Broodwar->drawTextMap(BWAPI::Position(left + 3, top + 4), "%s", type.getName().c_str());
		}

		if (!type.isResourceContainer() && type.maxHitPoints() > 0)
		{
			double hpRatio = (double)hitPoints / (double)type.maxHitPoints();

			BWAPI::Color hpColor = BWAPI::Colors::Green;
			if (hpRatio < 0.66) hpColor = BWAPI::Colors::Orange;
			if (hpRatio < 0.33) hpColor = BWAPI::Colors::Red;

			int ratioRight = left + (int)((right - left) * hpRatio);
			int hpTop = top + verticalOffset;
			int hpBottom = top + 4 + verticalOffset;

			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Grey, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(ratioRight, hpBottom), hpColor, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Black, false);

			int ticWidth = 3;

			for (int i(left); i < right - 1; i += ticWidth)
			{
				BWAPI::Broodwar->drawLineMap(BWAPI::Position(i, hpTop), BWAPI::Position(i, hpBottom), BWAPI::Colors::Black);
			}
		}

		if (!type.isResourceContainer() && type.maxShields() > 0)
		{
			double shieldRatio = (double)shields / (double)type.maxShields();

			int ratioRight = left + (int)((right - left) * shieldRatio);
			int hpTop = top - 3 + verticalOffset;
			int hpBottom = top + 1 + verticalOffset;

			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Grey, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(ratioRight, hpBottom), BWAPI::Colors::Blue, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Black, false);

			int ticWidth = 3;

			for (int i(left); i < right - 1; i += ticWidth)
			{
				BWAPI::Broodwar->drawLineMap(BWAPI::Position(i, hpTop), BWAPI::Position(i, hpBottom), BWAPI::Colors::Black);
			}
		}

	}

	// draw neutral units and our units
	for (auto & unit : BWAPI::Broodwar->getAllUnits())
	{
		if (unit->getPlayer() == BWAPI::Broodwar->enemy())
		{
			continue;
		}

		const BWAPI::Position & pos = unit->getPosition();

		int left = pos.x - unit->getType().dimensionLeft();
		int right = pos.x + unit->getType().dimensionRight();
		int top = pos.y - unit->getType().dimensionUp();
		int bottom = pos.y + unit->getType().dimensionDown();

		//BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, top), BWAPI::Position(right, bottom), BWAPI::Colors::Grey, false);

		if (!unit->getType().isResourceContainer() && unit->getType().maxHitPoints() > 0)
		{
			double hpRatio = (double)unit->getHitPoints() / (double)unit->getType().maxHitPoints();

			BWAPI::Color hpColor = BWAPI::Colors::Green;
			if (hpRatio < 0.66) hpColor = BWAPI::Colors::Orange;
			if (hpRatio < 0.33) hpColor = BWAPI::Colors::Red;

			int ratioRight = left + (int)((right - left) * hpRatio);
			int hpTop = top + verticalOffset;
			int hpBottom = top + 4 + verticalOffset;

			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Grey, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(ratioRight, hpBottom), hpColor, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Black, false);

			int ticWidth = 3;

			for (int i(left); i < right - 1; i += ticWidth)
			{
				BWAPI::Broodwar->drawLineMap(BWAPI::Position(i, hpTop), BWAPI::Position(i, hpBottom), BWAPI::Colors::Black);
			}
		}

		if (!unit->getType().isResourceContainer() && unit->getType().maxShields() > 0)
		{
			double shieldRatio = (double)unit->getShields() / (double)unit->getType().maxShields();

			int ratioRight = left + (int)((right - left) * shieldRatio);
			int hpTop = top - 3 + verticalOffset;
			int hpBottom = top + 1 + verticalOffset;

			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Grey, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(ratioRight, hpBottom), BWAPI::Colors::Blue, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Black, false);

			int ticWidth = 3;

			for (int i(left); i < right - 1; i += ticWidth)
			{
				BWAPI::Broodwar->drawLineMap(BWAPI::Position(i, hpTop), BWAPI::Position(i, hpBottom), BWAPI::Colors::Black);
			}
		}

		if (unit->getType().isResourceContainer() && unit->getInitialResources() > 0)
		{

			double mineralRatio = (double)unit->getResources() / (double)unit->getInitialResources();

			int ratioRight = left + (int)((right - left) * mineralRatio);
			int hpTop = top + verticalOffset;
			int hpBottom = top + 4 + verticalOffset;

			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Grey, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(ratioRight, hpBottom), BWAPI::Colors::Cyan, true);
			BWAPI::Broodwar->drawBoxMap(BWAPI::Position(left, hpTop), BWAPI::Position(right, hpBottom), BWAPI::Colors::Black, false);

			int ticWidth = 3;

			for (int i(left); i < right - 1; i += ticWidth)
			{
				BWAPI::Broodwar->drawLineMap(BWAPI::Position(i, hpTop), BWAPI::Position(i, hpBottom), BWAPI::Colors::Black);
			}
		}
	}
}


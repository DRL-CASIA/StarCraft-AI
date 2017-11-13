# StarCraft-AI
use reinforcement learning and transfer learning to learn policy for starcraft micromanagement

# Introduction
presents a reinforcement learning model and transfer learning method to tackle the multi-agent control of StarCraft micromanagement. 

define an efficient state representation approach to break down the complexity caused by the large state and action space in game environment. 

propose a reward function to help balance units’ move and attack, and encourage units to learn cooperative behaviors. 

present a multi-agent gradient-descent Sarsa(λ) reinforcement learning model to train our units to operate micromanagement, and use a neural network as function approximator of the Q values. 

In small scale scenarios, our units can successfully learn to combat and defeat the built-in AI with 100% winning rate. 

propose a transfer learning method to extend our model to larger scale and more difficult scenarios. 

transfer learning method can accelerate the training process and improve the learning performance. 

the test episodes show our controlled units are able to operate StarCraft micromanagement with appropriate strategies in various scenarios.

## Network Input
The network input is composed of 3 parts

---- 1. Current step agent state informatoin [dim=42]:

-------- own weapon cool down time [0~1], dim=1

-------- own hitpoint [0~1], dim=1

-------- self units distance in 8 directions; multiple distances are SUM in the same direction if existed; if no units in certain direction, it is 0; if unit is out of sight range, it is 0.05; dim=8.

-------- self units distance in 8 directions; multiple distances are MAXIMIZE in the same direction if existed; if no units in certain direction, it is 0; if unit is out of sight range, it is 0.05; dim=8

-------- enemy units distance in 8 directions; multiple distances are SUM in the same direction if existed; blablabla

-------- enemy units distance in 8 directions; multiple distances are MAXIMIZE in the same direction if existed; blablabla

-------- terrain/obstacle distance in 8 direction; if it is out of sight range, 0; dim=8

---- 2. Last step agent state information [dim=42]

-------- same as current step 

---- 3. Last step action [dim=9]

-------- Last step action input node is set to 1, other action input nodes are set to 0; dim=9

## Train/Test
use train project to train; every 100 episodes, the network parameters are saved in db file;

use test project to test; rename the saved db file to "starcraft_combat_nn_#(num_input)_#(num_hidden)_#(num_output).db" in the project folder, so the project can load the learn network 

# Install
For details, you can refer to UAlbertaBot on how to setup the environment (https://github.com/davechurchill/ualbertabot/wiki) 

## StarCraft
you first need to install StarCraft:Broodwar 1.16.1 to run starcraft

## BWAPI
Download BWAPI library in https://github.com/bwapi/bwapi

Add the BWAPI folder directory to Windows system variable BWAPI_DIR (Property of your Computer--Advance System Set--Environment Variable--System Variable--New)

Here we use client mode to connect our program with starcraft

you can also use module mode to connect starcraft, but it is not handy for debug and repeat the game which is highly needed in RL algorithms

## BWTA
Download BWTA library in https://bitbucket.org/auriarte/bwta2/wiki/Home

Add BWTA foler directory to Windows system variable BWTA_DIR 

In this projects, I didn't use BWTA library, since the walkable of terrain can be checked by BWAPI library

If you are designing complicated path finding algorithm, I think BWTA may be useful

## Project
now you can open train/test project and compile

remember to add BWAPI/BWTA to your project: 

---- Project Property->C++>General->Additional Include Directories：$(BWAPI)/include and $(BWTA)/include

---- Project Property->Link->Input->Additional Dependency: 

$(BWAPI_DIR)/lib/BWAPId.lib
$(BWAPI_DIR)/lib/BWAPIClientd.lib
$(BWTA_DIR)/lib/BWTAd.lib
$(BWTA_DIR)/lib/libboost_system-vc120-mt-gd-1_56.lib
$(BWTA_DIR)/lib/libboost_thread-vc120-mt-gd-1_56.lib
$(BWTA_DIR)/lib/libCGAL-vc120-mt-gd-4.4.lib
$(BWTA_DIR)/lib/libgmp-10.lib
$(BWTA_DIR)/lib/libmpfr-4.lib
$(BWTA_DIR)/lib/libboost_filesystem-vc120-mt-gd-1_56.lib

(In this project, except BWAPId.lib and BWAPIClientd.lib, the rest are unneccesary)

(if you compile in RELEASE mode, change libs into release lib, like BWAPId.lib->BWAPI.lib)

## Chaoslauncher 
run Chaoslauncher.exe to start starcraft, check BWAPI Injector [DEBUG], Chaosplugin for 1.16.1, W-MODE 1.02 

config BWAPI Injector [DEBUG], open bwapi.ini, set ai, ai_dbg=null, auto_restart=ON, map=maps/BroodWar/Single player maps/the_name_of_map.scm, game_type = USE_MAP_SETTINGS, sound = OFF

Now you can see the running



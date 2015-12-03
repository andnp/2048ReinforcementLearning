#include <iostream>
#include <random>
#include <math.h>
#include <thread>

#include "grid.hpp"
#include "../CNeuralNets/FeedforwardNet/brain.h"
using namespace std;

#define LEARN_STEPS 5

vector<int> scores;

vector<double> getGrid(grid g){
	vector<double> ret;
	for(int x = 0; x < g.size(); x++){
		for(int y = 0; y < g.size(); y++){
			vector<double> dat = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
			dat[log2(g.getGrid(x, y))] = 1;
			ret.insert(ret.end(), dat.begin(), dat.end());
		}
	}
	return ret;
}

unsigned int moves = 0;
unsigned int game = 0;

int main(){
	brain b;
	b.initialize(260);
	while(moves < 1000000000){
		grid g;
		while(g.can_move()){
			int start_score = g.score();
			vector<double> inputs = getGrid(g);
			int action = b.forward(inputs);

			switch(action){
				case 0:
					g.action(direction::NORTH);
					break;
				case 1:
					g.action(direction::EAST);
					break;
				case 2:
					g.action(direction::SOUTH);
					break;
				case 3:
					g.action(direction::WEST);
					break;
				case -1:
					cout << "error";
					return 0;
			}

			int diff_score = g.score() - start_score;
			double reward = diff_score;
			b.backward(reward);
			moves++;
		}
		game++;
		scores.push_back(g.score());
		unsigned int avg = 0;
		unsigned int rollAvg = 0;
		for(int i = 0; i < scores.size(); i++){
			avg += scores[i];
			if(i + 20 > scores.size()){
				rollAvg += scores[i];
			}
		}
		cout << "game: " << game << " score: " << g.score() << " moves: " << moves << " buffer: " << b.experienceBuffer.size() << " avg: " << avg/scores.size() << " roll: " << rollAvg/20 << "\n";
	}

	return 0;
}
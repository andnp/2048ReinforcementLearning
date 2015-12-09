#include <iostream>
#include <random>
#include <math.h>
#include <thread>
// #include <tbb/concurrent_vector.h>
#include <mutex>

#include "grid.hpp"
#include "./CPPReinforcementLearning/FeedforwardNet/brain.h"
using namespace std;

#define THREADS 4
#define LEARN_STEPS THREADS * 2

mutex elock;

int high = 0;


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


vector<brain::experience> experiences;


void push(brain::experience e){
	elock.lock();
	experiences.push_back(e);
	elock.unlock();
}

random_device rd;
mt19937 gen(rd());	


void run(string thread, brain b){
	vector<int> scores;
	unsigned int moves = 0;
	unsigned int game = 0;
	while(moves < 1000000000){
		grid g;
		while(g.can_move()){
			brain::experience e;
			int start_score = g.score();
			vector<double> inputs = getGrid(g);
			e.state0 = inputs;
			int action = b.forward(inputs);
			e.action0 = action;

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
			}

			int diff_score = g.score() - start_score;
			double reward = diff_score;
			e.reward0 = reward;
			e.state1 = getGrid(g);
			b.backward(reward);
			moves++;
			uniform_real_distribution<> dist(0,experiences.size() - 1);
			if(experiences.size() > 10000000){
				experiences[dist(gen)] = e;
			} else {
				push(e);
			}
			if(experiences.size() > 100000){
				for(int i = 0; i < LEARN_STEPS; i++){
					b.learn(experiences[dist(gen)]);
				}
			}
		}
		game++;
		if(g.score() > high)
			high = g.score();
		scores.push_back(g.score());
		unsigned int avg = 0;
		unsigned int rollAvg = 0;
		for(int i = 0; i < scores.size(); i++){
			avg += scores[i];
			if(i + 20 > scores.size()){
				rollAvg += scores[i];
			}
		}
		cout << thread << " game: " << game << " score: " << g.score() << " moves: " << moves << " buffer: " << experiences.size() << " avg: " << avg/scores.size() << " roll: " << rollAvg/20 << " high: " << high <<"\n";
	}

}

int main(){
	experiences.reserve(1000000);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);  
	uniform_int_distribution<> dist(25,100);
	brain b;

	b.burnIn = 250000;
	b.learnSteps = 2500000;
	b.gamma = .75;


	vector<int> layers = {(int)dist(gen), (int)(.5 * dist(gen)), 1};
	vector<int> types = {5, 5, 1};
	vector<double> dropout = {.01, .02, 0};
	vector<double> lambda = {0,0,0};
	b.valueNet.instantiate(260, layers, types, dropout, lambda, .1);
	thread t1(run, "thread 1", b);

	layers = {(int)dist(gen), (int)(.5 * dist(gen)), 1};
	brain b1;
	b1.valueNet.instantiate(260, layers, types, dropout, lambda, .1);
	thread t2(run, "thread 2", b1);

	layers = {(int)dist(gen), (int)(.5 * dist(gen)), 1};
	brain b2;
	b2.valueNet.instantiate(260, layers, types, dropout, lambda, .1);
	thread t3(run, "thread 3", b2);

	layers = {(int)dist(gen), (int)(.5 * dist(gen)), 1};
	brain b3;
	b3.valueNet.instantiate(260, layers, types, dropout, lambda, .1);
	thread t4(run, "thread 4", b3);

	t4.join();
	return 0;
}

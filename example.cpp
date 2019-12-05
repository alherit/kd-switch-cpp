using namespace std;
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>
#include <list>

#include <iostream>
#include <sstream>
#include <fstream>
#include "kds/LogWeightProb.hpp"
#include <chrono>

#include <math.h>

#include "kds/kds.hpp"




int main()
{

	auto start = std::chrono::high_resolution_clock::now();
	size_t dim = 50;
	size_t alphaSize = 2;
	std::mt19937 rnd(12345);
	//KDSTree* kds = new KDSTree(0, rnd, dim, alphaSize, true, NULL);
	//vector<double> theta0 = { .5,.5 };
	vector<double> theta0; //empty
	KDSForest* kds = new KDSForest(50, 12345, dim, alphaSize, false, theta0);

	std::cout << "Tree created" << endl;

	DistT predDist(alphaSize);
	LWPT cumCProb(1.);
	CountT n = 0;
	PointT previous(dim);

	std::normal_distribution<> dnorm{ 0, 1 };
	std::uniform_int_distribution<> dunif(0, 1);

	PointT point(dim);
	LabelT label;

	while (n < 20000) {
		n++;

		for (size_t i = 0; i < dim; i++)
			point[i] = dnorm(rnd);

		label = dunif(rnd);


		kds->predict(predDist, point, false);

		LWPT condProb = predDist[label];

		cumCProb *= condProb;

		kds->update(point, label, false);

		previous = point;
	}


	std::cout << "NLL: " << -cumCProb.getLog2() / (FT)n << endl;

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	delete kds;

}

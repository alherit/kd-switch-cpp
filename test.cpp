

// Test case on breast_cancer dataset
// compile in debug mode to have deterministic cuts
// last line must be: 0.919725  0.441891 for kdw
// last line must be: 0.860516  0.45828  for kds




using namespace std;
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>
#include <list>

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

#include <math.h>

#include "kds/kds.hpp"




int main()
{
	{
		auto start = std::chrono::high_resolution_clock::now();
		//double theta0[2] = { .5,.5 };
		size_t dim = 30;
		size_t alphaSize = 2;
		//std::mt19937 rnd(12345);
		//KDSTree* kds = new KDSTree(0, rnd, dim, alphaSize, true, NULL);
		vector<double> theta0; //empty
		KDSForest* kds = new KDSForest(50,12345,dim, alphaSize, false, theta0);

		std::cout << "Tree created" << endl;

		ifstream fileF("../data/features.csv");
		ifstream fileL("../data/labels.csv");

		std::cout << "After stream creation" << endl;


		string line;

		DistT predDist(alphaSize);
		LWPT cumCProb(1.);
		CountT n = 0;
		PointT previous(dim);
		while (std::getline(fileF, line)) {
			n++;

			// these will be deleted by the tree itself
			LabeledPoint* lp = new LabeledPoint;
			lp->point = PointT(dim);
			lp->label = 0;  //initialize


			std::istringstream s(line);
			std::string field;
			size_t i = 0;
			while (getline(s, field, ',')) {
				lp->point[i] = atof(field.c_str());
				i++;

			}
			//std::cout << lp->point[0] << endl;

			string label;
			std::getline(fileL, label);
			lp->label = atoi(label.c_str());

			kds->predict(predDist, lp->point, false);

			LWPT condProb = predDist[lp->label];
			std::cout << condProb << "  ";

			cumCProb *= condProb;

			std::cout << -cumCProb.getLog2() / (FT)n << endl;

			kds->update(lp->point,lp->label, false);

			previous = lp->point;
		}

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";

		delete kds;
		std::cout << "Sucess" << endl;
	}
}



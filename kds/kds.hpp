#ifndef __KDS_HPP__
#define __KDS_HPP__


/******************************
	  Author: Alix Lheritier
		Date: 2019
******************************/


using namespace std;
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>
#include <list>

#include <iostream>
#include <sstream>
#include <fstream>
#include "LogWeightProb.hpp"
#include <chrono>

#include <math.h>


typedef double FT;
typedef vector<FT> PointT;
typedef size_t LabelT;

struct LabeledPoint {
	PointT point;
	LabelT label;
};


typedef unsigned long CountT;
typedef LogWeightProb<double> LWPT;
typedef list<LabeledPoint*> ItemsT;
typedef vector<CountT> CountersT;
typedef vector<LWPT> DistT;


#define M_PI       3.14159265358979323846
#define M_LNPI log(M_PI)
#define _M_LN2  log(2.0)

LWPT KTEstimator(const CountersT& v)
{
	size_t alpha = v.size();
	FT alpha_2 = (FT)alpha / 2.0;
	FT value = 0;
	FT total = 0;
	CountT occuring = 0;

	for (size_t symbolIter = 0; symbolIter < alpha; symbolIter++)
	{
		total += v[symbolIter];
		if (v[symbolIter] > 0)
		{
			occuring++;
			value -= lgamma(v[symbolIter] + 0.5);
		}

	}

	value -= ((CountT)alpha - occuring) * 0.5 * M_LNPI;

	value -= lgamma(alpha_2);

	value += lgamma(total + alpha_2) + alpha_2 * M_LNPI;

	LWPT prob;
	prob.setLog2(-value / _M_LN2);
	return prob;
}

// result is returned in dist that must be created with the correct number of elements
void seqKTDist(DistT& dist, const CountersT& v) {
	FT virt = 0.5; // number of virtual initial observations for each symbol

	CountT total = 0;

	for (CountersT::const_iterator it = v.begin(); it != v.end(); it++)
		total += *it;

	size_t i = 0;
	for (CountersT::const_iterator it = v.begin(); it != v.end(); it++, i++)
		dist[i] = (LWPT(*it + virt) / ((FT)total + virt * v.size()));

}

static bool deleteAll(LabeledPoint* theElement) { delete theElement; return true; }


class Node {
private:
	size_t projDir;
	FT pivot;
	ItemsT items;
	vector<Node*> children;
	CountersT counts;
	Node* parent;
	LWPT ctProb; // probability of the sequence observed at this node given by CT*
	DistT ctProbNext; // distribution for next symbol computed during predict and then used in update
	DistT probKTNext; // kt distribution for next symbol
	DistT pRec; // recursive case distribution for next symbol
	LWPT wa, wb; //weights for switching

	friend class KDSTree;
	friend class KDSForest;

public:
	Node(size_t _projDir, size_t alphaSize, Node* _parent) :
		projDir(_projDir), pivot(0), counts(alphaSize, 0), parent(_parent), ctProb(1.), ctProbNext(alphaSize), probKTNext(alphaSize), pRec(alphaSize), wa(.5), wb(.5) {
	}

	void setPivot(const PointT& point) {
		pivot = point[projDir];
	}


	size_t selectBranch(const PointT& point) {
		FT proj = point[projDir];

		if (proj <= pivot)
			return 0;
		else
			return 1;
	}



};

typedef vector<Node*> ChildrenT;


class KDSTree {
private:
	size_t alphaSize;
	size_t dim;
	Node* root;
	bool ctw;
	DistT p0dist;
	CountT n; //number of items in the tre

	std::mt19937& gen; //Standard mersenne_twister_engine seeded 
	std::uniform_int_distribution<> dis;

	size_t debug_idx;

	friend class KDSForest;

public:
	KDSTree(size_t _debug_idx, std::mt19937& _gen, size_t _dim, size_t _alphaSize, bool _ctw, const vector<FT>& theta0) : alphaSize(_alphaSize), dim(_dim),
		ctw(_ctw), n(0), gen(_gen), dis(0, dim - 1), debug_idx(_debug_idx) {

		if (theta0.size() > 0) {
			for (size_t i = 0; i < alphaSize; i++)
				p0dist.push_back(LWPT(theta0[i]));
		}

#ifdef _DEBUG
		root = new Node(debug_idx % dim, alphaSize, NULL);
#else
		root = new Node(dis(gen), alphaSize, NULL);
#endif

	}


	void destroy(bool delete_items) {
		Node* save;
		for (Node* it = root; it != NULL; it = save) {
			if (it->children.size() == 0) {
				if (delete_items)
					it->items.remove_if(deleteAll);
				delete it;
				save = NULL;
			}
			else if (it->children[0] == NULL) {
				save = it->children[1];
				if (delete_items)
					it->items.remove_if(deleteAll);
				delete it;
			}
			else {
				// Rotate the left link into a right link
				save = it->children[0];
				if (save->children.size() == 0)
					save->children.resize(2, NULL);

				it->children[0] = save->children[1];
				save->children[1] = it;

			}
		}

	}


	~KDSTree() {
		destroy(true);
	}


	LWPT alpha(size_t n) {
		if (ctw)
			return LWPT(0.);
		else
			return LWPT(1.) / LWPT((double)n);
	}


	//when calling this function, KDSTree becomes the owner of the labeledPoint and thus has the responsibility of freeing it
	void update(LabeledPoint* lpoint_ptr, bool frozen = true) {

		if (frozen) {// predict first
			DistT dummy(alphaSize);
			predict(dummy, lpoint_ptr->point, false); //frozen=false => affects structure
		}

		n++;
		Node* cur = root;
		bool stop = false;

		while (!stop) {
			cur->counts[lpoint_ptr->label]++;

			if (cur->children.empty()) // the split was done before, we need to add the point to the leaf
				cur->items.push_back(lpoint_ptr); //add current point


			// now we know the label. cur.CTprob_next was computed during predict
			cur->ctProb = cur->ctProbNext[lpoint_ptr->label];

			// update weights wa wb
			CountT totalCounts = 0;
			for (size_t i = 0; i < alphaSize; i++)
				totalCounts += cur->counts[i];

			LWPT alpha_n_plus_1 = alpha(totalCounts + 1);

			cur->wa = alpha_n_plus_1 * cur->ctProb + (LWPT(1.) - LWPT(2.) * alpha_n_plus_1) * cur->wa * cur->probKTNext[lpoint_ptr->label];
			cur->wb = alpha_n_plus_1 * cur->ctProb + (LWPT(1.) - LWPT(2.) * alpha_n_plus_1) * cur->wb * cur->pRec[lpoint_ptr->label];

			if (!cur->children.empty())
				cur = cur->children[cur->selectBranch(lpoint_ptr->point)];
			else
				stop = true;
		}
	}

	// returns result in dist. 
	void predict(DistT& dist, const PointT& point, bool frozen = true) {

		Node* cur = root;
		bool stop = false;

#ifdef _DEBUG
		CountT depth = 0;
#endif

		Node* temp = NULL; // to be destroyed at the end, if frozen version

		//going down the tree
		while (!stop) {
			if (cur->children.empty()) {
				if (frozen)
					temp = cur;

				cur->setPivot(point);

#ifdef _DEBUG
				cur->children.push_back(new Node((debug_idx + depth + 1) % dim, alphaSize, cur));
				cur->children.push_back(new Node((debug_idx + depth + 1) % dim, alphaSize, cur));
#else
				cur->children.push_back(new Node(dis(gen), alphaSize, cur));
				cur->children.push_back(new Node(dis(gen), alphaSize, cur));
#endif

				// make all the update steps, notice that we are not yet adding current point, it will be done during update
				for (ItemsT::iterator it = cur->items.begin(); it != cur->items.end(); it++) {
					size_t i = cur->selectBranch((*it)->point);
					cur->children[i]->items.push_back(*it);
					cur->children[i]->counts[(*it)->label]++;
				}


				//initialize children using already existing symbols(Eq. 14)
				//CTPROB is P_s of the sequence observed so far : just KT
				for (ChildrenT::iterator it = cur->children.begin(); it != cur->children.end(); it++) {
					(*it)->ctProb = KTEstimator((*it)->counts);
					(*it)->wa *= (*it)->ctProb;
					(*it)->wb *= (*it)->ctProb;
				}

				if (!frozen)
					cur->items.clear();

				stop = true;
			}
			cur = cur->children[cur->selectBranch(point)];
#ifdef _DEBUG
			depth++;
#endif

		}

		//going up the tree
		DistT childCTprobNext; // child's Ps on sequence including next symbol
		LWPT childCTprob; // child's Ps on sequence without next symbol

		stop = false;
		while (!stop) {

			// save KT dist and the other distributions for the update
			if ((cur == root) && !p0dist.empty()) // known distribution
				cur->probKTNext = p0dist;
			else
				seqKTDist(cur->probKTNext, cur->counts);  //labels in self.items are not used, just counts are used

			if (cur->children.empty())
				cur->pRec = cur->probKTNext;
			else
				for (size_t i = 0; i < alphaSize; i++)
					cur->pRec[i] = childCTprobNext[i] / childCTprob;

			for (size_t i = 0; i < alphaSize; i++)
				cur->ctProbNext[i] = cur->wa * cur->probKTNext[i] + cur->wb * cur->pRec[i];

			if (cur->parent == NULL)
				stop = true;
			else {
				childCTprob = cur->ctProb;
				childCTprobNext = cur->ctProbNext;
				cur = cur->parent;
			}


		}

		for (size_t i = 0; i < alphaSize; i++)
			dist[i] = cur->ctProbNext[i] / cur->ctProb;

		if (temp != NULL) {
			for (ChildrenT::iterator it = temp->children.begin(); it != temp->children.end(); it++) {
				delete* it;
			}
			temp->children.clear();
		}



	}

};



class KDSForest {
private:
	size_t J;
	vector<KDSTree*> trees;
	size_t alphaSize;
	DistT weights;

	std::mt19937 gen; //Standard mersenne_twister_engine seeded 


public:
	KDSForest(size_t _J, int seed, size_t _dim, size_t _alphaSize, bool _ctw, const vector<FT>& theta0) :J(_J), alphaSize(_alphaSize), weights(J, 1. / J), gen(seed) {
		for (size_t j = 0; j < J; j++)
			trees.push_back(new KDSTree(j, gen, _dim, _alphaSize, _ctw, theta0));
	}


	~KDSForest() {
		if (J > 0) {
			for (size_t j = 0; j < J - 1; j++)
				trees[j]->destroy(false);
			trees[J - 1]->destroy(true); //delete items
		}
	}

	void predict(DistT& dist, const PointT& point, bool frozen = true) {

		for (size_t i = 0; i < alphaSize; i++)
			dist[i] = LWPT(0.);

		DistT dist_tree(alphaSize);
		// MIX USING POSTERIOR
		for (size_t j = 0; j < J; j++) {
			trees[j]->predict(dist_tree, point, frozen);

			for (size_t i = 0; i < alphaSize; i++)
				dist[i] = dist[i] + dist_tree[i] * weights[j];

		}

	}

	vector<FT> predict_proba(const PointT& point, bool frozen = true) {
		DistT dist(alphaSize);
		predict(dist, point, frozen);

		vector<FT> res(alphaSize);
		for (size_t i = 0; i < alphaSize; i++)
			res[i] = dist[i].getWeightProb();

		return res;

	}

	vector<FT> predict_log2_proba(const PointT& point, bool frozen = true) {
		DistT dist(alphaSize);
		predict(dist, point, frozen);

		vector<FT> res(alphaSize);
		for (size_t i = 0; i < alphaSize; i++)
			res[i] = dist[i].getLog2();

		return res;

	}


	void update(const PointT& point, const LabelT& label, bool frozen = true) {
		LabeledPoint* lp_ptr = new LabeledPoint;
		lp_ptr->label = label;
		lp_ptr->point = point; // copy it


		for (size_t j = 0; j < J; j++)
			trees[j]->update(lp_ptr, frozen);

		LWPT acc(0.);
		for (size_t j = 0; j < J; j++) {
			weights[j] = trees[j]->root->ctProb * LWPT(1. / J);
			acc += weights[j];
		}

		for (size_t j = 0; j < J; j++)
			weights[j] /= acc;

	}


};

#endif // __KDS_HPP__
#include<Eigen\Dense>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<ctime>
#include<limits>
#include<random>
#include<chrono>
#include<algorithm>
#include<cmath>
#include<thread>
#include<ratio>
extern "C" {
	#include "netcdf.h"
}

using namespace std;

double sampling_time = 0.0;
double svd_time = 0.0;
random_device rd;
int seed = rd();
mt19937 mt(seed);

//computes the mean of a sample given as vector of doubles
double compute_mean(vector<double> sample) {
	double mean = 0.0;
	int n = sample.size();
	for (int i = 0; i < n; i++) {
		mean += sample[i];
	}
	mean /= n;
	return mean;
}

//using the mean of a sample, it computes the sample variance
double compute_variance(vector<double> sample, double mean) {
	int n = sample.size();
	double variance = 0.0;
	for (int i = 0; i < n; i++) {
		variance += pow(sample[i] - mean, 2);
	}
	variance /= n;
	return variance;
}

//for a vector of indices, being sorted and having values < n, it gives all the indices not contained in it
// for ind = {1,3,5} and n = 7 the output is {0,2,4,6}
vector<int> opp_ind(vector<int> ind, int n) {
	vector<int> opp_ind;
	opp_ind.reserve(n - ind.size());
	int counter = 0;
	bool stop_counter = true;
	for (int i = 0; i < n; i++) {
		if (i == ind[counter] && stop_counter) {
			counter++;
			if (counter == ind.size()) {
				counter--;
				stop_counter = false;
			}
		}
		else {
			opp_ind.push_back(i);
		}
	}
	return opp_ind;
}

//return the indices not contained in ind but in set_of_ind
vector<int> opp_ind(vector<int> ind, vector<int> set_of_ind) {
	vector<int> opp_ind;
	opp_ind.reserve(set_of_ind.size() - ind.size());
	int counter = 0;
	bool stop_counter = true;
	for (int i = 0; i < set_of_ind.size(); i++) {
		if (set_of_ind[i] == ind[counter] && stop_counter) {
			counter++;
			if (counter == ind.size()) {
				counter--;
				stop_counter = false;
			}
		}
		else {
			opp_ind.push_back(set_of_ind[i]);
		}
	}
	return opp_ind;
}

//computes the norm of a given vector
//this name was chosen to rule out confusions with already existing function from any library
double mynorm(Eigen::VectorXd vector) {
	return sqrt(vector.dot(vector));
}

//given vectors v_1, ..., v_n and a start_ind between 1 and n, this method produces an orthonormal span v_1, ..., v_start_ind-1, w_start_ind, ... , w_n
void stableGramSchmidt(vector<Eigen::VectorXd>& span, int start_ind) {
	for (int i = start_ind; i < span.size(); i++) {
		Eigen::VectorXd w_i = span[i];
		for (int j = 0; j < i; j++) {
			Eigen::VectorXd w_j = span[j];
			w_i = w_i - w_j.dot(w_i) / w_j.dot(w_j) * w_j;
		}
		w_i = w_i / mynorm(w_i);
		span[i] = w_i;
	}
}

//a class summarizing all important properties for a given point:
//its coordinates, the hypothesis which cluster it belongs to, its current cost, weight and probability for the sampling algorithm
//contains setters and getters for each variable
class point {
private:
	Eigen::VectorXd coordinates;
	int hypothesis;
	double cost;
	double probability;
	double weight;
public:

	//a default constructor

	point() {}

	//a constructor using a VectorXf v
	point(Eigen::VectorXd v) {
		coordinates = v;
		hypothesis = -1;
		probability = 0;
		weight = 1;
		cost = std::numeric_limits<double>::max();
	}

	//getters
	double getcost() {
		return cost;
	}

	double getprob() {
		return probability;
	}

	double getweight() {
		return weight;
	}

	int gethypothesis() {
		return hypothesis;
	}

	Eigen::VectorXd getcoord() {
		return coordinates;
	}

	//setters
	void setcost(double c) {
		cost = c;
	}

	void setprob(double pr) {
		probability = pr;
	}

	void setweight(double w) {
		weight = w;
	}

	void sethypothesis(int h) {
		hypothesis = h;
	}

	void setcoord(Eigen::VectorXd p) {
		coordinates = p;
	}

};

//loads the reddit data and returns a pointset
//the data represents a matrix and each line contains: the row number, the column number and the value of a non-zero entry of the matrix
//as the dimension of the matrix of the given data is not known, first the dimensions are determined
//and then the data will be loaded
vector<point> loadredditdata() {
	int max_n = 0;
	int max_d = 0;
	string str;
	ifstream f;
	f.open("twitter_ny_train.csv");
	while (getline(f, str)) {
		vector<string> vec;
		stringstream ss(str);
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			vec.push_back(substr);
		}
		int current_n = stof(vec[0]);
		int current_d = stof(vec[1]);
		if (max_n < current_n) max_n = current_n;
		if (max_d < current_d) max_d = current_d;
	}
	max_n = max_n + 1;
	max_d = max_d + 1;
	f.close();
	string str2;
	ifstream g;
	g.open("twitter_ny_train.csv");
	Eigen::MatrixXd data = Eigen::MatrixXd::Zero(max_n, max_d);
	while (getline(g, str2)) {
		vector<string> vec;
		stringstream ss(str2);
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			vec.push_back(substr);
		}
		int i = stoi(vec[0]);
		int j = stoi(vec[1]);
		data(i,j) = stod(vec[2]);
	}
	vector<point> pointset;
	for (int i = 0; i < max_n; i++) {
		point p(data.row(i));
		pointset.push_back(p);
	}
	return pointset;
}

//loads the epileptic seizure data and returns a pointset
//format of the given data: each line is one medical case
//the first entry is the name, the last the class it belongs to, the rest are attributes we are going to use
vector<point> loadepilecticseizuredata() {
	vector<point> pointset;
	Eigen::VectorXd v(178);
	string str;
	ifstream f;
	f.open("EpilecticSeizure.txt");
	while (getline(f, str)) {
		vector<string> vec;
		stringstream ss(str);
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			vec.push_back(substr);
		}
		for (int i = 0; i < 178; i++) {
			//vec[0] contains the name of the file, so we just start one index later
			//vec[179] the class
			v(i) = stof(vec[i + 1]);
		}
		point p(v);
		pointset.push_back(p);
	}
	return pointset;
}

//loads the census data and returns a pointset
//format of the data: each line contains the census information about one person; the first value of each line is the case file number, the others numerical attributes we are going to use
//we use the first 100000 case
vector<point> loadcensusdata() {
	vector<point> pointset;
	Eigen::VectorXd v(68);
	string str;
	ifstream f;
	f.open("Census.txt");
	int iter = 0;
	while (getline(f, str) && iter < 100000) {
		iter++;
		vector<string> vec;
		stringstream ss(str);
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			vec.push_back(substr);
		}
		for (int i = 0; i < 68; i++) {
			//vec[0] contains case file number, so we just start one index later
			v(i) = stof(vec[i + 1]);
		}
		point p(v);
		pointset.push_back(p);
	}
	return pointset;
}

//creating an initial random uniform assignment for every point to a class represented by an integer in the range of [0, k-1]
//k determines the number of classes
void initassign(vector<point>& pointset, int k) {
	int randomclass;
	uniform_int_distribution<int> dist(0, k - 1);
	for (int i = 0; i < pointset.size(); i++) {
		randomclass = dist(mt);
		pointset[i].sethypothesis(randomclass);
	}
}

//a struct with constructors intended to form the data matrix for a given cluster i
//it saves all the relevant information
//the data matrix, the indices of the pointset indicating which points belong a cluster and a boolean variable being true if the matrix exists ( the cluster size is not zero )
struct subspacematrix {
	Eigen::MatrixXd matrix;
	vector<int> cluster_indices;
	bool status;

	//computes the data matrix for a pointset for the points being in cluster i
	subspacematrix(vector<point>& pointset, int i) {
		//initiliaze a vector to save the indices of the points being in cluster i in a vector matrixind
		vector<int> matrixind;
		//define status to be false if the cluster is empty
		status = true;
		//define the dimension of the given pointset by taking the coordinates from the first point
		int dim = (pointset[0].getcoord()).size();
		//save the indices of cluster i and determine the mean
		for (int j = 0; j < pointset.size(); j++) {
			if (pointset[j].gethypothesis() == i) {
				matrixind.push_back(j);
			}
		}
		//declare a matrix of dimensions  clustersize x dim to save the cluster matrix
		Eigen::MatrixXd m(matrixind.size(), dim);
		//fill the rows of the matrix with the points, if the cluster is nonempty
		if (matrixind.size() == 0) {
			status = false;
		}
		else {
			for (int j = 0; j < matrixind.size(); j++) {
				//vectors in the eigen package are matrices of the form size x 1, therefere we transpose the vector for our purposes
				m.row(j) = (pointset[matrixind[j]].getcoord()).transpose();
			}
		}
		//save the data matrix and the cluster_indices
		matrix = m;
		cluster_indices = matrixind;
	}

	//computes the matrix for the SVD for the points in the sample subset (determined by indices of the pointset being in cluster i
	subspacematrix(vector<point>& pointset, vector<int> indices, int i) {
		//initializing a vector saving the indices when the corresponding point is in cluster i
		vector<int> matrixind;
		//define status to be false if the cluster is empty
		status = true;
		//define the dimension of the given pointset by taking the coordinates from the first point
		int dim = (pointset[0].getcoord()).size();
		for (int j = 0; j < indices.size(); j++) {
			if (pointset[indices[j]].gethypothesis() == i) {
				matrixind.push_back(indices[j]);
			}
		}
		//declare a matrix of dimensions  clustersize x dim to save the cluster matrix
		Eigen::MatrixXd m(matrixind.size(), dim);
		//fill the rows of the matrix with the points, if the cluster is nonempty
		//subtract the cluster mean from each points coordinates to do so
		if (matrixind.size() == 0) {
			status = false;
		}
		else {
			for (int j = 0; j < matrixind.size(); j++) {
				//vectors in the eigen package are matrices of the form size x 1, therefere we transpose the vector for our purposes
				m.row(j) = (sqrt(pointset[matrixind[j]].getweight()) * pointset[matrixind[j]].getcoord()).transpose();
			}
		}
		//save the mean, the data matrix and the cluster_indices
		matrix = m;
		cluster_indices = matrixind;
	}
};

//struct to save the optimal subspaces computed by the SVD
//obtains the optimal subspaces by applying SVD to the data matrix for a cluster obtained by the struct subspacematrix
//span stands for the span of the subspace - an earlier implementation was using affine subspaces
//contains two constructors to obtain the optimal subspace of dimension q for a given cluster i of the pointset
//for the sampling k-means, the constructor additionally takes the indices of the sample as input
struct optimal_subspace {
	vector<Eigen::VectorXd> span;

	//empty constructor
	optimal_subspace() {}

	//constructor taking a pointset, the cluster number i and the size of subspaces q
	//used for the k-means subspace algorithm on an entire pointset (in contrast to a sample)
	optimal_subspace(vector<point>& pointset, int i, int q) {
		//declare a vector to contain the span
		vector<Eigen::VectorXd> subspace_span;
		//declare integers n,d to save the dimensions of the current data matrix
		int n, d;
		//using the constructor of the struct subspacematrix to get the data matrix of cluster i
		subspacematrix sm(pointset, i);
		Eigen::MatrixXd m = sm.matrix;
		//save the dimensions of m
		n = m.rows();
		d = m.cols();
		//check if the cluster contains points
		if (sm.status == true) {
			//use either Jacobi or BDCSVD according to the size of m, declare v to save V from the SVD D = U E V^T or thin SVD
			//Jacobi better for matrices smaller than 16x16
			Eigen::MatrixXd v;
			clock_t start = clock();
			if (n < 16 & d < 16) {
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
				v = svd.matrixV();
			}
			else {
				Eigen::BDCSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
				v = svd.matrixV();
			}
			clock_t stop = clock();
			svd_time += (double) (stop - start) / CLOCKS_PER_SEC;
			int fill_up_index = min(q, (int)v.cols());
			for (int j = 0; j < fill_up_index; j++) {
				//V is of the form dxfill_up_index, so, we take fill_up_index columns
				subspace_span.push_back(v.col(j));
			}
			/*if fill_up_index < q, we fill the span of the subspace by taking the coordinates of random points outside the cluster*/
			if (fill_up_index < q) {
				vector<int> non_cluster_indices = opp_ind(sm.cluster_indices, pointset.size());
				//pick randomly a point outside the cluster and add it
				for (int j = fill_up_index; j < q; j++) {
					uniform_int_distribution<int> uniform_dist(0, non_cluster_indices.size() - 1);
					int picked_ind = uniform_dist(mt);
					Eigen::VectorXd non_cluster_vector = pointset[non_cluster_indices[uniform_dist(mt)]].getcoord();
					non_cluster_indices.erase(non_cluster_indices.begin() + picked_ind);
					subspace_span.push_back(non_cluster_vector);
				}
				//orthonormalize the span
				stableGramSchmidt(subspace_span, fill_up_index);
			}
			if (subspace_span.size() == 0) cout << "error: empty subspace added" << endl;
		}
		span = subspace_span;
	}

	//constructor taking a pointset, the cluster number i and the size of subspaces q and a vector of indices representing a sample
	//used for sampling k-means
	optimal_subspace(vector<point>& pointset, int i, int q, vector<int> indices) {
		//declare a vector to contain the span
		vector<Eigen::VectorXd> subspace_span;
		//declare integers n,d to save the dimensions of the current data matrix
		int n, d;
		//using the constructor of the struct subspacematrix to get the data matrix of cluster i, the cluster mean is already subtracted
		subspacematrix sm(pointset, indices, i);
		Eigen::MatrixXd m = sm.matrix;
		//save the dimensions of m
		n = m.rows();
		d = m.cols();
		//check if the cluster contains points
		if (sm.status == true) {
			//use either Jacobi or BDCSVD according to the size of m, declare v to save V from the SVD D = U E V^T or thin SVD
			//Jacobi better for matrices smaller than 16x16
			Eigen::MatrixXd v;
			clock_t start = clock();
			if (n < 16 & d < 16) {
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
				v = svd.matrixV();
			}
			else {
				Eigen::BDCSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
				v = svd.matrixV();
			}
			clock_t stop = clock();
			svd_time += (double) (stop - start) / CLOCKS_PER_SEC;
			int v_cols = v.cols();
			int fill_up_index = min(q, v_cols);
			for (int j = 0; j < fill_up_index; j++) {
				//V is of the form dxfill_up_index, so, we take fill_up_index columns
				subspace_span.push_back(v.col(j));
			}
			//if we don't have enough columns, we fill the span of the subspace by taking the coordinates of random points outside the cluster
			if (fill_up_index < q) {
				vector<int> non_cluster_indices = opp_ind(sm.cluster_indices, pointset.size());
				//pick randomly a point outside the cluster and add it
				for (int j = fill_up_index; j < q; j++) {
					uniform_int_distribution<int> uniform_dist(0, non_cluster_indices.size() - 1);
					int picked_ind = uniform_dist(mt);
					Eigen::VectorXd non_cluster_vector = pointset[non_cluster_indices[uniform_dist(mt)]].getcoord();
					non_cluster_indices.erase(non_cluster_indices.begin() + picked_ind);
					subspace_span.push_back(non_cluster_vector);
				}
				//orthonormalize the span
				stableGramSchmidt(subspace_span, fill_up_index);
			}
			if (subspace_span.size() == 0) cout << "error: empty subspace added" << endl;
		}
		span = subspace_span;
	}

};

//method computing all the optimal subspaces of size q for a pointset, k clusters
//for k means by calling the corresponding constructor in optimal_subspace
//note that the cluster assignment is contained in the pointset
vector<optimal_subspace> opt_subsp(vector<point>& pointset, int k, int q) {
	vector<optimal_subspace> opt_subsp;
	for (int i = 0; i < k; i++) {
		opt_subsp.push_back(optimal_subspace(pointset, i, q));
	}
	return opt_subsp;
}

//method computing all the optimal subspaces of size q for a pointset, k clusters
//for sampling k means by calling the corresponding constructor in optimal_subspace
//note that the cluster assignment is contained in the pointset
vector<optimal_subspace> opt_subsp(vector<point>& pointset, int k, int q, vector<int> indices) {
	vector<optimal_subspace> opt_subsp;
	for (int i = 0; i < k; i++) {
		opt_subsp.push_back(optimal_subspace(pointset, i, q, indices));
	}
	return opt_subsp;
}

//computes the projection of an vector v onto a subspace spanned by the vectors in w
//assumes that the vectors spanning w are orthonormal, as the singular vectors are orthonormal
Eigen::VectorXd projection(Eigen::VectorXd v, vector<Eigen::VectorXd> w) {
	Eigen::VectorXd a = v.dot(w[0]) * w[0];
	if (w.size() > 1) {
		for (int i = 1; i < w.size(); i++) {
			a = a + v.dot(w[i]) * w[i];
		}
	}
	return a;
}

//calculates the distance of a point v to a subspace
//if w is empty, it returns an "infinite" distance, such that another subspace is chosen, when determining the nearest subspace
double distancesubspace(Eigen::VectorXd v, optimal_subspace w) {
	if ((w.span).size() == 0) return numeric_limits<double>::infinity();
	//subtract the support vector from v to determine the distance from v to w as if the support vector would be zero
	else {
		Eigen::VectorXd proj = projection(v, w.span);
		return ((v - proj).dot(v - proj));
	}
}

//a struct containing the index of the subspace for which the distance is minimal and the distance itself
//contains a constructor returning these values for a given vector v and a subspace w
struct mindistsubspace {
	int index;
	double dist;

	//a constructor which determines the nearest subspace and its distance to a point v amongst several subspaces containend in w
	mindistsubspace(Eigen::VectorXd v, vector<optimal_subspace> w) {
		//set the dist to "infinity" and update it every time, the distance of v to the current subspace is smaller
		dist = numeric_limits<double>::infinity();
		//declare current_dist to save the distance of a point to the subspace we currently look at
		double current_dist;
		//we go through all the subspaces in w, determining the distance between v and the current subspace and set update dist if we find a smaller distance
		for (int i = 0; i < w.size(); i++) {
			current_dist = distancesubspace(v, w[i]);
			if (current_dist < dist) {
				dist = current_dist;
				index = i;
			}
		}
	}

};

//updates the cost and the hypothesis class for each point for a given pointset and a vector of subspaces w
// indices specifies the subset of all points to be updated
void update(vector<point>& pointset, vector<int> indices, vector<optimal_subspace> w) {
	for (int i = 0; i < indices.size(); i++) {
		mindistsubspace mindist(pointset[indices[i]].getcoord(), w);
		pointset[indices[i]].sethypothesis(mindist.index);
		pointset[indices[i]].setcost(mindist.dist);
	}
}

//updates the cost and the hypothesis class for each point for a given pointset and a vector of subspaces w
void update(vector<point>& pointset, vector<optimal_subspace> w) {
	for (int i = 0; i < pointset.size(); i++) {
		mindistsubspace mindist(pointset[i].getcoord(), w);
		pointset[i].sethypothesis(mindist.index);
		pointset[i].setcost(mindist.dist);
	}
}

double weightedcost(vector<point>& pointset, vector<int> indices) {
	double cost = 0;
	double sum_of_weights = 0;
	double cost_point;
	double weight_point;
	for (int i = 0; i < indices.size(); i++) {
		cost_point = pointset[indices[i]].getcost();
		weight_point = pointset[indices[i]].getweight();
		cost = cost + weight_point * cost_point;
	}
	return cost;
}

//computes the total of the whole pointset, i.e. the sum of the costs of every point is computed
double globalcost(vector<point>& pointset) {
	double cost = 0;
	double cost_point;
	for (int i = 0; i < pointset.size(); i++) {
		cost_point = pointset[i].getcost();
		cost = cost + cost_point;
	}
	return cost;
}

//implementation of kmeanssubspace for a weighted subset of the given point marked by indices
//updates here are just done on the particular subset on purpose
//also just weightedcost is used as we operate on the subset of points
//input: a pointset
//a integer k determining the desired number of clusters
//an integer maxiter determining the maximal number of iterations
//an integer q setting the dimension of the subspaces
//output: the optimal subspaces
vector<optimal_subspace> kmeanssubspace(vector<point>& pointset, vector<int> indices, int k, int q, int maxiter) {
	//assign every point uniformly random to the k clusters
	initassign(pointset, k);
	//initialize two vectors of optimal subspace to save old and new optimal subspaces
	vector<optimal_subspace> oldSubspaces, newSubspaces;
	//compute the optimal subspaces for the first time
	newSubspaces = opt_subsp(pointset, k, q, indices);
	//update the cost and the hypothesis according to the new optimal subspaces
	update(pointset, indices, newSubspaces);
	//declare newcost and oldcost to save the cost
	double newcost, oldcost;
	//compute the new cost, here we use weighted cost, as we operate on a weighted sample
	newcost = weightedcost(pointset, indices);
	//define iter to count the iterations and stop after maxiter iterations
	int iter = 0;
	do {
		iter++;
		//save current optimal subspace and cost
		oldSubspaces = newSubspaces;
		oldcost = newcost;
		//compute new optimal subspaces, update cost and assignment and compute new cost
		newSubspaces = opt_subsp(pointset, k, q, indices);
		update(pointset, indices, newSubspaces);
		newcost = weightedcost(pointset, indices);
		//repeat until the cost does not improve anymore or the maximal iterations are reached
	} while (newcost < oldcost && iter < maxiter);
	if (newcost < oldcost) {
		return newSubspaces;
	}
	else {
		return oldSubspaces;
	}
}

//struct to save the results from the kmeanssubspace algorithm applied on a pointset ( in contrast to a weighted subset of it)
struct kmeansres {
	vector<optimal_subspace> subspaces;
	double init_cost;
	double cost;
	int iterations;

	kmeansres(){}

	kmeansres(vector<optimal_subspace> subsp, double ic, double c, int iter) {
		subspaces = subsp;
		init_cost = ic;
		cost = c;
		iterations = iter;
	}
};

//global version of kmeanssubspace
//differs in three things from the version for weighted subsets: uses methods without sample indices and computes global cost instead of weighted cost and most importantly the cluster matrix is not weighted
kmeansres kmeanssubspace(vector<point>& pointset, int k, int q, int maxiter) {
	initassign(pointset, k);
	vector<optimal_subspace> oldSubspaces, newSubspaces;
	newSubspaces = opt_subsp(pointset, k, q);
	double newcost, oldcost;
	update(pointset, newSubspaces);
	newcost = globalcost(pointset);
	double init_cost = newcost;
	int iter = 0;
	do {
		iter++;
		oldSubspaces = newSubspaces;
		oldcost = newcost;
		newSubspaces = opt_subsp(pointset, k, q);
		update(pointset, newSubspaces);
		newcost = globalcost(pointset);
	} while (newcost < oldcost && iter < maxiter);
	if (newcost < oldcost) {
		return kmeansres(newSubspaces, init_cost, newcost, iter);
	}
	else {
		update(pointset, oldSubspaces);
		return kmeansres(oldSubspaces, init_cost, oldcost, iter);
	}
}

//initializes the weight and probabilities
void initwp(vector<point>& pointset, int k, int q) {
	double n = pointset.size();
	if (10 * k > n) cout << "In initwp, k is too big" << endl;
	double probability = 1/n * (k*q);
	double weight = 1 / probability;
	for (int i = 0; i < n; i++) {
		pointset[i].setprob(probability);
		pointset[i].setweight(weight);
	}
}

//increases the weights and decreases the probabilities for points which have higher cost than cavg* averagecost
//adjusts the weights and probabilites by the factor cadj then
void adjustwp_aboveavg(vector<point>& pointset, double total_cost, int cavg, int cadj) {
	int n = pointset.size();
	double average_cost = total_cost / n;
	for (int i = 0; i < n; i++) {
		point& current_point = pointset[i];
		if (current_point.getcost() > cavg * average_cost) {
			current_point.setweight(max(1.0, current_point.getweight() / cadj));
			current_point.setprob(min(1.0, current_point.getprob() * cadj));

		}
	}
}

//similar as adjustwp_aboveavg, but takes samplecost/n instead of averagecost
void adjustwp_abovesamplavg(vector<point>& pointset, double samplecost, int cavg, int cadj) {
	double avg_samplecost = samplecost / pointset.size();
	for (int i = 0; i < pointset.size(); i++) {
		point& current_point = pointset[i];
		double weighted_pointcost = current_point.getweight() * current_point.getcost();
		if (weighted_pointcost > cavg * avg_samplecost) {
			current_point.setweight(max(1.0, current_point.getweight() / cadj));
			current_point.setprob(min(1.0, current_point.getprob() * cadj));
		}
	}
}


//adjusts the probabilities proportional to the cost of the point divided by the total cost
//if the probability does not increase, the old probability is taken
void adjustwp_proptocost(vector<point>& pointset, double total_cost, int iter) {
	for (int i = 0; i < pointset.size(); i++) {
		point& current_point = pointset[i];
		double prob = min(1.0, max(current_point.getprob(), pow(2,iter)*current_point.getcost() / total_cost));
		current_point.setprob(prob);
		current_point.setweight(1 / prob);
	}
}

//adjusts the probabilities and weights, such that the probabilities are all equal, but increase by a factor 2 in each iteration
void adjustwp_uniform(vector<point>& pointset, int iter) {
	for (int i = 0; i < pointset.size(); i++) {
		point& current_point = pointset[i];
		double prob = min(1.0, pow(2, iter) * current_point.getprob());
		current_point.setprob(prob);
		current_point.setweight(1 / prob);
	}
}

//samples the points according to their probabilites
//returns a vector of integers containing all indices of the points which were sampled
vector<int> sampling(vector<point>& pointset) {
	vector<int> sample_indices;
	clock_t start = clock();
	do {
		sample_indices.clear();
		for (int i = 0; i < pointset.size(); i++) {
			bernoulli_distribution dist(pointset[i].getprob());
			if (dist(mt)) {
				sample_indices.push_back(i);
			}
		}
	} while (sample_indices.size() < 1);
	clock_t stop = clock();
	double duration = (double) (stop-start) / CLOCKS_PER_SEC;
	sampling_time += duration;
	return sample_indices;
}

//struct to save the results from the sample-based algorithm
struct skmeans_results {
	vector<optimal_subspace> subspaces;
	double init_totalcost;
	double init_samplecost;
	double totalcost;
	double samplecost;
	int iterations;
	int samplesize;

	skmeans_results(){}

	skmeans_results(vector<optimal_subspace> subsp, double itc, double isc, double tc, double sc, int iter, int ss) { 
		subspaces = subsp;
		init_totalcost = itc;
		init_samplecost = isc;
		totalcost = tc;
		samplecost = sc;
		iterations = iter;
		samplesize = ss;
	}
};

//implementation of sample k means subspace algorithm
//input variables are: the cost constant, the number of cluster k, the dimension of subspaces q, localmaxiter determines how many iterations the k means subspace algorithm uses for the samples, maxiter bounds the iterations for the sample k-means subspace algorithm, pointset is a structure containing all points with their information, cavg and cadj are constant for the sampling process
//returns a set of optimal subspaces
skmeans_results samplingkmeans(double cost_constant, int k, int q, int localmaxiter, int maxiter, vector<point>& pointset, int samplingOption, int cavg, int cadj) {
	//initialize the weights and probabilities
	initwp(pointset, k, q);
	//choose first sample to perform the kmeans subspace algorithm on 
	vector<int> sampleIndices = sampling(pointset);
	//use oldsubspaces to store the old subspaces and subspace for the new subspaces
	vector<optimal_subspace> oldsubspaces;
	//perform kmeans with k clusters, optimal subspaces of size q and localmaxiter iterations
	vector<optimal_subspace> subspaces = kmeanssubspace(pointset, sampleIndices, k, q, localmaxiter);
	//update the cost and hypothesis for the pointset
	update(pointset, subspaces);
	//declare sample and total cost and compute it
	double samplecost = weightedcost(pointset, sampleIndices);
	double totalcost = globalcost(pointset);
	double init_samplecost = samplecost;
	double init_totalcost = totalcost;
	//iter is used to count the iterations
	int iter = 0;
	while (samplecost < totalcost * cost_constant && iter < maxiter) {
		iter++;
		//sample according to samplingOption chosen
		switch (samplingOption) {
		case 1:
			adjustwp_aboveavg(pointset, totalcost, cavg, cadj);
			break;
		case 2:
			adjustwp_abovesamplavg(pointset, samplecost, cavg, cadj);
			break;
		case 3:
			adjustwp_proptocost(pointset, totalcost, iter - 1);
			break;
		case 4:
			adjustwp_uniform(pointset, iter - 1);
			break;
		default:
			cout << "No such option avaiable" << endl;
		}
		//delete old sample and pick a new sample, then repeat the steps of applying kmeans, updating the weights, probabilites, cost and hypothesis
		sampleIndices.clear();
		sampleIndices = sampling(pointset);
		oldsubspaces = subspaces;
		subspaces = kmeanssubspace(pointset, sampleIndices, k, q, localmaxiter);
		update(pointset, subspaces);
		samplecost = weightedcost(pointset, sampleIndices);
		totalcost = globalcost(pointset);
	}
	skmeans_results results(subspaces, init_totalcost, init_samplecost, totalcost, samplecost, iter, sampleIndices.size());
	return results;
}

//a function which returns a nxn matrix representing a rotation in the n-dimensional space
//each entry of the matrix is taken from a normal distribution, then a QR decomposition is applied
//and Q will be returned
Eigen::MatrixXd rotation_matrix(int n) {
	normal_distribution<double> n_dist(0.0, 1.0);
	Eigen::MatrixXd rotation_matrix(n, n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			rotation_matrix(i, j) = n_dist(mt);
		}
	}
	Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr_decomposition(rotation_matrix);
	rotation_matrix = qr_decomposition.matrixQ();
	return rotation_matrix;
}

//a function to apply the implementation of the k means subspace algorithm for the given settings: a pointset, a number of clusters k, the dimension of subspaces q, a limit of iterations of the while loop maxiter
//number_of_repetitions specifies the amount of times the algorithm get applied
//the results are saved in the file with name filename with the mode append, i.e. they are added to the given file
void apply_subspacekmeans(vector<point>& pointset, int k, int q, int maxiter, int number_of_repetitions, string filename) {
	vector<double> totalcost(number_of_repetitions, 0.0);
	vector<double> init_cost(number_of_repetitions, 0.0);
	vector<double> iterations(number_of_repetitions, 0.0);
	vector<double> time(number_of_repetitions, 0.0);
	vector<double> svdtime(number_of_repetitions, 0.0);
	ofstream file(filename, ofstream::app);
	file << "Vanilla Algorithm for k = " << k << " , q = " << q << " maxiter = " << maxiter << " number of rep. " << number_of_repetitions << endl;
	file.close();
	for (int i = 0; i < number_of_repetitions; i++) {
		seed = rd();
		mt.seed(seed);
		file.open(filename, ofstream::app);
		file << "used seed for repetition " << i + 1 << " : " << seed << endl;
		file.close();
		clock_t start = clock();
		kmeansres kresults = kmeanssubspace(pointset, k, q, maxiter);
		clock_t stop = clock();
		double duration = (double) (stop-start) / CLOCKS_PER_SEC;
		init_cost[i] = kresults.init_cost;
		totalcost[i] = kresults.cost;
		iterations[i] = kresults.iterations;
		time[i] = duration;
		svdtime[i] = svd_time;
		svd_time = 0.0;
	}
	//compute averages
	double avg_init_cost = compute_mean(init_cost);
	double avg_totalcost = compute_mean(totalcost);
	double avg_iterations = compute_mean(iterations);
	double avg_time = compute_mean(time);
	double avg_svdtime = compute_mean(svdtime);
	//compute variances
	double var_init_cost = compute_variance(init_cost, avg_init_cost);
	double var_totalcost = compute_variance(totalcost, avg_totalcost);
	double var_iterations = compute_variance(iterations, avg_iterations);
	double var_time = compute_variance(time, avg_time);
	double var_svdtime = compute_variance(svdtime, avg_svdtime);

	file.open(filename, ofstream::app);
	file << "Results: totalcost, initcost, iterations, time, svd time " << endl;
	for (int i = 0; i < number_of_repetitions; i++) {
		file << totalcost[i] << ";" << init_cost[i] << ";" << iterations[i] << ";" << time[i] << ";" << svdtime[i] << endl;
	}
	file << " " << endl;
	file << "Average of all repetitions:" << endl;
	file << avg_totalcost << ";" << avg_init_cost << ";" << avg_iterations << ";" << avg_time << ";" << avg_svdtime <<endl;
	file << "Variance of all repetitions:" << endl;
	file << var_totalcost << ";" << var_init_cost << ";" << var_iterations << ";" << var_time << ";" << var_svdtime << endl;
	file << " " << endl;
	file.close();
}

//similar as apply_subspacekmeans
//additional input is due to the additional input needed for the implementation of the sample-based algorithm
void apply_sampling_subspacekmeans(vector<point>& pointset, int k, int q, int local_maxiter, int maxiter, int sampling_option, double cost_constant, double cavg, double cadj, int number_of_repetitions, string filename) {
	vector<double> totalcost(number_of_repetitions, 0.0);
	vector<double> samplecost(number_of_repetitions, 0.0);
	vector<double> init_totalcost(number_of_repetitions, 0.0);
	vector<double> init_samplecost(number_of_repetitions, 0.0);
	vector<double> time(number_of_repetitions, 0.0);
	vector<double> samplesize(number_of_repetitions, 0.0);
	vector<double> iterations(number_of_repetitions, 0.0);
	vector<double> samplingtime(number_of_repetitions, 0.0);
	vector<double> svdtime(number_of_repetitions, 0.0);
	ofstream file(filename, ofstream::app);
	file << "Sample based algorithm with option " << sampling_option << endl;
	file << "costconst = " << cost_constant << endl;
	file.close();
	for (int i = 0; i < number_of_repetitions; i++) {
		seed = rd();
		mt.seed(seed);
		file.open(filename, ofstream::app);
		file << "used seed for repetition " << i + 1 << " : " << seed << endl;
		file.close();
		clock_t start = clock();
		skmeans_results skres = samplingkmeans(cost_constant, k, q, local_maxiter, maxiter, pointset, sampling_option, cavg, cadj);
		clock_t stop = clock();
		double duration = (double) (stop-start) / CLOCKS_PER_SEC;
		totalcost[i] = skres.totalcost;
		samplecost[i] = skres.samplecost;
		init_totalcost[i] = skres.init_totalcost;
		init_samplecost[i] = skres.init_samplecost;
		time[i] = duration;
		samplesize[i] = skres.samplesize;
		iterations[i] = skres.iterations;
		samplingtime[i] = sampling_time;
		sampling_time = 0.0;
		svdtime[i] = svd_time;
		svd_time = 0.0;
	}
	file.open(filename, ofstream::app);
	file << "Results: totalcost, samplecost, time, iterations, samplesize, sampling time, svd time, init_totalcost, init_samplecost" << endl;
	for (int i = 0; i < number_of_repetitions; i++) {
		file << "Results of repetition nr. " << i + 1 << endl;
		file << totalcost[i] << ";" << samplecost[i] << ";" << time[i] << ";" << iterations[i] << ";" << samplesize[i] << ";" << samplingtime[i] << ";" << svdtime[i] << ";" << init_totalcost[i] << ";" << init_samplecost[i] << endl;
	}
	//compute averages
	double avg_totalcost = compute_mean(totalcost);
	double avg_samplecost = compute_mean(samplecost);
	double avg_init_totalcost = compute_mean(init_totalcost);
	double avg_init_samplecost = compute_mean(init_samplecost);
	double avg_time = compute_mean(time);
	double avg_samplesize = compute_mean(samplesize);
	double avg_iterations = compute_mean(iterations);
	double avg_samplingtime = compute_mean(samplingtime);
	double avg_svdtime = compute_mean(svdtime);
	//compute variances
	double var_totalcost = compute_variance(totalcost, avg_totalcost);
	double var_samplecost = compute_variance(samplecost, avg_samplecost);
	double var_init_totalcost = compute_variance(init_totalcost, avg_init_totalcost);
	double var_init_samplecost = compute_variance(init_samplecost, avg_init_samplecost);
	double var_time = compute_variance(time, avg_time);
	double var_samplesize = compute_variance(samplesize, avg_samplesize);
	double var_iterations = compute_variance(iterations, avg_iterations);
	double var_samplingtime = compute_variance(samplingtime, avg_samplingtime);
	double var_svdtime = compute_variance(svdtime, avg_svdtime);
	file << " " << endl;
	file << "Average of all repetions:" << endl;
	file << avg_totalcost << ";" << avg_samplecost << ";" << avg_time << ";" << avg_iterations << ";" << avg_samplesize << ";" << avg_samplingtime << ";" << avg_svdtime << ";" << avg_init_totalcost << ";" << avg_init_samplecost << endl;
	file << "Variance of all repetitions:" << endl;
	file << var_totalcost << ";" << var_samplecost << ";" << var_time << ";" << var_iterations << ";" << var_samplesize << ";" << var_samplingtime << ";" << var_svdtime << ";" << var_init_totalcost << ";" << var_init_samplecost << endl;
	file << " " << endl;
	file.close();
}

//generates cluster as described in the thesis
//span_ind should already be ordered to apply opp_ind
vector<point> generate_cluster(int number_pts, vector<int> span_ind, int dim_pts, int uniform_range, double mean, double std_deviation) {
	if (span_ind.size() >= dim_pts) cout << "Error: The dimension of the points should be higher than the one of the subspaces";

	vector<int> non_span_ind = opp_ind(span_ind, dim_pts);
	vector<point> data;
	data.reserve(number_pts);
	Eigen::MatrixXd data_matrix(number_pts, dim_pts);
	mt.seed(seed);
	for (int j = 0; j < span_ind.size(); j++) {
		uniform_real_distribution<double> u_dist(-uniform_range, uniform_range);
		int col_ind = span_ind[j];
		for (int i = 0; i < number_pts; i++) {
			data_matrix(i, col_ind) = u_dist(mt);
		}
	}
	for (int j = 0; j < non_span_ind.size(); j++) {
		normal_distribution<double> n_dist(mean, std_deviation);
		int col_ind = non_span_ind[j];
		for (int i = 0; i < number_pts; i++) {
			data_matrix(i, col_ind) = n_dist(mt);
		}
	}
	data_matrix = data_matrix * rotation_matrix(dim_pts);
	for (int i = 0; i < number_pts; i++) {
		point p(data_matrix.row(i));
		data.push_back(p);
	}
	return data;
}

//a function to randomly choose the coordinates which span a cluster out of {0, ... , range-1}
//sorts the indices in order to be used by opp_ind in generatecluster
vector<int> generate_span_ind(int span_size, int range) {
	vector<int> span_ind;
	span_ind.reserve(span_size);
	mt.seed(seed);
	vector<int> all_ind;
	all_ind.reserve(range);
	for (int i = 0; i < range; i++) {
		all_ind.push_back(i);
	}
	for (int i = 0; i < span_size; i++) {
		uniform_int_distribution<> u(0, range - 1);
		int picked_ind = u(mt);
		span_ind.push_back(all_ind[picked_ind]);
		all_ind.erase(all_ind.begin() + picked_ind);
		range--;
	}
	sort(span_ind.begin(), span_ind.end());
	return span_ind;
}

//a function to create a pointset with symmetric clusters
//pointset_size determines the size of the pointset returned, number_cluster the cluster of numbers,
//clusterdim the cluster dimension, dim_pts the dimension of the space they lie in
//uniform_range, mean and std_derivation specify the values used for the distributions for the generation of the points
vector<point> symmetric_cluster(int pointset_size, int number_cluster, int clusterdim, int dim_pts, int uniform_range, double mean, double std_deviation) {
	//define the cluster size
	int cluster_size = pointset_size / number_cluster;
	//generate indices representing the spanning dimensions of the first cluster
	vector<int> span_ind = generate_span_ind(clusterdim, dim_pts);
	vector<point> pointset;
	//save first cluster
	pointset = generate_cluster(cluster_size, span_ind, dim_pts, uniform_range, mean, std_deviation);
	//repeatedly, generate new spanning indices for each cluster and generate cluster, then append cluster
	for (int i = 1; i < number_cluster; i++) {
		span_ind = generate_span_ind(clusterdim, dim_pts);
		vector<point> current_cluster = generate_cluster(cluster_size, span_ind, dim_pts, uniform_range, mean, std_deviation);
		pointset.insert(pointset.end(), current_cluster.begin(), current_cluster.end());
	}
	return pointset;
}

//produces a pointset with 10 unbalanced clusters 
vector<point> unbalanced_cluster(int pointset_size, int clusterdim, int dim_pts, int uniform_range, double mean, double std_deviation) {
	vector<int> clustersizes = { 5, 5, 10, 20, 30, 30};
	vector<point> pointset;
	for (int i = 0; i < clustersizes.size(); i++) {
		vector<int> span_ind = generate_span_ind(clusterdim, dim_pts);
		int clustersize = (double) clustersizes[i] / 100 * pointset_size;
		vector<point> current_cluster = generate_cluster(clustersize, span_ind, dim_pts, uniform_range, mean, std_deviation);
		pointset.insert(pointset.end(), current_cluster.begin(), current_cluster.end());
	}
	return pointset;
}

//a function to apply the experiment with the structured generated data
//the results are saved in the file named filename
//number_of_repetitions specifies the number of times any algorithm is repeated with a given setting
void test_symmetric_clusters(int number_of_repetitions, string filename) {
	vector<int> ns = { 10000, 10000, 10000, 10000, 10000, 10000 ,10000 , 50000, 50000 , 50000 , 50000 , 50000 , 50000, 50000 };
	vector<int> ds = { 10, 50, 50, 50, 250, 250, 250, 10, 50, 50, 50, 250, 250, 250 };
	vector<int> ks = { 4, 4 ,4, 20, 4, 4, 20, 4, 4, 4, 20, 4, 4 , 20 };
	vector<int> qs = { 4, 4, 20, 4, 4, 20, 4, 4, 4, 20, 4, 4, 20, 4 };
	vector<double> cost_constants = { 0.1, 0.5 };
	double cadj = 2.0;
	double cavg = 1.5;
	int maxiter = 25;
	int local_maxiter = 20;
	int uniform_range = 200;
	double std_deviation = 1.0;
	double mean = 0.0;
	ofstream file(filename, ios::app);
	file << "Syntheticly generated structured pointsets" << endl << endl << endl;
	file << "General settings: cadj, cavg, maxiter, localmaxiter: " << cadj << "," << cavg << "," << maxiter << "," << local_maxiter << endl;
	file << " " <<  endl << " " << endl;
	file.close();
	for (int i = 0; i < ns.size(); i++) {
		int n = ns[i];
		int d = ds[i];
		int k = ks[i];
		int q = qs[i] + 1;
		int clusterdim = qs[i];
		file.open(filename, ios::app);
		file << " " << endl << " " << endl;
		file << "Syntheticly generated pointset with n = " << n << " , d =  " << d << " , k = " << k << " , q = " << q << endl;
		file << "seed used to generate the pointset: " << seed << endl;
		file << " " << endl << " " << endl;;
		file.close();
		vector<point> pointset = symmetric_cluster(n, k, clusterdim, d, uniform_range, mean, std_deviation);
		//start saving file here
		apply_subspacekmeans(pointset, k, q, local_maxiter, number_of_repetitions, filename);
		for (int j = 0; j < cost_constants.size(); j++) {
			double cost_constant = cost_constants[j];
			for (int option = 1; option < 5; option++) {
				apply_sampling_subspacekmeans(pointset, k, q, local_maxiter, maxiter, option, cost_constant, cavg, cadj, number_of_repetitions, filename);
			}
		}
	}
}

//a function to apply the experiment with the unbalanced generated data
//input is used similar as for the function for the structured data
void test_unbalanced_clusters(int number_of_repetitions, string filename) {
	vector<int> ns = { 10000, 10000, 10000 ,10000 ,10000, 50000 , 50000, 50000 , 50000 , 50000 };
	vector<int> ds = { 10, 50, 50, 250, 250, 10, 50, 50, 250, 250 };
	vector<int> qs = { 4, 4, 20, 4, 20, 4, 4, 20, 4, 20 };
	int k = 6;
	vector<double> cost_constants = { 0.1, 0.5 };
	double cadj = 2.0;
	double cavg = 1.5;
	int maxiter = 25;
	int local_maxiter = 20;
	int uniform_range = 200;
	double std_deviation = 3.0;
	double mean = 0.0;
	ofstream file(filename, ios::app);
	file << "Syntheticly generated unbalanced pointsets" << endl << endl << endl;
	file << "General settings: cadj, cavg, maxiter, localmaxiter: " << cadj << "," << cavg << "," << maxiter << "," << local_maxiter << endl;
	file.close();
	for (int i = 0; i < ns.size(); i++) {
		int n = ns[i];
		int d = ds[i];
		int q = qs[i] + 1;
		int clusterdim = qs[i];
		file.open(filename, ios::app);
		file << "Syntheticly generated unbalanced pointset with n = " << n << " , d =  " << d  << " , q = " << q << endl;
		file << "seed used to generate the pointset: " << seed << endl;
		file.close();
		vector<point> pointset = unbalanced_cluster(n, clusterdim, d, uniform_range, mean, std_deviation);
		//start saving file here
		apply_subspacekmeans(pointset, k, q, local_maxiter, number_of_repetitions, filename);
		for (int j = 0; j < cost_constants.size(); j++) {
			double cost_constant = cost_constants[j];
			for (int option = 1; option < 5; option++) {
				apply_sampling_subspacekmeans(pointset, k, q, local_maxiter, maxiter, option, cost_constant, cavg, cadj, number_of_repetitions, filename);
			}
		}
	}
}

//a function to conduct the experiment for the real world data sets
//number_of_repetitions and filename have the same purpose as in the two functions above
//pointset represents the data of the real world data
//ks and qs are vectors containing the settings for k and q
void test_real_data(vector<point>& pointset, int number_of_repetitions, string filename, vector<int> ks, vector<int> qs) {
	vector<double> cost_constants = { 0.1, 0.5 };
	double cadj = 2.0;
	double cavg = 1.5;
	int maxiter = 25;
	int local_maxiter = 20;
	ofstream file(filename, ios::app);
	file << filename << endl;
	file << "General settings: cadj, cavg, maxiter, localmaxiter: " << cadj << "," << cavg << "," << maxiter << "," << local_maxiter << endl << endl << endl;;
	file.close();
	for (int i = 0; i < ks.size(); i++) {
		int k = ks[i];
		int q = qs[i];
		file.open(filename, ios::app);
		file << " k = " << k << " , q = " << q << endl;
		file.close();
		apply_subspacekmeans(pointset, k, q, local_maxiter, number_of_repetitions, filename);
		for (int j = 0; j < cost_constants.size(); j++) {
			double cost_constant = cost_constants[j];
			for (int option = 1; option < 5; option++) {
				apply_sampling_subspacekmeans(pointset, k, q, local_maxiter, maxiter, option, cost_constant, cavg, cadj, number_of_repetitions, filename);
			}
		}
	}
}

int main() {
	//test_symmetric_clusters(5, "finalstructureddata241120.txt");

	//test_unbalanced_clusters(5, "finalunbalanceddata231120test.txt");

	//vector<int> ks = { 1, 10, 5, 1};
	//vector<int> qs = { 50, 5, 5, 25 };
	//vector<point> pointset = loadcensusdata();
	//test_real_data(pointset, 5, "censusdata251120.txt", ks, qs);

	//vector<point> pointset = loadepilecticseizuredata();
	//cout << "pointset size: " << pointset.size() << " point dimension: " << pointset[0].getcoord().size() << endl;
	//int dim = pointset[0].getcoord().size();
	//for (int i = 0; i < pointset.size(); i++) {
	//	if (dim != pointset[i].getcoord().size()) cout << "not equal dimensions at " << i << endl;
	//}
	//vector<int> ks = {5, 1, 10, 1};
	//vector<int> qs = {10, 50, 10, 100 };
	//test_real_data(pointset, 5, "epilepticseizuredata241120.txt", ks, qs);
}

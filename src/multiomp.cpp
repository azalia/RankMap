#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <ctime>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <mpi.h>

using namespace std;
using namespace Eigen;

typedef SparseMatrix<double> SparseMatrixD;
typedef Eigen::Triplet<double> Trip;
int glob_count =0;
int neg_count =0;

int verbos = 0;

inline void printPercent(int i, int n)
{
	if(verbos && ( (int)((100.0*(i-1)) / n) < (int)((100.0*i) / n)))
	{
		cout << "\r\033[K"; // erase line
		cout << (int)((100.0*i) / n) << "%"<<std::flush;
	}
}

double getTimeMs(const timeval t1,const  timeval t2)
{
	double tE = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	tE += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	
	return tE;
}



int maxID(const MatrixXd & e)
{
	double max = 0;
	int index = 0;
	if(	e.cols()>1)	
	{
		for(int i=0; i<e.cols(); i++)
		{
			if(e(0,i) >= max)
			{
				index = i;
				max = e(0,i);
			}
		}
	}
	else
	{
		for(int i=0; i<e.rows(); i++)
		{
			if(e(i,0) >= max)
			{
				index = i;
				max = e(i,0);
			}
		}
	}
	
	return index;
}
int OMPBATCHLLT(MatrixXd& D, MatrixXd& A, int k, double epsilon, vector<Trip>& tripletV)
{

	timeval t1, t2;
	int l = D.cols();
	int m = D.rows();
	int n = A.cols();
	assert(A.rows() == m);
	
	tripletV.clear();
	tripletV.reserve(k * n);
	
	
	
	
	MatrixXd DtA = D.transpose()*A;
	MatrixXd G = D.transpose()*D;
	
	for(int c=0;c<n;c++)
	{
		double anorm = A.col(c).norm();
		
		if(anorm < 1E-9)
			break;
				
		assert((anorm - 1) < 1E-3);
	
		
		int i=0;
		
		double dt1 = 0, dt2 = 0, dt3 = 0;

		double delta, delta_old = 0;
		double er, er_old = A.col(c).transpose()*A.col(c);
		
		MatrixXd supp(2, k);
		MatrixXd alpha_0 = DtA.col(c);
		MatrixXd alpha = alpha_0;
		MatrixXd alpha_I(k,1);
		MatrixXd G_I(l,k);
		MatrixXd G_II(k,k);
		MatrixXd beta;
		MatrixXd beta_I;
		MatrixXd L = MatrixXd::Zero(k,k);
		L(0,0) = 1;
		MatrixXd w;
		MatrixXd gamma_I;
		MatrixXd Linvalpha;
		for(i=0;i<k;i++)
		{
			int idx = maxID(alpha.cwiseAbs());
			G_I.col(i) = G.col(idx);
			
			
			if(i>0)
			{
				w = L.block(0,0,i,i).triangularView<Eigen::Lower>().solve(G_I.block(idx,0,1,i).transpose());
			
				L.block(i,0,1,i) = w.transpose();
				MatrixXd tempw = w.transpose()*w;
				double wnorm = tempw(0,0);
	
				if((1 - wnorm)<1E-6)
				{	
					glob_count++;
					break;
					
					}
				L(i,i) = sqrt(1 - wnorm);
				
			}
			
			supp(0,i) = idx;
		
			alpha_I(i, 0) =  alpha_0(idx, 0);

			Linvalpha = L.block(0,0,i+1,i+1).triangularView<Eigen::Lower>().solve(alpha_I.block(0,0,i+1,1));	
			gamma_I = L.block(0,0,i+1,i+1).transpose().triangularView<Eigen::Upper>().solve(Linvalpha);

			beta = G_I.block(0,0,l,i+1)*gamma_I;
			
			beta_I = MatrixXd::Zero(i+1,1);
			for(int j=0;j<=i;j++)
			{
				beta_I(j,0) = beta(supp(0,j),0);
			}
			
			
			alpha = alpha_0 - beta;
			
			
			delta = (gamma_I.transpose()*beta_I)(0,0);
			er = (er_old - delta + delta_old); 
		
			supp(1,i) = er/anorm;
			if(er< epsilon*epsilon)
			{
				neg_count++;
				i++;
				break;
			}	
			er_old = er;
			delta_old = delta;
		}
		
		
			
			printPercent(c, n);

		
		
		for(int j=0;j<i;j++)
		{
			tripletV.push_back(Trip(supp(0,j), c, gamma_I(j)));
				
		}
	
	}


	return 0;
}

int main(int argc, char*argv[])
{

	srand(0);
	
	
	if(argc!=11)
	{
		
			cout << "usage: ./omp infile outfile dfile m n l kperl epsilon of verbose" << endl;
			cout << "QR_batch: QR 0, batch 1" << endl;
			cout << "N>n for subset omp, N==n for normal omp" << endl;
			cout << "ncpu for openmp cores, ncpu==1 when using mpi" << endl;
			cout << "kperl= k / l" << endl;
			cout << "lmin <= l < lmax" << endl;
			cout << "of==1 for writing outputs" << endl;
	
		return -1;
	}
	int dfile = atoi(argv[3]);
	int m = atoi(argv[4]);
	int n = atoi(argv[5]);
	int l = atoi(argv[6]);
	double kperl = atof(argv[7]);
	double epsilon = atof(argv[8]);
	int of = atoi(argv[9]);
	verbos = atoi(argv[10]);
	
	timeval t1, t2, t3;

	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);		
	
	int myn;
	int stepn = n/npes;
	if(myrank<npes)
		myn = (n)/npes;
	else
		myn = n-stepn*(npes-1);	
	
	
	MatrixXd A = MatrixXd::Zero(m, myn);	
	MatrixXd D = MatrixXd::Zero(m, l);
	vector<Trip> tripletV;

gettimeofday(&t1, NULL);	
	if(!myrank) cout<<"Start reading A"<<endl;
	stringstream  sofA;


ifstream fr;
fr.open(argv[1]);
for(int i=0; i< m; i++)
	{
		for(int j=0; j< n; j++)
		{
			if(!fr.eof( ))
			{
				double temp;
				fr >> temp;
				if(myrank*stepn<=j && j<myrank*stepn+myn)
					{

						A(i, j-myrank*stepn) = temp;
					}
		}
	}
	
}


MPI_Barrier(MPI_COMM_WORLD);



fr.close();
	for(int i=0;i<A.cols();i++)
	{
		if(A.col(i).norm() > 1E-3)
		{
			A.col(i) =  A.col(i) / A.col(i).norm();
		}
	}
	
	if(!myrank) cout<<"start reading D"<<endl;
	
	ifstream fd;
	fd.open(argv[3]);
	

	
	for(int i=0; i< m; i++)
	{
		for(int j=0; j< l; j++)
		{
			double temp;
			fd >> temp;
				D(i, j) = temp;
		}
	}
	fd.close();
	

	    int k = kperl*D.cols();
	    gettimeofday(&t2, NULL);
	    
	    OMPBATCHLLT(D, A, k, epsilon, tripletV);
	  	cout << "Myrank: "<< myrank << endl;
	  	
	    MPI_Barrier(MPI_COMM_WORLD);
	    
	    gettimeofday(&t3, NULL);
	    
		int Vnnzt = 0;
	   	int Vnnz = tripletV.size();
		MPI_Reduce(&Vnnz, &Vnnzt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			
		
		
		if(!myrank)
			{cout << l << " " << Vnnzt <<endl;
			cout<< "omptime  " << getTimeMs(t2,t3) << "totaltime "<< getTimeMs(t1,t3)<< endl; }
	    
	    
	if(of)
		{
			stringstream  sof;
			sof << argv[2] << "x" << l << "_" << npes << "_" << myrank;
		
			ofstream fout;
			fout.open(sof.str().c_str());
		
			if(!myrank)
				cout << "start writing to " << sof.str() << endl;
		
			for(int i=0;i<tripletV.size();i++)
			{
				fout << tripletV[i].row() << " "<<  tripletV[i].col()+stepn*myrank << " " << tripletV[i].value() << endl;		
			}
			fout.close();
			



		}


MPI_Finalize();
	return 0;
	
	
	}

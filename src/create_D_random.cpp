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



struct errorID
{
	double error;
	unsigned int id;
	
	errorID():
		error(0), id(0)
	{	};
	
	errorID(double error, unsigned int id):
		error(error), id(id)
	{	}; 
	
	static int compare(const errorID& a, const errorID& b)
	{
		return (int)(a.error > b.error);
	}
}errorID;


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

void randperm(unsigned int n, unsigned int perm[])
{
	unsigned int i, j, t;

	for(i=0; i<n; i++)
		perm[i] = i;
	
	for(i=0; i<n; i++) 
	{
		j = rand()%(n-i)+i;
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
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






int main(int argc, char*argv[])
{

	srand(0);
	
	
	if(argc!=10)
	{
		cout << "usage: ./omp infile outfile m n lmin lstep lmax epsilon verbos" << endl;
		cout << "lmin <= l < lmax" << endl;
		return -1;
	}
	
	int m = atoi(argv[3]);
	int n = atoi(argv[4]);
	int lmin = atoi(argv[5]);
	int lstep = atoi(argv[6]);
	int lmax = atoi(argv[7]);
	double epsilon = atof(argv[8]);
	verbos = atoi(argv[9]);
	
	timeval t1, t2;

	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);		
	
	int myn;
	int stepn = n/npes;
	
	if(myrank<npes-1)
		myn = (n)/npes;
	else
		myn = n-stepn*(npes-1);	
	

	MatrixXd A = MatrixXd::Zero(m, myn);	
	MatrixXd D = MatrixXd::Zero(m, lmin);
	vector<Trip> tripletV;
	
	cout<<"Start reading A"<<endl;
	stringstream  sofA;


	ifstream fr;
	fr.open(argv[1]);
	for(int i=0; i<m; i++)
		{
			for(int j=0; j<n; j++)
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
		
	for(int i=0;i<A.cols();i++)
	{
		if(A.col(i).norm() > 1E-3)
		{
			A.col(i) =  A.col(i) / A.col(i).norm();
		}
	}
	
	fr.close();
	
/* Create D randomly in lstep*/

	double global_error_D = 100;
	int l = lmin;
	int lold = 0;
	unsigned int *idx = new unsigned int[n];		
	if(!myrank)
	{
		randperm(n, idx);
	}
	MPI_Bcast(idx, n, MPI_INT, 0, MPI_COMM_WORLD);
	
	while (global_error_D>(epsilon*epsilon)&& l<lmax+1)
		{	
	
			D.conservativeResize(m, l);
		
			for(int i=lold; i<l; i++)
			{
				if(idx[i]/stepn == myrank)
				{
					D.col(i) = A.col(idx[i]-myrank*stepn)/A.col(idx[i]-myrank*stepn).norm();
				
					assert((D.col(i).norm() - 1) < 1E-3);
				}
				int root = idx[i]/stepn;
				MPI_Bcast(D.col(i).data(), D.col(i).size(), MPI_DOUBLE, root, MPI_COMM_WORLD);
			}
			double error_D = 0;
	  
			MatrixXd x_a = D.jacobiSvd(ComputeThinU | ComputeThinV).solve(A);
			MatrixXd tmp;
			for (int colid=0; colid<myn; colid++)
			{
				tmp = D*x_a.col(colid)-A.col(colid);
				error_D = error_D +(tmp.transpose()*tmp)(0,0);
				}
			
	
			
			MPI_Barrier(MPI_COMM_WORLD);

			MPI_Allreduce(&error_D, &global_error_D, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			
			global_error_D = global_error_D/n;
			
			lold = l;
			l = l + lstep;
	
		}
		

		if(!myrank)
			{		
				stringstream  sofD;
				sofD << argv[2] << "x" << D.cols() << "_D";
				
				ofstream foutD;
				foutD.open(sofD.str().c_str());
				
				cout << "start writing to " << sofD.str() << endl;
				
				foutD << D;
				
				foutD.close();
			}


	MPI_Finalize();
	return 0;
	
}

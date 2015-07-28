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


typedef struct errorID
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


void findIdx(int n, MatrixXd &A, MatrixXd& D, unsigned int * idx)
{
	int npes, myrank;
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	int myn = A.cols();
	int stepn = n/npes;
	
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	
	errorID test;
	
	MPI_Datatype ERROR_ID_type;
	int array_of_blocklengths[2] = {1, 1};
	MPI_Aint array_of_displacements[2];
	array_of_displacements[0] = 0;
	array_of_displacements[1] = sizeof(double);
	MPI_Datatype array_of_types[2] = {MPI_DOUBLE, MPI_UNSIGNED};
	MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &ERROR_ID_type);
	MPI_Type_commit(&ERROR_ID_type);
		
	std::vector<errorID> vError_local(myn);
	std::vector<errorID> vError(n);
	
	MatrixXd GD = (D.transpose()*D);
	MatrixXd E =  D*GD.inverse()*D.transpose()*A - A;	
	

	for(unsigned  j=0;j<A.cols();j++)
	{
		vError_local[j] = errorID(E.col(j).norm(), j+myrank*stepn); 
		//cout << myrank << ": " << vError_local[j].id << "-> " <<  vError_local[j].error << endl;	
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(&vError_local.front(), myn, ERROR_ID_type,
               &vError.front(), myn, ERROR_ID_type,
               0, MPI_COMM_WORLD);
	           
    if(!myrank)
    {           
		sort(vError.begin(), vError.end(), errorID::compare);
		//cout << "errors" << endl;
		for(unsigned i = 0; i < n; i++)
		{
			idx[i] = vError[i].id;
			//cout << vError[i].id << ": " <<  vError[i].error << endl;
		}
	}

	MPI_Bcast(idx, n, MPI_INT, 0, MPI_COMM_WORLD);
}


int main(int argc, char*argv[])
{
	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	srand(0);
	
	if(argc!=10)
	{
		cout << "usage: ./omp infile outfile m n lmin lstep lmax epsilon verbos" << endl;
		cout << "lmin <= l < lmax" << endl;
		MPI_Finalize();
		return -1;
	}
	
	int m = atoi(argv[3]);
	int n = atoi(argv[4]);
	int lmin = atoi(argv[5]);
	int lstep = atoi(argv[6]);
	int lmax = atoi(argv[7]);
	double epsilon = atof(argv[8]);
	//int ncpu = atoi(argv[9]);
	verbos = atoi(argv[9]);

	
	//Eigen::setNbThreads(ncpu);
//	Eigen::initParallel();

	timeval t1, t2;
	
	int myn;
	int stepn = n/npes;
	
	if(myrank<npes-1)
		myn = (n)/npes;
	else
		myn = n-stepn*(npes-1);	
			
	
	MatrixXd A = MatrixXd::Zero(m, myn);
	MatrixXd D;
	vector<Trip> tripletV;
	
	
	unsigned int *idx = new unsigned int[n];		
	if(!myrank)
	{
		randperm(n, idx);
	}
	MPI_Bcast(idx, n, MPI_INT, 0, MPI_COMM_WORLD);

		
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
	
/* Create D adaptively in lstep*/

	D = MatrixXd::Zero(m, 0);
	
	double global_error_D = 100;
	int lold = 0;
	int l = lmin;
	
	while (global_error_D>(epsilon*epsilon) && l<lmax+1)
		{	
			if(l!=lmin)
			{
				findIdx(n, A, D, idx); 
			}
			
			D.conservativeResize(m, l);
			
			for(int i=lold; i<l; i++)
				{
					if(idx[i-lold]/stepn == myrank)
						{
							cout<<"myrank = "<<myrank<<" idx= "<<idx[i-lold]<<endl;
							D.col(i) = A.col(idx[i-lold]-myrank*stepn);
					
							assert((D.col(i).norm() - 1) < 1E-3);
						}
					int root = idx[i-lold]/stepn;
					MPI_Bcast(D.col(i).data(), D.col(i).size(), MPI_DOUBLE, root, MPI_COMM_WORLD);
				}
			
			double error_D = 0;
			
			//MatrixXd x_a = D.jacobiSvd(ComputeThinU | ComputeThinV).solve(A);
			MatrixXd x_a = D.jacobiSvd(ComputeThinU | ComputeThinV).solve(A);
			
			MatrixXd tmp;
			for (int colid=0; colid<myn; colid++)
			{
				tmp = D*x_a.col(colid)-A.col(colid);
				error_D = error_D +(tmp.transpose()*tmp)(0,0);
				}
			
	
			cout<<"myrank = "<<myrank<<"  l = "<<l<<" err = "<<error_D<<endl;
			MPI_Barrier(MPI_COMM_WORLD);

			MPI_Allreduce(&error_D, &global_error_D, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			
			global_error_D = global_error_D/n;
			if(!myrank) cout<<"l = "<<l<<" err = "<<global_error_D<<endl;
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
		
	delete[] idx;
	MPI_Finalize();
	return 0;
}

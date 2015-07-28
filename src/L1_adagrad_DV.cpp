#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <vector>
#include <cstdint>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <mpi.h>

using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> Trip;

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


int verbos = 0;

inline void printPercent(int i, int n)
{
	if(verbos && ( (int)((100.0*(i-1)) / n) < (int)((100.0*i) / n)))
	{
		cout << "\r\033[K"; // erase line
		cout << (int)((100.0*i) / n) << "%"<<std::flush;
	}
}
int L1_Adagrad_dense(MatrixXd& V, MatrixXd& D, MatrixXd& y, MatrixXd& Xopt, VectorXd& err, int MaxIt, double lamda)
{
	

	
    int npes, myrank;
	
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    int eta = 1;
    int m = D.rows();
    int myn = V.cols();
    int l = D.cols();
    int ns = y.cols();
    

    
    if (!myrank)
		{cout<<"m = "<<m<<endl;
		cout<<"myn = "<<myn<<endl;
		cout<<"l = "<<l<<endl;
		cout<<"ns = "<<ns<<endl;
		cout<<"y.rows = "<<y.rows()<<endl;
	}
    Xopt = MatrixXd::Zero(myn,ns);

    err = VectorXd::Zero(MaxIt);
	
	MatrixXd G = MatrixXd::Zero(myn,ns);
	MatrixXd Gsqrt = MatrixXd::Zero(myn,ns);
	MatrixXd grad = MatrixXd::Zero(myn,ns);
	MatrixXd g = MatrixXd::Zero(myn,ns);
	MatrixXd u = MatrixXd::Zero(myn,ns);
	
    MatrixXd proxy = MatrixXd::Zero(myn, ns);
    MatrixXd temp0 = MatrixXd::Zero(l, ns);
    MatrixXd Vx = MatrixXd::Zero(l, ns);
  
	MatrixXd b = D.transpose() * y;

    b = V.transpose() * b;
   
    MatrixXd DtD = D.transpose() * D;
    
    
    for(int t = 1; t < MaxIt; t++)
    {
		//if(!myrank) cout <<"in L1"<<endl;
        temp0 = V * Xopt;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, temp0.data(), temp0.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        temp0 = DtD * temp0;
        proxy = V.transpose() * temp0; 
        g = proxy - b;
        G = G + g.cwiseProduct(g);
        u = u + g;
		g = u/(t+1);		

        for(int tt=0; tt<ns; tt++)
        {
            for (int j = 0; j < myn; j++)
            {
                if (abs(g(j, tt)) <= lamda)
					{
						Xopt(j,tt) = 0;
					}
					else
					{
						Xopt(j,tt) = -1*(copysign(1,u(j,tt))* eta * ((t+1)/sqrt(G(j,tt))) * (abs(g(j,tt)) - lamda));
					}
            }

        }
        if(t%10==0)
        {
    		temp0 = V * Xopt;
    		MatrixXd Vx = temp0;
        	MPI_Barrier(MPI_COMM_WORLD);
        	MPI_Reduce(temp0.data(), Vx.data(), temp0.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        	if(!myrank)
			{
            	err(t%10) = (D*Vx - y).norm()/(y).norm();
				cout<<"err("<<t<<") = "<<err(t%10)<<endl;
            }
			printPercent(t, MaxIt);
		}
    }
    return 0;
}


int L1_Adagrad(SparseMatrix<double>& V, MatrixXd& D, MatrixXd& y, MatrixXd& Xopt, VectorXd& err, int MaxIt, double lamda)
{
	

	timeval t4,t5;
    
    int npes, myrank;
	
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if(!myrank)
    {
		
		gettimeofday(&t4, NULL);
		
	}
    
    int eta = 1;
    int m = D.rows();
    int myn = V.cols();
    int l = D.cols();
    int ns = y.cols();
    

    
    if (!myrank)
		{cout<<"m = "<<m<<endl;
		cout<<"myn = "<<myn<<endl;
		cout<<"l = "<<l<<endl;
		cout<<"ns = "<<ns<<endl;
		cout<<"y.rows = "<<y.rows()<<endl;
	}
    Xopt = MatrixXd::Zero(myn,ns);

    err = VectorXd::Zero(MaxIt);
	
	MatrixXd G = MatrixXd::Zero(myn,ns);
	MatrixXd Gsqrt = MatrixXd::Zero(myn,ns);
	MatrixXd grad = MatrixXd::Zero(myn,ns);
	MatrixXd g = MatrixXd::Zero(myn,ns);
	MatrixXd u = MatrixXd::Zero(myn,ns);
	
    MatrixXd proxy = MatrixXd::Zero(myn, ns);
    MatrixXd temp0 = MatrixXd::Zero(l, ns);
    MatrixXd Vx = MatrixXd::Zero(l, ns);
  
	MatrixXd b = D.transpose() * y;

    b = V.transpose() * b;
   
    MatrixXd DtD = D.transpose() * D;
    
    if(!myrank)
    {
		
		gettimeofday(&t5, NULL);
		
	}
    for(int t = 1; t < MaxIt; t++)
    {
		//if(!myrank) cout <<"in L1"<<endl;
        temp0 = V * Xopt;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, temp0.data(), temp0.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        temp0 = DtD * temp0;
        proxy = V.transpose() * temp0; 
        g = proxy - b;
        G = G + g.cwiseProduct(g);
        u = u + g;
		g = u/(t+1);		

        for(int tt=0; tt<ns; tt++)
        {
            for (int j = 0; j < myn; j++)
            {
                if (abs(g(j, tt)) <= lamda)
					{
						Xopt(j,tt) = 0;
					}
					else
					{
						Xopt(j,tt) = -1*(copysign(1,u(j,tt))* eta * ((t+1)/sqrt(G(j,tt))) * (abs(g(j,tt)) - lamda));
					}
            }

        }  
        

       if(t%200==0)
        {
    		temp0 = V * Xopt;
    		MatrixXd Vx = temp0;
        	MPI_Barrier(MPI_COMM_WORLD);
        	MPI_Reduce(temp0.data(), Vx.data(), temp0.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        	if(!myrank)
			{
            	err(t%200) = (D*Vx - y).norm()/(y).norm();
				cout<<"err("<<t<<") = "<<err(t%200)<<endl;
				//cout<<"delay-init"<<getTimeMs(t4,t5)<<endl;
            }
			printPercent(t, MaxIt);
		}
    }
    return 0;
}



int main(int argc, char*argv[])
{
	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    srand(time(0)) ;

	if(argc!=12)
	{
		if(!myrank)
		{
			cout << "usage: ./gdonDV Vfile Dfile yfile nf m n l ns MaxIt lambda verbos" << endl;
		}
        
		MPI_Finalize();
		return -1;
	}
	
	int nf = atoi(argv[4]);
	int m = atoi(argv[5]);
	int n = atoi(argv[6]);
	int l = atoi(argv[7]);
	int ns = atoi(argv[8]);
    int MaxIt = atoi(argv[9]);
    double lamda = atof(argv[10]);
    verbos = atoi(argv[11]);
    
	timeval t1, t2, t3;
    
    int myn;
    int myshare;
	int stepn = n/npes;
	int file_step = nf/npes;
	
	if(myrank<npes-1)
	{
		myn = (n)/npes;
		myshare = nf/npes;
	}
	else
	{
		myn = n-stepn*(npes-1);
		myshare = nf-file_step*(npes-1);
	}	
	
    
	vector<Trip> tripletV;
    tripletV.clear();
    tripletV.reserve(l * myn);
    
    int q = n/nf;
    
  
    for(int ff=myrank*myshare; ff<myrank*myshare+myshare; ff++)
    {
		//cout<<"myrank = "<<myrank<<" myshare= "<<ff<<endl;
		stringstream sofV;
		sofV << argv[1]<<"_"<<ff;
		ifstream fr;
		fr.open(sofV.str().c_str());
    	if(!fr.is_open())
		{
			if(!myrank)
				cout << "File not found: " << sofV.str().c_str() << endl;
        
			MPI_Finalize();
			return -1;
		}
	
		if(!myrank)
			cout<< "Start reading V from " << sofV.str().c_str() <<endl;
			
		
		
		 gettimeofday(&t1, NULL);
		while(1)
		{
			unsigned int row,col;
			double value;		
			fr >> row >> col >> value;
			
			if(fr.eof())
				break;
			
			tripletV.push_back(Trip(row,col - q*ff,value));
		}
		fr.close();    	
    }
    
    int Vnnz = tripletV.size();
    
    SparseMatrix<double> V(l,myn);
    V.setFromTriplets(tripletV.begin(),tripletV.end());
	
    MatrixXd y = MatrixXd::Zero(m, ns) ;
    MatrixXd D = MatrixXd::Zero(m,l);
    ifstream fr0;
	fr0.open(argv[2]);
    if(!fr0.is_open())
	{
		if(!myrank)
			cout << "File not found: " << argv[2] << endl;
        
		MPI_Finalize();
		return -1;
	}
	
	if(!myrank)
		cout<< "Start reading D from " << argv[2] <<endl;
    
	for(int i=0; i<m; i++)
    {
        for(int j=0; j<l; j++)
        {
            if(!fr0.eof( ))
            {
                double temp;
                fr0 >> temp;
                D(i,j) = temp;
            }
        }
    }
    
	fr0.close();

	
    ifstream fr1;
	fr1.open(argv[3]);
    if(!fr1.is_open())
	{
		if(!myrank)
			cout << "File not found: " << argv[3] << endl;
        
		MPI_Finalize();
		return -1;
	}
	
	if(!myrank)
		cout<< "Start reading y from " << argv[3] <<endl;
    
	for(int i=0; i<m; i++)
    {
        for(int j=0; j<ns; j++)
        {
            if(!fr1.eof( ))
            {
                double temp;
                fr1 >> temp;
                y(i,j) = temp;
            }
        }
    }
    
	fr1.close();
	

	
		
    MatrixXd Xopt = MatrixXd::Zero(myn,ns);
    VectorXd err = VectorXd::Zero(MaxIt);
	gettimeofday(&t2, NULL);

/* Calling gd with Adagrad*/
	
		if((Vnnz/(l*myn))>.4)
		{
		MatrixXd  vs = MatrixXd(V);
		 L1_Adagrad_dense(vs, D, y, Xopt, err, MaxIt, lamda);
		}
	else{

		 L1_Adagrad(V, D, y, Xopt, err, MaxIt, lamda);
		}	
		
   

    
	gettimeofday(&t3, NULL);
    
    stringstream  sof;
    sof << argv[2] << "_Xopt_"<< lamda << "_" << npes << "_" << myrank;
    

    
    
    if(!myrank)
    {

        
        cout << " filetime " << getTimeMs(t1,t2) << endl; 
        cout << " sgdtime " << getTimeMs(t2,t3) << endl; 
        cout << " totaltime " << getTimeMs(t1,t3) << endl; 
        
        
    }
    
	MPI_Finalize();
	return 0;

}






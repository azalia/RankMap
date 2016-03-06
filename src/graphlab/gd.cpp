/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WI THOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */


/**
 * \file
 * 
 * \brief The main file for the ALS matrix factorization algorithm.
 *
 * This file contains the main body of the ALS matrix factorization
 * algorithm. 
 */
#include <Eigen/Dense>
#define EDGE_GROUP_ID

#include <sstream>
#include <vector>
#include <cstdlib>
/*
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
*/

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


// This file defines the serialization code for the eigen types.
#include "eigen_serialization.hpp"
#include "eigen_wrapper.hpp"
#include <graphlab.hpp>
#include <graphlab/util/stl_util.hpp>
#include "stats.hpp"


#include <graphlab/macros_def.hpp>

const int SAFE_NEG_OFFSET = 2; //add 2 to negative node id
//to prevent -0 and -1 which arenot allowed



#include "cosamp.hpp"


#define ABS(X) (((X)>0)?(X):(-(X)))


double soft_thereshold(double x, double t)
{
	if(ABS(x) < t)
		return 0;
	else if(x > 0)
		return (ABS(x) - t);
	else
		return -(ABS(x) - t);
}

/** 
 * \ingroup toolkit_matrix_factorization
 *
 * \brief the vertex data type which contains the latent factor.
 *
 * Each row and each column in the matrix corresponds to a different
 * vertex in the ALS graph.  Associated with each vertex is a factor
 * (vector) of latent parameters that represent that vertex.  The goal
 * of the ALS algorithm is to find the values for these latent
 * parameters such that the non-zero entries in the matrix can be
 * predicted by taking the dot product of the row and column factors.
 */
struct vertex_data {
	enum vertex_role_type {X, P, Y, Q, E};

	vertex_role_type role;
	
	/** \brief The number of times this vertex has been updated. */
	uint32_t nupdates;
	
	/** \brief t(k) in X and error in Y. */
	double tk;
	
	/** \brief xk in X, ytilda in Y, p in P, q in Q*/
	std::vector<double> x;
	
	/** \brief xkm1 in X, y in Y, p1 in P, q1 in Q*/
	std::vector<double> y;
	
	
	/** \brief zk in X*/
	std::vector<double> zk;
	
	
	vertex_data() : 
		role(E), nupdates(0), tk(1), x(0), y(0), zk(0)
	{  }
	
		
	vertex_data(std::vector<double>& v) : 
		role(vertex_data::Y), nupdates(0), tk(1), x(v.size()), y(v), zk(0)
	{	
		for(unsigned int i=0;i<x.size();i++)
		{
			x[i] = 0;//((rand()%256)/256.0);
		}
	
	}
	

	vertex_data(vertex_role_type role) : 
		role(role), nupdates(0), tk(1), x(0), y(0), zk(0)
	{	}
	
	vertex_data(const vertex_data &other) : 
		role(other.role), nupdates(other.nupdates), tk(1), x(other.x), y(other.y), zk(other.zk)
	{	}
	
	/** \brief Save the vertex data to a binary archive */
	void save(graphlab::oarchive& arc) const 
	{
		arc << nupdates << tk << role << x << y << zk;
	}
	
	/** \brief Load the vertex data from a binary archive */
	void load(graphlab::iarchive& arc) 
	{
		arc >> nupdates >> tk >> role >> x >> y >> zk;
	}
	
	//~vertex_data()
	//{	}
	
	
}; // end of vertex data

char roleToChar(vertex_data::vertex_role_type role)
{
	switch(role)
	{
	case vertex_data::P:
			return 'P';
	case vertex_data::X:
			return 'X';
	case vertex_data::Y:
			return 'Y';
	case vertex_data::Q:
			return 'Q';
	case vertex_data::E:
		return 'E';
	}
	return ' ';
}

/**
 * \brief The edge data stores the entry in the matrix.
 *
 * In addition the edge data also stores the most recent error estimate.
 */
struct edge_data {
	/** \brief V or W */
	double coeff;
	int group_id;
	
	edge_data() : 
		coeff(0), group_id(0)
	{	}
	
	/** \brief basic initialization */
	edge_data(double coeff, int group_id) : 
		coeff(coeff), group_id(group_id)
	{	}
	
	edge_data(const edge_data &other) :
		coeff(other.coeff), group_id(other.group_id)
	{	}
	
	
	/** \brief Save the edge data to a binary archive */
	void save(graphlab::oarchive& arc) const
	{
		arc << coeff << group_id;
	}

	/** \brief Load the edge data from a binary archive */
	void load(graphlab::iarchive& arc) 
	{
		arc >> coeff >> group_id ;
	}

}; // end of edge data


/**
 * \brief The graph type is defined in terms of the vertex and edge
 * data.
 */ 
typedef graphlab::distributed_graph<vertex_data, edge_data> graph_type;

//#include "implicit.hpp"


/*stats_info count_edges(const graph_type::edge_type & edge){
  stats_info ret;

  if (edge.data().role == edge_data::TRAIN)
     ret.training_edges = 1;
  else if (edge.data().role == edge_data::VALIDATE)
     ret.validation_edges = 1;
  ret.max_user = (size_t)edge.source().id();
  ret.max_item = (-edge.target().id()-SAFE_NEG_OFFSET);
  return ret;
}*/



/**
 * \brief Given a vertex and an edge return the other vertex in the
 * edge.
 */
inline graph_type::vertex_type
get_other_vertex(graph_type::edge_type& edge, 
                 const graph_type::vertex_type& vertex) {
  return vertex.id() == edge.source().id()? edge.target() : edge.source();
}; // end of get_other_vertex




class gather_type {
public:

	/**
	* \brief if updating X, dot is v(i,j)*x(j). if updating P, dot is w(i,j)*p(j)
	*/
	std::vector<double> dot;

	gather_type(): 
		dot(0)
	{ }
	
	
	gather_type( const gather_type& other ) : 
		dot(other.dot)
	{	}
	
	
	/**
	* \brief This constructor computes dot
	*/
	gather_type(const std::vector<double>& a, const double b) : 
		dot(a.size())
	{	
	
		//std::cout << "gather_type() " << " len = " << a.size() << " ";
		for(unsigned int i = 0 ; i< a.size() ; i++)
		{
			dot[i] = a[i] * b;	
			//std::cout << dot[i] << " ";		
		}
		//std::cout << std::endl;
	} 
	
	
	gather_type(const std::vector<double>& ytilda, const std::vector<double>& y , const double b) : 
		dot(ytilda.size())
	{	
		ASSERT_TRUE(ytilda.size()==y.size());
	
		//std::cout << "gather_type() " << " len = " << a.size() << " ";
		for(unsigned int i = 0 ; i< ytilda.size() ; i++)
		{
			dot[i] = (ytilda[i] - y[i]) * b;	
			//std::cout << dot[i] << " ";		
		}
		//std::cout << std::endl;
	} 

	
	/** \brief Save the values to a binary archive */
	void save(graphlab::oarchive& arc) const 
	{ 
		arc << (unsigned int)dot.size();
		for(unsigned int i = 0 ; i< dot.size() ; i++)
		{
			arc << dot[i];			
		}
	}

	/** \brief Read the values from a binary archive */
	void load(graphlab::iarchive& arc) 
	{ 
		unsigned int dot_len = 0;
		arc >> dot_len;
		dot.resize(dot_len,0);
		for(unsigned int i = 0 ; i< dot_len ; i++)
		{
			arc >> dot[i];			
		}
	}

	/**
	* \brief Computes XtX += other.XtX and Xy += other.Xy updating this
	* tuples value
	*/
	gather_type& operator +=(const gather_type& other)
	{
		ASSERT_TRUE(dot.size() == other.dot.size() || other.dot.size()==0 || dot.size() == 0);
		
		if(other.dot.size()==0)
			return *this;
		
		dot.resize(other.dot.size());
		for(unsigned int i = 0 ; i< dot.size() ; i++)
		{
			dot[i] += other.dot[i];
		}
		return *this;
	} // end of operator+=
	
	//~gather_type()
	//{	}
	
}; // end of gather type


class factor_vertex_program :
  public graphlab::ivertex_program<graph_type, gather_type,graphlab::messages::sum_priority>,
  public graphlab::IS_POD_TYPE {
public:
	/** The convergence tolerance */
	static size_t MAX_UPDATES;
	static double GAMMA;
	static int Q_OFFSET;
	static double DIFF_THRESHOLD;
	static double LAMBDA;
	
	/** The set of edges to gather along */
	edge_dir_type gather_edges(icontext_type& context,
				const vertex_type& vertex) const 
	{
		return graphlab::IN_EDGES;
	};

	/** The gather function computes XY */
	gather_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const
	{
		const vertex_data& other_vdata = get_other_vertex(edge, vertex).data();
		
		if(other_vdata.role == vertex_data::Y)
		{
			return gather_type(other_vdata.x, other_vdata.y , edge.data().coeff);
		}
		else if(other_vdata.role == vertex_data::X)
		{
			return gather_type(other_vdata.zk , edge.data().coeff);
		}
		
		return gather_type(other_vdata.x, edge.data().coeff);	
	} // end of gather function

	/** apply collects the sum of XY */
	void apply(icontext_type& context, vertex_type& vertex,
		const gather_type& sum) {

		// Get and reset the vertex data
		vertex_data& vdata = vertex.data();
		
		if(vdata.role == vertex_data::X)
		{
			if(vdata.x.size()==0)//initialization
			{
				vdata.x.resize(sum.dot.size());
			}
			if(vdata.y.size()==0)//initialization
			{
				vdata.y.resize(sum.dot.size());
			}
			if(vdata.zk.size()==0)//initialization
			{
				vdata.zk.resize(sum.dot.size());
			}
			
			double tkp1 = (1 + std::sqrt(1 + 4 * (vdata.tk * vdata.tk))) / 2;

			for(unsigned int i=0;i< vdata.x.size();i++)
			{
				vdata.y[i] = vdata.x[i];
				
				vdata.x[i] = soft_thereshold(vdata.zk[i] - GAMMA*(sum.dot[i]),LAMBDA*GAMMA);

				vdata.zk[i] = vdata.x[i] + (vdata.tk - 1)*(vdata.x[i] - vdata.y[i])/(tkp1);
				
				
			}
			
			vdata.tk = tkp1;
		}
		else if(vdata.role == vertex_data::Y)
		{
			double error = 0;
			double norm = 0;
			for(unsigned int i = 0; i < vdata.x.size(); i++)
			{
				vdata.x[i] = sum.dot[i];
				error +=(vdata.y[i] - vdata.x[i])*(vdata.y[i] - vdata.x[i]);
				norm += (vdata.y[i])*(vdata.y[i]); 
			}
			
			vdata.tk = std::sqrt(error/norm); // error
		}
		else
		{
			vdata.y = vdata.x;
			vdata.x = sum.dot;
		}
		
		vdata.nupdates++;
		
	} // end of apply
  
	/** The edges to scatter along */
	edge_dir_type scatter_edges(icontext_type& context,
				const vertex_type& vertex) const 
	{
		const vertex_data& vdata = vertex.data();
		if(vdata.role == vertex_data::Y)
		{
			if( vdata.tk  < DIFF_THRESHOLD || vdata.nupdates > MAX_UPDATES)
				return graphlab::NO_EDGES;
		}	
		else if(vdata.x.size() == vdata.y.size())
		{	
			double error = 0;
			double norm = 0;
			for(unsigned int i = 0 ; i< vdata.x.size() ; i++)
			{
				error +=(vdata.y[i] - vdata.x[i])*(vdata.y[i] - vdata.x[i]);
				norm += (vdata.y[i])*(vdata.y[i]); 
			}
			
			if( std::sqrt(error/norm) < DIFF_THRESHOLD )
				return graphlab::NO_EDGES;
		}
		
		return graphlab::OUT_EDGES;
		
	}; // end of scatter edges

	/** Scatter reschedules neighbors */
	void scatter(icontext_type& context, const vertex_type& vertex,
		edge_type& edge) const 
	{
		const vertex_type other_vertex = get_other_vertex(edge, vertex);
		

		ASSERT_TRUE(vertex.data().role != vertex_data::X || other_vertex.data().role == vertex_data::P);
		ASSERT_TRUE(vertex.data().role != vertex_data::P || other_vertex.data().role == vertex_data::Y);
		ASSERT_TRUE(vertex.data().role != vertex_data::Y || other_vertex.data().role == vertex_data::Q);
		ASSERT_TRUE(vertex.data().role != vertex_data::Q || other_vertex.data().role == vertex_data::X);
		
		context.signal(other_vertex);
	} // end of scatter function


	/**
	* \brief Signal all vertices on one side of the bipartite graph
	*/
	static graphlab::empty signal_P(icontext_type& context,
					const vertex_type& vertex) 
	{
		if(vertex.data().role == vertex_data::Q)
			context.signal(vertex);
		return graphlab::empty();
	} // end of signal_P

}; // end of gd vertex program


graph_type::vertex_id_type getID(vertex_data::vertex_role_type role, int id)
{
	id++;
	if(role == vertex_data::X)
	{
		return id;
	}
	else if(role == vertex_data::Y)
	{
		return -id  - SAFE_NEG_OFFSET;
	}
	else if(role == vertex_data::P)
	{
		return id + factor_vertex_program::Q_OFFSET;
	}
	else if(role == vertex_data::Q)
	{
		return -id - factor_vertex_program::Q_OFFSET;
	}
	
	return -1;
}


/**
 * \brief The graph loader function is a line parser used for
 * distributed graph construction.
 */
inline bool graph_edge_vertexPQ_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
	
	ASSERT_FALSE(line.empty());

	
	//int group_id = -1;
	double value;
	std::stringstream sline(line);
	

	
	if(hasEnding(filename,".d"))
	{
		int id1, id2;
		sline >> id1 >> id2 /*>> group_id*/ >> value;
		
		
		vertex_data vdataP(vertex_data::P);
		vertex_data vdataQ(vertex_data::Q);
		
		
		graph.add_edge(getID(vertex_data::P, id2), getID(vertex_data::Y, id1), edge_data(value,(int)(getID(vertex_data::Y, id1))));	
		graph.add_edge(getID(vertex_data::Y, id1), getID(vertex_data::Q, id2), edge_data(value,(int)(getID(vertex_data::Y, id1))));	
		
		//std::cout << "edge p y " << id2 << " " << id1 << std::endl;
		//std::cout << "edge y q " << id1 << " " << id2 << std::endl;
		
		
		graph.add_vertex(getID(vertex_data::P, id2), vdataP);
		graph.add_vertex(getID(vertex_data::Q, id2), vdataQ); 
		
		//std::cout << "vertex p " << id2 << std::endl;
		//std::cout << "vertex q " << id2 << std::endl;
		
		
		//std::cout << " P " << (int)getID(vertex_data::P, id1) << std::endl;
		//std::cout << " Q " << (int)getID(vertex_data::Q, id1) << std::endl;

		
	}
	else if(hasEnding(filename, ".v"))
	{
		int id1, id2;
		sline >> id1 >> id2 /*>> group_id*/ >> value;;
		
		
		
		graph.add_edge(getID(vertex_data::X, id2), getID(vertex_data::P, id1), edge_data(value,(int)(getID(vertex_data::X, id2))));	
		graph.add_edge(getID(vertex_data::Q, id1), getID(vertex_data::X, id2), edge_data(value,(int)(getID(vertex_data::X, id2))));	
		
		
		//std::cout << "edge q x " << id1 << " " << id2 << std::endl;
		//std::cout << "edge x p " << id2 << " " << id1 << std::endl;
		
		
		//std::cout << " need P " << (int)getID(vertex_data::P, id2) << std::endl;
		
		
		vertex_data vdataX(vertex_data::X);
		
		
		graph.add_vertex(getID(vertex_data::X, id2), vdataX);
		
		
		//std::cout << "vertex x " << id2 << std::endl;
	}

	return true;

} // end of graph_loader



/**
 * \brief The graph loader function is a line parser used for
 * distributed graph construction.
 */
inline bool graph_vertexX_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
	
	ASSERT_FALSE(line.empty());
	
	if(hasEnding(filename,".y"))
	{
		unsigned int len;
		int id;
		std::vector<double> y;
		std::stringstream sline(line);
		
		sline >> id >> len;
	
		y.resize(len,0);
		for(unsigned int i = 0 ; i < len; i++)
		{
			sline >> y[i]; 
		}	
		
		vertex_data vdataY(y);
		
		//add vertex X
		graph.add_vertex(getID(vertex_data::Y, id), vdataY); 
		//std::cout << "graph.add_vertex(vertex_id, vdataX); " << (int)vertex_id  << " len = " << vdataX.x.size() << std::endl; 
		
	}
	return true;

}  // end of graph_loader




/**
 * \brief The error aggregator is used to accumulate the overal
 * prediction error.
 *
 * The error aggregator is itself a "reduction type" and contains the
 * two static methods "map" and "finalize" which operate on
 * error_aggregators and are used by the engine.add_edge_aggregator
 * api.
 */
struct error_aggregator : 
	public graphlab::IS_POD_TYPE 
{
	typedef factor_vertex_program::icontext_type icontext_type;
	typedef graph_type::vertex_type vertex_type;

	double delta;
	double xnorm;
	unsigned int xnnz;
	unsigned int iter;
	
	error_aggregator() : 
		delta(0), xnorm(0), xnnz(0), iter(0)
	{ }

	error_aggregator& operator+=(const error_aggregator& other) 
	{
		delta += other.delta;
		xnorm += other.xnorm;
		xnnz += other.xnnz;
		iter = (other.iter > iter)? other.iter: iter;
		return *this;
	}

	static error_aggregator map(icontext_type& context, const graph_type::vertex_type& vertex) 
	{
		error_aggregator agg;
		if(vertex.data().role == vertex_data::Y) 
		{
			double error = 0; 
			double norm = 0; 
			//std::cout << (int)vertex.id() << " len =  " << vertex.data().x.size();
			for(unsigned int i=0;i<vertex.data().x.size();i++)
			{
				//std::cout << vertex.data().x[i] << " " << vertex.data().oldX[i] << " ";
				error += (vertex.data().x[i] - vertex.data().y[i])*(vertex.data().x[i] - vertex.data().y[i]);
				norm += (vertex.data().y[i]) * (vertex.data().y[i]);
			}
			agg.delta = error; 
			agg.xnorm = norm;

			//std::cout << std::endl;
			//context.cout() << (int)vertex.id() <<" agg.error = " << agg.error << std::endl;
		}
		else if(vertex.data().role == vertex_data::X) 
		{
			agg.iter = vertex.data().nupdates;
			agg.xnnz = 0;
			for(unsigned int i=0;i<vertex.data().x.size();i++)
				if(ABS(vertex.data().x[i]) > 1e-20)
					agg.xnnz++;
		}
		return agg;
	}

	static void finalize(icontext_type& context, const error_aggregator& agg) 
	{
		const double delta = (agg.delta);
		const double xnorm = (agg.xnorm);
		double error = delta / xnorm;
		//context.cout() << "Iteration = " << context.iteration() 
		//				<< "\tTime = " << context.elapsed_seconds() 
		//				<< " s\t||AX - Y||2/||Y||2 = " << error
		//				<< "\tnnz(X) = " << agg.xnnz << std::endl;
		
		context.cout() << "REPORT :( " << agg.iter << ", " 
			<< context.elapsed_seconds()  << ", "
			<< error << ", "
			<< agg.xnnz << " )" 
			<< std::endl;

		
		//factor_vertex_program::LAMBDA = factor_vertex_program::LAMBDA * 0.95;
		
		if(error < 1e-8)
			context.stop();
	}
}; // end of error aggregator




/**
 * \brief The X saver is used by the graph.save routine to
 * output the final X vector back to the filesystem.
 */
struct linear_model_saver_Y
{
	typedef graph_type::vertex_type vertex_type;
	typedef graph_type::edge_type   edge_type;
	// save the linear model, using the format:
	//   nodeid) factor1 factor2 ... factorNLATENT \n
	//
	std::string save_vertex(const vertex_type& vertex) const
	{
		std::stringstream ss;
		if (vertex.data().role == vertex_data::Y)
		{
			ss << (int)vertex.id() << " " << (unsigned int)vertex.data().x.size() << " ";
			
			for(unsigned int i=0;i<vertex.data().x.size();i++)
				ss << vertex.data().x[i] << " ";
				
			ss << std::endl;
			return ss.str();
		}
		return "";
	}
		
	std::string save_edge(const edge_type& edge) const
	{
		return "";
	}
}; 


typedef graphlab::omni_engine<factor_vertex_program> engine_type;


double factor_vertex_program::DIFF_THRESHOLD = 0;
size_t factor_vertex_program::MAX_UPDATES = -1;
double factor_vertex_program::GAMMA = 0.000001;
int factor_vertex_program::Q_OFFSET = -(((unsigned)-1)>>1); // TODO
double factor_vertex_program::LAMBDA = 0;

int main(int argc, char** argv) 
{
	global_logger().set_log_level(LOG_INFO);
	global_logger().set_log_to_console(true);

	float interval = 5; // sec for aggregate error
	// Parse command line options -----------------------------------------------
	const std::string description =
	"Compute GD.";
	graphlab::command_line_options clopts(description);
	std::string input_dir;
	std::string result;
	std::string exec_type = "synchronous";
	clopts.attach_option("matrix", input_dir,
			"The directory containing the matrix file");
	clopts.add_positional("matrix");

	clopts.attach_option("diff_th", factor_vertex_program::DIFF_THRESHOLD,
			"Min normalized different between X and previous X to halt vertex X");

	clopts.attach_option("gamma", factor_vertex_program::GAMMA,
			"Step size in GD algorithm");
			
	clopts.attach_option("max_iter", factor_vertex_program::MAX_UPDATES,
			"The maximum number of updates allowed for a vertex");
	clopts.attach_option("result", result,
                       "The prefix (folder and file name) to save result.");
	clopts.attach_option("interval", interval,
                       "The interval time in second between reporting Norm(x - oldX).");   
	clopts.attach_option("lambda", factor_vertex_program::LAMBDA,
                       "lambda L1 norm");   
	//parse_implicit_command_line(clopts);

	if(!clopts.parse(argc, argv) || input_dir == "") {
		std::cout << "Error in parsing command line arguments." << std::endl;
		clopts.print_description();
		return EXIT_FAILURE;
	}



	///! Initialize control plain using mpi
	graphlab::mpi_tools::init(argc, argv);
	graphlab::distributed_control dc;

	dc.cout() << "Loading graph" << std::endl;
	graphlab::timer timer;
	graph_type graph(dc, clopts);
	
	//add Q
	graph.load(input_dir, graph_vertexX_loader);
	graph.load(input_dir, graph_edge_vertexPQ_loader);
	

	dc.cout() << "Loading graph. Finished in " << timer.current_time() << std::endl;

	///! what is going on here?!
	//if (dc.procid() == 0)
	//  add_implicit_edges<edge_data>(implicitratingtype, graph, dc);

	dc.cout() << "Finalizing graph." << std::endl;
	timer.start();
	graph.finalize();
	dc.cout() << "Finalizing graph. Finished in "
		<< timer.current_time() << std::endl;

	dc.cout() << "Creating engine" << std::endl;
	engine_type engine(dc, graph, exec_type, clopts);

	// Add error reporting to the engine
	const bool success = engine.add_vertex_aggregator<error_aggregator>	("error", error_aggregator::map, error_aggregator::finalize) 
						&&	engine.aggregate_periodic("error", interval);
	
	ASSERT_TRUE(success);


	// Signal all vertices on the vertices on the left (liberals)
	engine.map_reduce_vertices<graphlab::empty>(factor_vertex_program::signal_P);
	//info = graph.map_reduce_edges<stats_info>(count_edges);
	//dc.cout()<<"Training edges: " << info.training_edges << " validation edges: " << info.validation_edges << std::endl;

	// Run GD ---------------------------------------------------------
	dc.cout() << "Running GD" << std::endl;
	timer.start();
	engine.start();

	const double runtime = timer.current_time();
	dc.cout() << "----------------------------------------------------------"
		<< std::endl
		<< "Final Runtime (seconds):   " << runtime
		<< std::endl
		<< "Updates executed: " << engine.num_updates() << std::endl
		<< "Update Rate (updates/second): "
		<< engine.num_updates() / runtime << std::endl;

	// Compute the final training error -----------------------------------------
	dc.cout() << "Final error: " << std::endl;
	engine.aggregate_now("error");

	// Make predictions ---------------------------------------------------------
	if(!result.empty()) {
		std::cout << "Saving X" << std::endl;
		const bool gzip_output = false;
		const bool save_vertices = true;
		const bool save_edges = false;
		const size_t threads_per_machine = 1;

		//save the linear model
		graph.save(result + ".x", linear_model_saver_Y(),
			gzip_output, save_vertices, save_edges, threads_per_machine);

	}

	graphlab::mpi_tools::finalize();
	return EXIT_SUCCESS;
} // end of main




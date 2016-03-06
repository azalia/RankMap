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
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
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
	if(x > t)
		return (x - t);
	else if(x < -1 * t)
		return (x + t);
		
	return 0;
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

	enum vertex_role_type {X, P, Q};

	vertex_role_type role;
	
	/** \brief The number of times this vertex has been updated. */
	uint32_t nupdates;

	/** \brief x */
	double x;
	double oldX;
	
	/** \brief AtY */
	double aty;
	
	vertex_data() : 
		role(X), nupdates(0) , x(0), oldX(0), aty(0)
	{  }
	
		
	vertex_data(double x, double aty) : 
		role(X), nupdates(0) , x(x), oldX(x), aty(aty)
	{	}

	vertex_data(vertex_role_type role) : 
		role(role), nupdates(0) , x(0), oldX(0), aty(0)
	{  }
	
	vertex_data(const vertex_data &other) : 
		role(other.role), nupdates(other.nupdates) , x(other.x), oldX(other.oldX), aty(other.aty)
	{	}
	
	/** \brief Save the vertex data to a binary archive */
	void save(graphlab::oarchive& arc) const 
	{
		arc << nupdates << x << oldX  << aty << role;
	}
	
	/** \brief Load the vertex data from a binary archive */
	void load(graphlab::iarchive& arc) 
	{
		arc >> nupdates >> x >> oldX >> aty >> role;
	}
	
}; // end of vertex data

char roleToChar(vertex_data::vertex_role_type role)
{
	switch(role)
	{
	case vertex_data::P:
			return 'P';
	case vertex_data::X:
			return 'X';
	case vertex_data::Q:
			return 'Q';
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
	double dot;

	gather_type(): 
		dot(0)
	{ }
	
	
	gather_type( const gather_type& other ) : 
		dot(other.dot)
	{	}
	
	
	/**
	* \brief This constructor computes dot
	*/
	gather_type(const double a, const double b) : 
		dot(a*b)
	{	} 
	

	
	/** \brief Save the values to a binary archive */
	void save(graphlab::oarchive& arc) const 
	{ 
		arc << dot ;
	}

	/** \brief Read the values from a binary archive */
	void load(graphlab::iarchive& arc) 
	{ 
		arc >> dot;
	}

	/**
	* \brief Computes XtX += other.XtX and Xy += other.Xy updating this
	* tuples value
	*/
	gather_type& operator +=(const gather_type& other)
	{
		dot += other.dot;
		return *this;
	} // end of operator+=
	
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
	
	/** The set of edges to gather along */
	edge_dir_type gather_edges(icontext_type& context,
				const vertex_type& vertex) const 
	{
		return graphlab::IN_EDGES;
	};

	/** The gather function computes XY */
	gather_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const
	{
		const vertex_type other_vertex = get_other_vertex(edge, vertex);
		const vertex_data& other_vdata = other_vertex.data();
				
		return gather_type(other_vdata.x, edge.data().coeff);	
	} // end of gather function

	/** apply collects the sum of XY */
	void apply(icontext_type& context, vertex_type& vertex,
		const gather_type& sum) {

		// Get and reset the vertex data
		vertex_data& vdata = vertex.data();


		if(vdata.role == vertex_data::X)
		{
			vdata.oldX = vdata.x;
			vdata.x = vdata.x - GAMMA*(sum.dot - vdata.aty); //TODO: L2, L1 norm
		}
		else
		{
			vdata.x = sum.dot;
		}
		
		vdata.nupdates++;
		
	} // end of apply
  
	/** The edges to scatter along */
	edge_dir_type scatter_edges(icontext_type& context,
				const vertex_type& vertex) const 
	{
		const vertex_data& vdata = vertex.data();
	
		if(vdata.role == vertex_data::X)
			if( ABS((vdata.oldX - vdata.x)/vdata.x) < DIFF_THRESHOLD 
				|| vdata.nupdates > MAX_UPDATES)
				return graphlab::NO_EDGES;
		return graphlab::OUT_EDGES;
		
	}; // end of scatter edges

	/** Scatter reschedules neighbors */
	void scatter(icontext_type& context, const vertex_type& vertex,
		edge_type& edge) const 
	{
		const vertex_type other_vertex = get_other_vertex(edge, vertex);
		
		context.signal(other_vertex);
	} // end of scatter function


	/**
	* \brief Signal all vertices on one side of the bipartite graph
	*/
	static graphlab::empty signal_P(icontext_type& context,
					const vertex_type& vertex) {
		if(vertex.data().role == vertex_data::P)
			context.signal(vertex);
		return graphlab::empty();
	} // end of signal_P

}; // end of gd vertex program



/**
 * \brief The graph loader function is a line parser used for
 * distributed graph construction.
 */
inline bool graph_edge_vertexPQ_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
	
	ASSERT_FALSE(line.empty());
	graph_type::vertex_id_type X_id(-1), P_id(-1), Q_id(-1);
	int  p_id_int, q_id_int, x_id_int;
	
	int group_id = -1;
	double value;
	std::stringstream sline(line);
	

	
	if(hasEnding(filename,".w"))
	{
		
		sline >> q_id_int >> p_id_int /*>> group_id*/ >> value;;
		
		P_id = -(graphlab::vertex_id_type(p_id_int + SAFE_NEG_OFFSET));	
		Q_id = factor_vertex_program::Q_OFFSET - q_id_int;
		graph.add_edge(P_id, Q_id, edge_data(value,group_id));

		
		if(q_id_int == p_id_int) 
		{
			vertex_data vdataP(vertex_data::P);
			vertex_data vdataQ(vertex_data::Q);
			
			//add vertex P
			graph.add_vertex(P_id, vdataP); 
			//add vertex Q
			graph.add_vertex(Q_id, vdataQ); 
			
		}
		else
		{
			P_id = -(graphlab::vertex_id_type(q_id_int + SAFE_NEG_OFFSET));	
			Q_id = factor_vertex_program::Q_OFFSET - p_id_int;
			graph.add_edge(P_id, Q_id, edge_data(value,group_id));	

		}
		

		
	}
	else if(hasEnding(filename, ".v"))
	{
		sline >> p_id_int >> x_id_int /*>> group_id*/ >> value;
		
		
		Q_id = factor_vertex_program::Q_OFFSET - p_id_int;
		
		P_id = -(graphlab::vertex_id_type(p_id_int + SAFE_NEG_OFFSET));

		X_id = x_id_int;

		graph.add_edge(X_id, P_id, edge_data(value,group_id));
		
		graph.add_edge(Q_id, X_id, edge_data(value,group_id));
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
	graph_type::vertex_id_type vertex_id(-1);
	
	if(hasEnding(filename,".aty"))
	{
		int n, index;
		double aty;
		double x;
		std::stringstream sline(line);
		
		sline >> index >> n >> aty;
		
		vertex_id = index;

		x = rand()%256;
			
		vertex_data vdataX(x, aty);
		graph.add_vertex(vertex_id, vdataX); 
	}
	return true;

} // end of graph_loader




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

	error_aggregator() : 
		delta(0), xnorm(0)
	{ }

	error_aggregator& operator+=(const error_aggregator& other) 
	{
		delta += other.delta;
		xnorm += other.xnorm;
		return *this;
	}

	static error_aggregator map(icontext_type& context, const graph_type::vertex_type& vertex) 
	{
		error_aggregator agg;
		if(vertex.data().role == vertex_data::X) 
		{
			double er = vertex.data().x - vertex.data().oldX;
			agg.delta = er * er; 
			
			agg.xnorm = vertex.data().oldX * vertex.data().oldX ;

			//context.cout() << (int)vertex.id() <<" agg.error = " << agg.error << std::endl;

		}
		return agg;
	}

	static void finalize(icontext_type& context, const error_aggregator& agg) 
	{
		const double delta = std::sqrt(agg.delta);
		const double xnorm = std::sqrt(agg.xnorm);
		context.cout() << "Time = " << context.elapsed_seconds() << "s\tNorm(x - oldX)/Norm(oldX) = " << delta / xnorm << std::endl;
		//if(error < 1e-10)
		//	context.stop();
	}
}; // end of error aggregator




/**
 * \brief The X saver is used by the graph.save routine to
 * output the final X vector back to the filesystem.
 */
struct linear_model_saver_X
{
	typedef graph_type::vertex_type vertex_type;
	typedef graph_type::edge_type   edge_type;
	// save the linear model, using the format:
	//   nodeid) factor1 factor2 ... factorNLATENT \n
	//
	std::string save_vertex(const vertex_type& vertex) const
	{
		std::stringstream ss;
		if (vertex.data().role == vertex_data::X)
		{
			ss << (int)vertex.id() << " " << vertex.data().x << std::endl;
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
int factor_vertex_program::Q_OFFSET = -1000;



int main(int argc, char** argv) 
{
	global_logger().set_log_level(LOG_INFO);
	global_logger().set_log_to_console(true);

	int interval = 5; // sec for aggregate error
	// Parse command line options -----------------------------------------------
	const std::string description =
	"Compute factorized Pagerank.";
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
	const bool success = engine.add_vertex_aggregator<error_aggregator>	("error", error_aggregator::map, error_aggregator::finalize) &&	engine.aggregate_periodic("error", interval);
	
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
		graph.save(result + ".x", linear_model_saver_X(),
			gzip_output, save_vertices, save_edges, threads_per_machine);

	}

	graphlab::mpi_tools::finalize();
	return EXIT_SUCCESS;
} // end of main




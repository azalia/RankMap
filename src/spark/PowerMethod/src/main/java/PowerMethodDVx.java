import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;

import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;







/** 
 * Computes Power method is not working :D
 */
public final class PowerMethodDVx {
  
  
  static public class MulByQ 
		implements Function<Tuple2<Double,List<Double>>,Tuple2<Double,List<Double>>>
	{
		public List<Double> q;
		
		public MulByQ(List<Double> _q)
		{
			q = _q;
		}
		
		@Override
		public Tuple2<Double, List<Double>> call(Tuple2<Double, List<Double>> ax)
		{
			double dot = 0;
			for(int i=0;i<q.size();i++)
			{
				dot += ax._2.get(i) * q.get(i);				
			}
			return new Tuple2<Double, List<Double>>(dot, ax._2);
		}
	}
  
  
  	static public class Divbynorm 
		implements Function<Tuple2<Double,List<Double>>,Tuple2<Double,List<Double>>>
	{
		public Double norm;
		
		public Divbynorm(Double _norm)
		{
			norm = _norm;
		}
		@Override
		public Tuple2<Double, List<Double>> call(Tuple2<Double,List<Double>> ax)
		{
			return new Tuple2<Double, List<Double>>(ax._1/norm, ax._2);
		}
	}


  
  
	
  
  

  public static void main(String[] args) throws Exception 
  {
	long t1, t2;
	
	System.out.println("Power mthod on DV");	    
    if(args.length != 7)
	{
		System.out.println("enter m n l Dfile Vdic xfile steps");
		return;
	}
	
	m = Integer.parseInt(args[0]);
	n = Integer.parseInt(args[1]);
	l = Integer.parseInt(args[2]);
    

        
    
    SparkConf sparkConf = new SparkConf().setAppName("PowerMethod");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    
    //read D file
	List<List<Double>> Dmat = jsc.textFile(args[3]).map(new Function<String, List<Double>>()
	{
	
		@Override
		public List<Double> call(String s)
		{
			List<Double> ret = new ArrayList<Double>();
			
			String sval[] = s.split(" ");
			for(int i=0;i<sval.length;i++)
			{
				ret.add(Double.parseDouble(sval[i]));
			}
			return ret;
		}
	}).collect();
    
    
    //read data from text file
    JavaRDD<String> data = jsc.textFile(args[4]);
       
    
    
    //transform text data to a coff of x and a col of A
	JavaRDD<Tuple2<Double,List<Double>>> Vx = data.map(new Function<String, Tuple2<Double, List<Double>>>()
		{
			@Override
			public Tuple2<Double,List<Double>> call(String s)
			{
				String[] parts = s.split(" ");
				Double x = Double.parseDouble(parts[0]);
				
				List<Double> a = new ArrayList<Double>(parts.length - 1);
				for(int i=0;i<parts.length - 1;i++)
				{
					a.add(i, Double.parseDouble(parts[i + 1]));
				}
				return new Tuple2<Double,List<Double>>(x , a);
			}
		});
    
    
    t1 = System.nanoTime();
    
    //steps of power method
    for(int step=0; step<steps; step++)
    {
		//calculate p = Vx
		List<Double> p = Vx.map(new Function<Tuple2<Double,List<Double>>, List<Double>>() {
			@Override
			public List<Double> call(Tuple2<Double,List<Double>> ax) {
				List<Double> l_y = new ArrayList<Double>(ax._2.size());
				for(int i=0;i<ax._2.size();i++)
				{
					l_y.add(i, ax._1 * ax._2.get(i));
				}
				return l_y;
			}
		}).reduce(new Function2<List<Double>, List<Double>, List<Double>>() {
			@Override
			public List<Double> call(List<Double> a, List<Double> b) {
				List<Double> ret = new ArrayList<Double>(a.size());
				
				for(int i=0;i<a.size();i++)
				{
					ret.add(i, a.get(i) + b.get(i));
				}
				return ret;
			}
		});
		
		// calculate y = Dp
		ArrayList<Double> y = new  ArrayList<Double>();
		for(int i=0;i<Dmat.size();i++)
		{
			double dot = 0;
			for(int j=0;j<p.size();j++)
			{
				dot += p.get(j) * Dmat.get(i).get(j);
			}
			y.add(dot);
		}
		
		
		// calculate q = Dty
		ArrayList<Double> q = new  ArrayList<Double>();
		for(int j=0;j<p.size();j++)
		{
			double dot = 0;
			for(int i=0;i<Dmat.size();i++)
			{
				dot += y.get(i) * Dmat.get(i).get(j);
			}
			q.add(dot);
		}
		
		
		//calculate x = Vtq
		Vx = Vx.map(new MulByQ(q));
		
		
		//reduce norm of x
		Double normX2 = Vx.map(new Function<Tuple2<Double,List<Double>>, Double>() {
			@Override
			public Double call(Tuple2<Double,List<Double>> ax) {
				return (ax._1*ax._1);
			}
		}).reduce(new Function2<Double, Double, Double>() {
			@Override
			public Double call(Double a, Double b) {
				return a + b;
			}
		});
		
		//normalized x
		Vx = Vx.map(new Divbynorm(Math.sqrt(normX2)));
		
		t2 = System.nanoTime();
		
		{
			System.out.format("%d %d %f\n", step, t2 - t1, normX2);
		}
		
	}
  }
}

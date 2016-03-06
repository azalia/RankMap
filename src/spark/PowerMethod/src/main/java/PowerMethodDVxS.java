import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;







/** 
 * Computes Power method of a matrix
 */
public final class PowerMethodDVxS {
  
  
  static public class MulByQ 
		implements Function<Tuple2<Double,List<Tuple2<Integer,Double>>>,Tuple2<Double,List<Tuple2<Integer,Double>>>>
	{
		public List<Double> q;
		
		public MulByQ(List<Double> _q)
		{
			q = _q;
		}
		
		@Override
		public Tuple2<Double, List<Tuple2<Integer,Double>>> call(Tuple2<Double, List<Tuple2<Integer,Double>>> ax)
		{
			double dot = 0;
			for(int i=0;i<ax._2.size();i++)
			{
				dot += ax._2.get(i)._2 * q.get(ax._2.get(i)._1);				
			}
			return new Tuple2<Double, List<Tuple2<Integer,Double>>>(dot, ax._2);
		}
	}
  
  
  	static public class Divbynorm 
		implements Function<Tuple2<Double,List<Tuple2<Integer,Double>>>,Tuple2<Double,List<Tuple2<Integer,Double>>>>
	{
		public Double norm;
		
		public Divbynorm(Double _norm)
		{
			norm = _norm;
		}
		@Override
		public Tuple2<Double, List<Tuple2<Integer,Double>>> call(Tuple2<Double,List<Tuple2<Integer,Double>>> ax)
		{
			return new Tuple2<Double, List<Tuple2<Integer,Double>>>(ax._1/norm, ax._2);
		}
	}

  	static public class Vmulx 
		implements
	Function<Tuple2<Double,List<Tuple2<Integer,Double>>>, List<Double>> 
	{
		
			public int l;
			
			public Vmulx(int _l)
			{
				l = _l;
			}
		
			@Override
			public List<Double> call(Tuple2<Double,List<Tuple2<Integer,Double>>> ax) 
			{
				List<Double> l_y = new ArrayList<Double>(l);
				
				for(int i=0;i<l;i++)
				{
					l_y.add(i, 0.0);
				}
				
				for(int i=0;i<ax._2.size();i++)
				{
					l_y.set(ax._2.get(i)._1, ax._1 * ax._2.get(i)._2);
				}
				return l_y;
			}
	}


  
  
	public static int n,m,l;
  
  

  public static void main(String[] args) throws Exception 
  {
  
	long t1, t2;
	
	System.out.println("Power mthod on DVxS");	  
    if(args.length != 7)
	{
		System.out.println("enter m n l Dfile Vdic xfile steps");
		return;
	}
	
	m = Integer.parseInt(args[0]);
	n = Integer.parseInt(args[1]);
	l = Integer.parseInt(args[2]);
	
	
    int steps = Integer.parseInt(args[6]);
    
    
        
    
    SparkConf sparkConf = new SparkConf().setAppName("PowerMethod");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    
    //read D file
	List<List<Double>> Dmat = jsc.textFile(args[4]).map(new Function<String, List<Double>>()
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
    
    //read V from text files
    JavaPairRDD<String,String> data = jsc.wholeTextFiles(args[4]);

    //transform text data to a ccol of v
	JavaPairRDD<Integer, List<Tuple2<Integer,Double>>> V = data.flatMapToPair(new PairFlatMapFunction<Tuple2<String,String>, Integer, List<Tuple2<Integer,Double>>>()
		{
			@Override
			public List<Tuple2<Integer, List<Tuple2<Integer,Double>>>> call(Tuple2<String,String> f)
			{
	
				List<Tuple2<Integer, List<Tuple2<Integer,Double>>>> ret = new ArrayList<Tuple2<Integer, List<Tuple2<Integer,Double>>>>();
				
				
				String sp[] = f._1.split("/");
				sp = sp[sp.length-1].split("_");
				int nf = Integer.parseInt(sp[2]);
				int fid = Integer.parseInt(sp[3]);
				String[] dim = sp[1].split("x");
				//m = Integer.parseInt(dim[0]);
				int n = Integer.parseInt(dim[1]);
				//l = Integer.parseInt(dim[2]);
				
				
				String[] lines = f._2.split("\n");
				
				int irow = 0, icol = 0;
				int picol = icol;
				double val = 0;
				
				List<Tuple2<Integer,Double>> col = new ArrayList<Tuple2<Integer,Double>>(); 
				
				for(int i=0;i<lines.length;i++)
				{
					String l[] = lines[i].split(" ");
					irow = Integer.parseInt(l[0]);
					icol = Integer.parseInt(l[1]);
					val = Double.parseDouble(l[2]);
					
					if(picol!= icol)//new col
					{
						ret.add(new Tuple2(fid*(n/nf) + picol, col));	
						col = new ArrayList<Tuple2<Integer,Double>>(); 				
					}				
					col.add(new Tuple2(irow, val));
					picol = icol;
				}
				ret.add(new Tuple2(fid*(n/nf) + picol, col));
				
				return ret;
			}
		});
    
    
    //read x file
     JavaPairRDD<String,String> xdata = jsc.wholeTextFiles(args[5]);

    //transform x text file to x
	JavaPairRDD<Integer, Double> x = xdata.flatMapToPair(new PairFlatMapFunction<Tuple2<String,String>, Integer, Double>()
		{
			@Override
			public List<Tuple2<Integer, Double>> call(Tuple2<String,String> f)
			{
				List<Tuple2<Integer, Double>> ret = new ArrayList<Tuple2<Integer, Double>>();
				
							
				String[] lines = f._2.split("\n");
				int rows = lines.length;
			
				for(int j=0;j<rows;j++)
				{
					ret.add(j,new Tuple2(j,Double.parseDouble(lines[j])));
				}
				
				return ret;
			}
			
		});
    
    
    JavaRDD<Tuple2<Double, List<Tuple2<Integer,Double>>>> Vx = x.join(V).values();
    

    
    t1 = System.nanoTime();

    
    //steps of power method
    for(int step=0; step<steps; step++)
    {
		//calculate p = Vx
		List<Double> p = Vx.map(new Vmulx(l)).reduce(new Function2<List<Double>, List<Double>, List<Double>>() {
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
		Double normX2 = Vx.map(new Function<Tuple2<Double,List<Tuple2<Integer,Double>>>, Double>() {
			@Override
			public Double call(Tuple2<Double,List<Tuple2<Integer,Double>>> ax) {
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

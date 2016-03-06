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


/** 
 * Computes Power method of a matrix
 */
public final class PowerMethodAx {
  
  
  static public class MulByY 
		implements Function<Tuple2<Double,List<Double>>,Tuple2<Double,List<Double>>>
	{
		public List<Double> y;
		
		public MulByY(List<Double> _y)
		{
			y = _y;
		}
		
		@Override
		public Tuple2<Double, List<Double>> call(Tuple2<Double, List<Double>> ax)
		{
			double dot = 0;
			for(int i=0;i<y.size();i++)
			{
				dot += ax._2.get(i) * y.get(i);				
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
	boolean verbos = false;
	long t1, t2;
		
    if(args.length != 3)
	{
		System.out.println("enter ADic xfile steps");
		return;
	}
    int steps = Integer.parseInt(args[2]);
    
    SparkConf sparkConf = new SparkConf().setAppName("PowerMethodAx");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    
    //read A from text file
    JavaPairRDD<String,String> data = jsc.wholeTextFiles(args[0]);

    //transform text data to col of A
	JavaPairRDD<Integer, List<Double>> A = data.flatMapToPair(new PairFlatMapFunction<Tuple2<String,String>, Integer, List<Double>>()
		{
			@Override
			public List<Tuple2<Integer, List<Double>>> call(Tuple2<String,String> f)
			{
	
				List<Tuple2<Integer, List<Double>>> ret = new ArrayList<Tuple2<Integer, List<Double>>>();
				
				
				String sp[] = f._1.split("/");
				sp = sp[sp.length-1].split("_");
				int nf = Integer.parseInt(sp[2]);
				int fid = Integer.parseInt(sp[3]);
				String[] dim = sp[1].split("x");
				int n = Integer.parseInt(dim[1]);
				
				
				String[] lines = f._2.split("\n");
				int rows = lines.length;
				int cols = lines[0].split(" ").length;
			
				String[][] data = new String[rows][];
				for(int j=0;j<rows;j++)
				{
					data[j] = lines[j].split(" "); 
				}
				
				for(int i=0;i<cols;i++)
				{
					List<Double> col = new ArrayList<Double>(rows);
					for(int j=0;j<rows;j++)
					{
						col.add(j,Double.parseDouble(data[j][i]));
					}
					ret.add(i,new Tuple2(fid*(n/nf) + i, col));
				}
				
				return ret;
			}
		});
    
    
    //read x file
     JavaPairRDD<String,String> xdata = jsc.wholeTextFiles(args[1]);

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
    
    
    JavaRDD<Tuple2<Double, List<Double>>> Ax = x.join(A).values();
    
        
    t1 = System.nanoTime();
        
    //steps of power method
    for(int step=0;step<steps;step++)
    {
		//calculate y = Ax
		List<Double> y = Ax.map(new Function<Tuple2<Double,List<Double>>, List<Double>>() {
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
				
		if (verbos)
		{
			StringBuilder sb = new StringBuilder();
			for (Double s : y)
			{
				sb.append(s);
				sb.append("\t");
			}
			System.out.println("y = " + sb.toString());
		}
		
		
		//calculate x = Aty
		Ax = Ax.map(new MulByY(y));
		
		
		if (verbos)
		{
			List<Tuple2<Double,List<Double>>> lax = Ax.collect();
			
			StringBuilder sb2 = new StringBuilder();
			for (Tuple2<Double,List<Double>> s : lax)
			{
				sb2.append(s._1 + "\t");
			}
			System.out.println("x = " + sb2.toString());
		}
		
		//reduce norm of x
		Double normX2 = Ax.map(new Function<Tuple2<Double,List<Double>>, Double>() {
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
		Ax = Ax.map(new Divbynorm(Math.sqrt(normX2)));
		
		t2 = System.nanoTime();
		
		{
			System.out.format("%d %d %f\n", step, t2 - t1, normX2);
		}
		
	}
  }
}

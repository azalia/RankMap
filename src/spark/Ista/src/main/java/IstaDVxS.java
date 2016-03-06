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
 * Computes Fista
 */
public final class IstaDVxS {
  
  
  static public class MulByQ 
		implements Function<Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>>,Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>>>
	{
		public List<Double> q;
		
		public MulByQ(List<Double> _q)
		{
			q = _q;
		}
		
		@Override
		public Tuple2<Tuple2<Double,Double>, List<Tuple2<Integer,Double>>> call(Tuple2<Tuple2<Double,Double>, List<Tuple2<Integer,Double>>> ax)
		{
			//update x = vtq
			double dot = 0;
			for(int i=0;i<ax._2.size();i++)
			{
				dot += ax._2.get(i)._2 * q.get(ax._2.get(i)._1);				
			}
			return new Tuple2<Tuple2<Double,Double>, List<Tuple2<Integer,Double>>>(new Tuple2(dot,ax._1._2), ax._2);
		}
	}
  
  
  	static public class sthreshold 
		implements Function<Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>>,Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>>>
	{
		public double gamma;
		public double lambda;
		
		public sthreshold(double gamma, double lambda)
		{
			this.gamma = gamma;
			this.lambda = lambda;
		}
		
		public static double softth(double v, double lambda)
		{
			lambda = (lambda>0)?lambda:-lambda;
			
			if(v>lambda)
				return v-lambda;
			else if(v<-lambda)
				return v+lambda;
			else
				return 0;
		}
		
		@Override
		public Tuple2<Tuple2<Double,Double>, List<Tuple2<Integer,Double>>> call(Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>> ax)
		{
			//update x soft thereshold
			// soft(xold - gamma*(VtDt(DVxold -y)),lambda)
			Double newX = softth(ax._1._2 - gamma*ax._1._1, gamma*lambda); 
			return new Tuple2<Tuple2<Double,Double>, List<Tuple2<Integer,Double>>>(new Tuple2(ax._1._2, newX), ax._2);
		}
	}
	

  	static public class Vmulx 
		implements
	Function<Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>>, List<Double>> 
	{
		
			public int l;
			
			public Vmulx(int _l)
			{
				l = _l;
			}
		
			@Override
			public List<Double> call(Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>> ax) 
			{
				List<Double> l_y = new ArrayList<Double>(l);
				
				for(int i=0;i<l;i++)
				{
					l_y.add(i, 0.0);
				}
				
				for(int i=0;i<ax._2.size();i++)
				{
					l_y.set(ax._2.get(i)._1, ax._1._2 * ax._2.get(i)._2);
				}
				return l_y;
			}
	}


  
  

  public static void main(String[] args) throws Exception 
  {
  
	long t1, t2;
	
	 
    if(args.length != 11)
	{
		System.out.println("enter m n l gammaN lambda Dfile Vdic xfile yfile steps verbos");
		return;
	}
	
	int m = Integer.parseInt(args[0]);
	int n = Integer.parseInt(args[1]);
	int l = Integer.parseInt(args[2]);
	double gamma = (2.0*Double.parseDouble(args[3])/n);  
	double lambda = Double.parseDouble(args[4]); 
	int steps = Integer.parseInt(args[9]);
    int verbos = Integer.parseInt(args[10]);
    
    
    if(verbos==1)
	{
		System.out.println("ISTA on DVxS");	 
    }
    
    SparkConf sparkConf = new SparkConf().setAppName("IstaDVxS");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    //read y
    List<Double> y =  jsc.textFile(args[8]).map(new Function<String, Double>()
	{
		@Override
		public Double call(String s)
		{
			return Double.parseDouble(s);
		}
	})
    .collect();

	if(verbos==1)
    {
		System.out.println("y = ");
		for(Double t : y)
		{
			System.out.print(" " + t);
		}
		System.out.println();
	}
	
	
	//read D
	List<List<Double>> Dmat = jsc.textFile(args[5]).map(new Function<String, List<Double>>()
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
    JavaPairRDD<String,String> data = jsc.wholeTextFiles(args[6]);

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
     JavaPairRDD<String,String> xdata = jsc.wholeTextFiles(args[7]);

    //transform x text file to x
	JavaPairRDD<Integer, Tuple2<Double,Double>> x = xdata.flatMapToPair(new PairFlatMapFunction<Tuple2<String,String>, Integer, Tuple2<Double,Double>>()
		{
			@Override
			public List<Tuple2<Integer, Tuple2<Double,Double>>> call(Tuple2<String,String> f)
			{
				List<Tuple2<Integer, Tuple2<Double,Double>>> ret = new ArrayList<Tuple2<Integer, Tuple2<Double,Double>>>();
				
							
				String[] lines = f._2.split("\n");
				int rows = lines.length;
			
				for(int j=0;j<rows;j++)
				{
					double val = Double.parseDouble(lines[j]);
					ret.add(j,new Tuple2(j, new Tuple2(val,val)));//VtDt(DVx-y), x
				}
				
				return ret;
			}
			
		});
    
    if(verbos==1)
    {
		System.out.println("x = ");
		List<Tuple2<Double,Double>> xcollect = x.values().collect();
		for(Tuple2<Double,Double> t : xcollect)
		{
			System.out.print(" " + t._1);
		}
		System.out.println();
	}
    
    
    
    JavaRDD<Tuple2<Tuple2<Double,Double>, List<Tuple2<Integer,Double>>>> Vx = x.join(V).values();
    

    if(verbos==1)
    {
		System.out.println("start ista with gamma = " + gamma + " lambda = " + lambda);	
		System.out.println("iter time diffY2 diffX2");	
	}
	
	
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
		
		// calculate yh = Dp
		ArrayList<Double> yh = new  ArrayList<Double>();
		for(int i=0;i<Dmat.size();i++)
		{
			double dot = 0;
			for(int j=0;j<p.size();j++)
			{
				dot += p.get(j) * Dmat.get(i).get(j);
			}
			yh.add(dot);
		}
		
		
		if(verbos==1)
		{	
			System.out.println("yres = ");
		}
		
		//yh(yres) = yh - y =DVx - y
		double diffY2 = 0;
		for(int i=0;i<Dmat.size();i++)
		{
			diffY2 += (yh.get(i) - y.get(i))*(yh.get(i) - y.get(i));
			yh.set(i, yh.get(i) - y.get(i));	
			if(verbos==1)
			{	
				System.out.print(" " + yh.get(i));
			}
		}
		if(verbos==1)
		{	
			System.out.println();
		}
		
		
		// calculate q = Dty
		ArrayList<Double> q = new  ArrayList<Double>();
		for(int j=0;j<p.size();j++)
		{
			double dot = 0;
			for(int i=0;i<Dmat.size();i++)
			{
				dot += yh.get(i) * Dmat.get(i).get(j);
			}
			q.add(dot);
		}
		
		
		//calculate x = Vtq
		Vx = Vx.map(new MulByQ(q));
		
		//soft thereshold
		Vx = Vx.map(new sthreshold(gamma, lambda));
		
		
		
		//find (x- xold)^2
		Double diffX2 = Vx.map(new Function<Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>>, Double>() {
			@Override
			public Double call(Tuple2<Tuple2<Double,Double>,List<Tuple2<Integer,Double>>> ax) {
				return (ax._1._1 - ax._1._2)*(ax._1._1 - ax._1._2);
			}
		}).reduce(new Function2<Double, Double, Double>() {
			@Override
			public Double call(Double a, Double b) {
				return a + b;
			}
		});
	
		if(diffX2 < 1E-5)
		{
			System.out.println("converge at " + step);
			break;
		}
		else if(diffX2 > 1E10)
		{
			System.out.println("not converge at " + step);
			break;
		}

		t2 = System.nanoTime();
		
		System.out.format("%d %f %f %f\n", step, (t2 - t1)/1000000.0, diffY2, diffX2);
	}
  }
}

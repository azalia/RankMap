#!/bin/sh
#enter m n l Dfile Vdic xfile steps
../../bin/spark-submit --driver-memory 1g --master spark://`hostname `:7077 --class "IstaDVxS" target/ista-1.0.jar 4 6 3 0.03 0.1 ../data/d ../data/v_4x6x3_1/ ../data/x ../data/y 20 0 2>/dev/null 

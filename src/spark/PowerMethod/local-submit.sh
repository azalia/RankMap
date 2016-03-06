#!/bin/sh
#enter m n l Dfile Vdic xfile steps
$SPARK/bin/spark-submit --driver-memory 1g --master spark://`hostname `:7077 --class "PowerMethodDVxS" target/power-method-1.0.jar 4 6 3 ../data/d ../data/v_4x6x3_1/ ../data/x 20 2>/dev/null 

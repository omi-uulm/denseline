This is the Python version of Dominik Moritz's code for the [Density Line Chart](https://github.com/domoritz/line-density) implementation.
The original publication can be [found here](https://arxiv.org/abs/1808.06019).


How to use:
Prepare your dataframe, all time-series should be stored in one dataframe. 
Each column represents one time series like so:

| index 	| TimeSeries0 	| TimeSeries1 	| TimeSeries2 	| TimeSeries3 	|
|-------	|-------------	|-------------	|-------------	|-------------	|
| 0     	| 0.1         	| 2.3         	| 1.2         	| 1.2         	|
| 1     	| 0.3         	| 1.0         	| 2.3         	| 2.3         	|
| 2     	| 0.1         	| 9.3         	| 1.2         	| 1.5        	  |


The code does not work with a datetime-index so be sure the index is numeric.

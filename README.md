# Inference of Gene Regulatory Networks Based on Nonlinear Ordinary Differential Equations
Baoshan Ma1,*, Mingkun Fang1 and Xiangtian Jiao1

1 College of Information Science and Technology, Dalian Maritime University, Dalian 116039, China



**The proposed method is a scalable method exploiting time-series and steady-state data jointly, in which nonlinear ODEs and XGBoost are employed to infer gene regulatory networks.** 

### The describe of the program 

```
The program of xgbgrn can combine time-series data and  steady-state data to infer GRNs, the steady-state data is not necessary.

The program of xgbgrn_2 can only be applied to  a type of  data.
```



### The version of Python and packages
    Python version=3.6
    Xgboost version=0.82
    scikit-learn version=l0.24.2
    numpy version=1.16.3


### Parameters
    xgbgrn:
    	TS_data: a matrix of time-series data
    	time_points: a list of time points
    	alpha:a constant or specify "from_data"
    	SS_data: a matrix of time-series data, the default is "none"
    	gene_names: a list of gene names
    	regulators: a list of names of regulatory genes, the default is "all", 
    	param: a dict of parameters of xgboost
    	
    xgbgrn_2:
    expr_data: a matrix of gene expression data
    gene_names: a list of gene names
    regulators: a list of names of regulatory genes
    param: a dict of parameters of xgboost



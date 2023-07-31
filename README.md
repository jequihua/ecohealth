# ecohealth
Prediction of and inference on ecosystem health with probabilistic graphical models.

The general workflow is all contained in the EcoNet() class and is as follows (a complete example is provided):

### 1.- Create an appropriate data table (preprocess_rasters.py)

All the example data can be obtained <a href="https://www.dropbox.com/sh/6r0cohj5rggi47i/AADOz6yoCPE5dZcxspGbPYdNa?dl=0">here</a>.

Reads in all the rasters located in the selected path. Then discretizes them using the <a href="[https://www.dropbox.com/sh/6r0cohj5rggi47i/AADOz6yoCPE5dZcxspGbPYdNa?dl=0](https://feature-engine.trainindata.com/en/latest/index.html)">feature engine</a> library. This is can be perfomred in any other way and is only provided as an example.

### 2.- Specify a bayesian network model using the previously created data table (ecohealth.py)

Uses a subset of the column names of the data table in order to initialize the nodes (variables) of a Bayesian Network (BN) model. 

A BN also requires a set of arcs in order to be fully specified, this can be done using an adjacency matrix, which is provided in the example data.

### 3.- Fit BN (ecohealth.py)

The BN must now be fitted on the data from which there is evidence of some level of ecosystem health. Currenly can only be performed using the Expectation Maximization algorithm.

### 4.- Predict ecosystem health over the region of interest (ecohealth.py)

Once the model fit is complete, the model can use the full data frame to produce an ecosystem health map of the region.

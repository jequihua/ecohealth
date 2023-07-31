import numpy as np

def computeCPTfromDF(bn,df,name):
    """
    Compute the CPT of variable "name" in the Bayesian Network bn from the Data Frame df
    """
    id=bn.idFromName(name)
    parents=list(reversed(bn.cpt(id).names))
    domains=[bn.variableFromName(name).domainSize()
             for name in parents]
    parents.pop()

    if (len(parents)>0):
        c=pd.crosstab(df[name],[df[parent] for parent in parents])
        s=c/c.sum().apply(np.float32)
    else:
        s=df[name].value_counts(normalize=True)

    bn.cpt(id)[:]=np.array((s).transpose()).reshape(*domains)

def ParametersLearning(bn,df):
    """
    Compute the CPTs of every varaible in the Bayesian Network bn from the Data Frame df
    """
    for name in bn.names():
        computeCPTfromDF(bn,df,name)
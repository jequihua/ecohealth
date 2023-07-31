import misc_functions as mf
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pyAgrum as gum
from pyAgrum import BayesNet
import pyAgrum.skbn as skbn
import numpy as np
from sklearn import preprocessing as pre
import rasterio as rio

class EcoNet(BayesNet):
    def add_nodes_df(self, df: pd.DataFrame, variables: list, missingvalue=-3.4e+38) -> pd.DataFrame:
        """
        Adds nodes to an empty bayesian network (BN) based on a list of columns of a pandas Data Frame.
        It assumes all columns are discrete and labeled. Numeric columns will be converted to str.

        Returns the transformed Data Frame for BN training.
        """
        for var in variables:
            if is_numeric_dtype(df[var]):
                var_labels = np.sort(df[var].unique())
                var_labels = var_labels[var_labels != missingvalue]
                df[var] = df[var].astype('str')
            else:
                var_labels = df[var].unique()

            var_labelized = gum.LabelizedVariable(var, var, len(var_labels))
            for label in var_labels:
                var_labelized.addLabel(str(label))
            self.add(var_labelized)
        return df

    def add_arcs_df(self, adj_matrix: pd.DataFrame, adj_variables: list) -> None:
        """
        Adds arcs to a bayesian network (BN) that already has a set of initialized nodes,
        based on a list of columns of a pandas Data Frame that represents an adjacency matrix.

        1 indicates an arc, always: row -> column
        0 or NA no arc

        Returns None.
        """
        for row, row_variable in enumerate(adj_variables):
            for col, col_variable in enumerate(adj_variables):
                if adj_matrix.iat[row, col + 1] == 1.0:
                    self.addArc(row_variable, col_variable)

    def fit_em(self, training_table: pd.DataFrame, epsilon=0.00007) -> BayesNet:
        """
        Fits a fully specified BN on a Data Frame using expectation maximization.
        """
        learner = gum.BNLearner(training_table, self)
        fittedbn = learner.useEM(epsilon=epsilon).learnBN()
        return fittedbn

    def fitted_classifier(self, bn, targetAttribute: str):
        """
        Converts a fitted BN to a scikit-learn compliant classifier.
        """
        skbn_classifier = skbn.BNClassifier()
        skbn_classifier.fromTrainedModel(bn=bn, targetAttribute=targetAttribute)
        return skbn_classifier

    def eco_index(self, skbn_classifier, test_table: pd.DataFrame, variables: list, nclasses=3) -> None:
        """
        Predicts the ecosystem integrity index based on an skbn model and a Data Frame.

        Returns a numpy array with the index values.
        """
        predicted_probas = skbn_classifier.predict_proba(test_table[variables])
        predicted_probas = predicted_probas[:, -1*nclasses:]
        eh_levels = np.array(range(nclasses))+1.0
        expected_value = np.matmul(predicted_probas, eh_levels)
        ecosystem_integrity = pre.MinMaxScaler().fit_transform(expected_value.reshape(-1, 1))
        return ecosystem_integrity

def main():

    PATHDATA = '../data/'

    full_training_table = pd.read_csv(PATHDATA + "data_table_v1.csv")
    full_training_table = full_training_table.loc[~full_training_table['zvh_31_lcc_h'].isna()]
    variables = list(full_training_table.columns)
    variables = mf.filter_list(variables, ['Unnamed: 0'])

    # Initialize bayesian network.
    bnmodel = EcoNet()

    # Add nodes based on the previous table.
    full_training_table = bnmodel.add_nodes_df(full_training_table, variables)

    # Initialize arcs based on adjacency matrix from a csv file.
    adj_matrix = pd.read_csv(PATHDATA + "ienet_v2.csv")
    adj_matrix.fillna(0, inplace=True)
    adj_variables = list(adj_matrix.columns)
    adj_variables = mf.filter_list(adj_variables, ['Unnamed: 0'])
    bnmodel.add_arcs_df(adj_matrix, adj_variables)

    # Fit bayesian network on only data with evidence of condition.
    clean_training_table = full_training_table[full_training_table['hemerobia_250m'].isin(['1.0', '2.0', '3.0'])]
    fittedmodel = bnmodel.fit_em(clean_training_table)

    # Convert fitted BN to a sklearn compliant classifier.
    skbn = bnmodel.fitted_classifier(bn=fittedmodel, targetAttribute='hemerobia_250m')

    # Calculate Ecosystem Integrity Index based on the previous.
    variables_ex = mf.filter_list(variables, ['hemerobia_250m'])
    ecosystem_integrity = bnmodel.eco_index(skbn, full_training_table, variables_ex)

    # Save results to a geotiff raster.
    files = mf.multiple_file_types(PATHDATA, ["*.tif"], recursive=True)
    files_ = list(files)
    rast = rio.open(files_[0])
    transform = rast.transform
    crs = rast.crs
    width = rast.width
    height = rast.height
    new_dataset = rio.open(PATHDATA + "ecohealth_index.tif", 'w', driver='GTiff',
                                height=height, width=width,
                                count=1, dtype=np.float64,
                                crs=crs,
                                transform=transform)

    new_dataset.write(ecosystem_integrity.reshape((height, width)), 1)
    new_dataset.close()

if __name__ == "__main__":
    main()

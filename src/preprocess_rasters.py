import misc_functions as mf
from feature_engine.discretisation import EqualWidthDiscretiser

def main():

    PATHDATA = '../data/'

    files = mf.multiple_file_types(PATHDATA, ["*.tif"], recursive=True)
    files_ = list(files)
    data_table = mf.raster_to_table(files_, cornernan=False)

    # Create missling values mask.
    variables = list(data_table.columns)
    categorical_variables = ['zvh_31_lcc_h', 'hemerobia_250m']
    variables = mf.filter_list(variables, categorical_variables)
    missings = mf.row_missings(data_table, variables)
    missings.to_csv(PATHDATA + "data_table_v1_mask.csv")

    # Drop missing values and discretise variables.
    data_table = data_table.loc[missings != 1]
    discretiser = EqualWidthDiscretiser(bins=5, variables=variables, return_boundaries=True)
    discretiser.fit(data_table)
    data_table = discretiser.transform(data_table)

    data_table.to_csv(PATHDATA + "data_table_v1.csv")

if __name__ == "__main__":
    main()

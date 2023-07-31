import os
import itertools
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling

def multiple_file_types(input_directory, patterns, recursive=False):
    """
    Return iterable with files that have a common pattern. Will search
    in a recursive or non recursive way.
    Args:
        input_directory (str): directory where files with common pattern
        will be searched.
        patterns (list): list of patterns to search for.
    Returns:
        iterable with files that have a common pattern.
    """
    if recursive:
        expression = "/**/*"
    else:
        expression = "/*"
    return itertools.chain.from_iterable(glob.iglob(input_directory +
                                                    expression + pattern,
                                                    recursive=recursive) for pattern in patterns)

def listdirs(path):
    """
    Create a list of folders in a given path.
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def filename(path):
    """
    Extract the filename from a path without extension.
    """
    return Path(path).stem

def filter_list(full_list, excludes, tolist=True):
    """
    Remove elements of one list from another list..
    """
    s = set(excludes)
    filteredlist = (x for x in full_list if x not in s)
    if tolist:
        filteredlist = list(filteredlist)
    return filteredlist

def raster_to_table(raster_list, dataframe = True, cornernan= True):
    """
    Receives a list of rasters and returns a data table where each column
    corresponds to a raster. Either a numpy array or a pandas dataframe.
    """
    numpys_list = []
    variables_list = []
    for raster in raster_list:
        variables_list.append(filename(raster))
        with rio.open(raster, "r+") as src:
            numpyraster = src.read()[0, :, :]
            if cornernan:
                nanvalue = numpyraster[0,0]
                numpyraster[numpyraster == nanvalue] = np.nan
            else:
                src.nodata = np.nan
            numpys_list.append(numpyraster.flatten())
    if dataframe:
        data_table = pd.DataFrame(dict(zip(variables_list, numpys_list)), columns=variables_list)
    else:
        data_table = np.hstack(numpys_list)
    return data_table

def row_missings(dataframe, variables):
    """
    Returns a mask of missing values found in a selection of columns of a dataframe.
    1 if there is any np.nan located in the row of the selection.
    0 in any other case.
    """
    missings = dataframe[variables].isna().sum(axis=1).apply(lambda x: 0 if x == 0 else 1)
    return missings


def swap_values(flattened_np, listOfInLists, listOfSwappingValues):
    """
    Takes each list in listOfInLists and swaps it by the
    corresponding value in listOfSwappingValues.
    For example if 
    listOfInLists=[[1,2],[3,4]] and  
    listOfSwappingValues=[10,11] then
    1 and 2 will become 10
    3 and 4 will become 11
    """
    aux=flattened_np
    if len(listOfInLists) != len(listOfSwappingValues):
        print("Lists must be of the same length.")
    else:
        for i in range(len(listOfInLists)):
            # list to numpy array
            nparray = np.array(listOfInLists[i])
            found_idx = np.in1d(flattened_np,nparray)
            aux[found_idx]=listOfSwappingValues[i]
    return aux

def lc_overlay(data,
               raster,
               reference_raster,
               lcids=[1, 2, 3, 4, 5, 6],
               lclabels=['forest', 'nonforest', 'farming', 'nonvegatated', 'water', 'none'],
               nanmask=None):
    """
    Overlays a coarse reference grid on a fine categorical raster (e.g. land cover) and calculates the proportion
    of each class inside each coarse cell. Returns a data frame.
    """
    raster_aux = raster.copy()
    data_aux = data.copy()

    data_aux[data_aux != lcids[0]] = 0
    data_aux[data_aux == lcids[0]] = 1
    data_aux[nanmask] = np.nan
    raster_aux.values = data_aux
    raster_aux = raster_aux.rio.reproject_match(reference_raster, resampling=Resampling.average)
    raster_aux = raster_aux.assign_coords({"x": reference_raster.x, "y": reference_raster.y, })
    xds_match_df = raster_aux.to_dataframe()
    xds_match_df.rename(columns={'band_data': lclabels[0]}, inplace=True)

    if len(lcids) > 1:
        for i in range(len(lcids)-1):
            raster_aux = raster.copy()
            data_aux = data.copy()

            data_aux[data_aux != lcids[i+1]] = 0
            data_aux[data_aux == lcids[i+1]] = 1
            data_aux[nanmask] = np.nan
            raster_aux.values = data_aux
            raster_aux = raster_aux.rio.reproject_match(reference_raster, resampling=Resampling.average)
            raster_aux = raster_aux.assign_coords({"x": reference_raster.x, "y": reference_raster.y, })
            xds_match_df_aux = raster_aux.to_dataframe()
            xds_match_df[lclabels[i+1]] = xds_match_df_aux['band_data']

    return xds_match_df
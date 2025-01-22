#!/usr/bin/env python
# coding: utf-8

"""
Overview:
========
Datamerge's reading and writing methods for filling the dataclasses information from external files, or to write 
the end result to external files. 
"""

__author__ = "Brian R. Pauw"
__contact__ = "brian@stack.nl"
__license__ = "GPLv3+"
__date__ = "2022/10/18"
__status__ = "beta"


import numpy as np
from .dataclasses import (
    HDFDefaultsObj,
    HDFPathsObj,
    outputRangeObj,
    readConfigObj,
    supplementaryDataObj,
)
from .dataclasses import rangeConfigObj
from .dataclasses import (
    scatteringDataObj,
    mergedDataObj,
    mergeConfigObj,
)
import h5py
from pathlib import Path
import yaml
import nexusformat.nexus as nx
from typing import List
import logging


def scatteringDataObjFromNX(
    filename: Path, readConfig: readConfigObj
) -> scatteringDataObj:
    """Returns a populated scatteringDataObj by reading the data from a Processed MOUSE NeXus file"""
    assert filename.is_file(), logging.warning(
        f'{filename=} cannot be accessed from {Path(".").absolute().as_posix()}'
    )
    with h5py.File(filename, "r") as h5f:
        kvs = {}
        for key in readConfig.hdfPaths.keys():
            hPath = getattr(readConfig.hdfPaths, key)

            if hPath in h5f:
                val = h5f[hPath][()]
                if isinstance(val, np.ndarray):
                    val = val.flatten()
            else:
                assert key in readConfig.hdfDefaults.keys(), logging.error(
                    f"NeXus file {filename=} does not contain information for {key=} at specified HDF5 Path {hPath}, filling in the default value."
                )
                val = getattr(readConfig.hdfDefaults, key)
            # convert bytestrings to normal strings assuming utf-8:
            if isinstance(val, bytes):
                val = val.decode("utf-8")

            if key in readConfig.hdfDefaults:
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val[0]  # numpy limitation: no longer casts arrays to scalars.
                else:
                    pass

            kvs.update({key: val})
    return scatteringDataObj(filename=filename, **kvs)


def outputToNX(
    ofname: Path,
    mco: mergeConfigObj,
    mdo: mergedDataObj,
    rangeList: List[rangeConfigObj],
    supplementaryData: supplementaryDataObj,
    writeOriginalData: bool = True,
    CheckAutomaticName: bool = True,
) -> None:
    """
    Stores the configuration, data and range list in the output file.
    """
    if ofname.stem == "automatic" and CheckAutomaticName:
        # "automatically determine an output name if the stem is called automatic"
        FileString = "merged_"
        if supplementaryData.sampleOwner is not None:
            sstr = supplementaryData.sampleOwner
            if isinstance(sstr, bytes):
                sstr = sstr.decode("utf-8")
            FileString += "".join(x for x in sstr if (x.isalnum() or x in "._- "))
        FileString += "_"
        if supplementaryData.sampleName is not None:
            sstr = supplementaryData.sampleName
            if isinstance(sstr, bytes):
                sstr = sstr.decode("utf-8")
            FileString += "".join(x for x in sstr if (x.isalnum() or x in "._- "))
        FileString += "_"
        followInt = 0
        unique = False
        while not unique:
            newofname = Path(ofname.parent, f"{FileString}{followInt}{ofname.suffix}")
            if newofname.exists():
                followInt += 1
            else:
                unique = True
        ofname = newofname

    # remove if output file already exists:
    if ofname.is_file():
        ofname.unlink()
    # this is the way (to make nexusformat write the base structure)
    nxf = nx.NXroot()
    nxf = nxf.save(ofname)
    # now we can start:
    nxf["/datamerge/"] = nx.NXentry()
    nxf["/datamerge/mergeConfig"] = nx.NXgroup()
    for key, val in mco.items():
        if key in ["df", "outputRanges", "ranges", "maxRange"]:
            continue  # skip
        if key == "filename":
            assert isinstance(val, Path)
            val = val.as_posix()
        nxf[f"/datamerge/mergeConfig/{key}"] = nx.NXfield(val)

    # store information on the input datasets and settings:
    # logging.debug(rangeList)
    nxf["/datamerge/ranges"] = nx.NXgroup()
    for drange in rangeList:
        # link original dataset in the structure
        # nxf[f"/entry{dfn}"] = nx.NXlink("/processed", file=drange.filepath)
        nxf[f"/datamerge/ranges/range{drange.rangeId}"] = nx.NXgroup()
        for key, val in drange.items():
            logging.debug(f"{key=}, {val=}")
            if key in ["scatteringData"] and writeOriginalData:
                # nxf[
                #     f"/datamerge/ranges/range{drange.rangeId}/scatteringData"
                # ] = nx.NXgroup()
                nxf[
                    f"/datamerge/ranges/range{drange.rangeId}/scatteringData"
                ] = nx.NXdata(
                    nx.NXfield(val.I, name="I"),
                    axes=(nx.NXfield(val.Q, name="Q")),
                    errors=nx.NXfield(val.ISigma, name="ISigma"),
                )  # ISigma is the combined uncertainty estimate
                for dkey, dval in val.items():
                    if dkey in [
                        "dataLen",
                        "asPandas",
                        "qMin",
                        "qMax",
                        "qRange",
                        "returnMaskByQRange",
                        "I",
                        "Q",
                        "ISigma",
                    ]:
                        continue  # skip
                    if dkey == "filename":
                        dval = dval.as_posix()
                    nxf[
                        f"/datamerge/ranges/range{drange.rangeId}/scatteringData/{dkey}"
                    ] = nx.NXfield(dval)
                continue

            nxf[f"/datamerge/ranges/range{drange.rangeId}/{key}"] = nx.NXfield(val)

    # store the resulting binned dataset itself
    nxf[f"/datamerge/result"] = nx.NXdata(
        nx.NXfield(mdo.I, name="I"),
        axes=(nx.NXfield(mdo.Q, name="Q")),
        errors=nx.NXfield(mdo.ISigma, name="ISigma"),
    )  # ISigma is the combined uncertainty estimate

    nxf[f"/datamerge/result"].attrs["sampleName"] = supplementaryData.sampleName
    nxf[f"/datamerge/result"].attrs["sampleOwner"] = supplementaryData.sampleOwner
    nxf[f"/datamerge/result"].attrs["configurations"] = supplementaryData.configurations

    # store the remainder of the merged data object:
    for key, val in mdo.items():
        if key in ["I", "Q", "ISigma"]:
            continue  # skip
        nxf[f"/datamerge/result/{key}"] = nx.NXfield(val)

    # nxf = nxf.save(ofname)

    # store the sample name and sample owner
    # nxf[f"/sample_name"] = self.sampleName
    # nxf[f"/sample_owner"] = self.sampleOwner
    # link the Q uncertainties:
    nxf[f"/datamerge/result/Q"].attrs["uncertainties"] = "QSigma"
    nxf[f"/datamerge/result/I"].attrs["uncertainties"] = "I_errors"
    # also set as resolution for now
    nxf[f"/datamerge/result/Q"].attrs["resolutions"] = "QSigma"
    # CHECK: this should be automatic no?
    # # set the default path to follow
    nxf.attrs["default"] = "entry"
    nxf[f"/datamerge"].attrs["default"] = "result"
    # nxf[f"/datamerge/result"].attrs["default"] = "I"

    # link main SASentry to datamerge dataset
    nxf["/entry"] = nx.NXlink("/datamerge")
    # canSAS compatibiity
    nxf["/datamerge"].attrs["canSAS_class"] = "SASentry"
    nxf["/datamerge"].attrs["version"] = "1.0"
    nxf["/datamerge/definition"] = nx.NXfield("NXcanSAS")
    nxf["/datamerge/run"] = nx.NXfield(0)
    nxf["/datamerge/title"] = nx.NXfield(
        f"merged dataset from {len(rangeList)} datasets"
    )
    nxf["/datamerge/result"].attrs["canSAS_class"] = "SASdata"
    nxf["/datamerge/result"].attrs["I_axes"] = "Q"
    nxf["/datamerge/result"].attrs["Q_indices"] = 0
    nxf["/datamerge/result"].attrs["default"] = "I"
    nxf["/datamerge/result/Q"].attrs["units"] = mco.outputQUnits
    nxf["/datamerge/result/I"].attrs["units"] = mco.outputIUnits
    # why doesn't this path exist?
    # nxf["/datamerge/result/ISigma"].attrs["units"] = mco.outputIUnits
    # nxf['/datamerge/result/IE'].attrs['units'] = "1/m"
    nxf.close()
    return ofname


def readConfigObjFromYaml(filename: Path) -> readConfigObj:
    assert (
        filename.is_file()
    ), f"Read configuration filename {filename.as_posix()} does not exist"
    with open(filename, "r") as f:
        configDict = yaml.safe_load(f)
    if "readConfig" not in configDict.keys():
        return readConfigObj(HDFPathsObj(), HDFDefaultsObj())
    logging.debug(configDict["readConfig"])
    HDFPaths = HDFPathsObj(**configDict["readConfig"].get("HDFPaths", {}))
    HDFDefaults = HDFDefaultsObj(**configDict["readConfig"].get("HDFDefaults", {}))
    return readConfigObj(HDFPaths, HDFDefaults)


def mergeConfigObjFromYaml(filename: Path) -> mergeConfigObj:
    assert (
        filename.is_file()
    ), f"Merge configuration filename {filename.as_posix()} does not exist"
    with open(filename, "r") as f:
        configDict = yaml.safe_load(f)

    mcoAcceptableParameters = [  # acceptable parameters are everything in mco except underscore-starting objects and range
        i
        for i in dir(mergeConfigObj)
        if (not i.startswith("_") and i not in ["ranges", "filename", "outputRanges"])
    ]
    # now we set some of these parameters if we have them in configDict:
    mcoFeedParams = {
        k: v
        for k, v in configDict.items()
        if (v is not None and k in mcoAcceptableParameters)
    }
    # and we need to construct the ranges list:
    rList = list()
    rngAcceptableParameters = [
        i
        for i in dir(rangeConfigObj)
        if (not i.startswith("_") and not i in ["scatteringData", "scale"])
    ]
    if configDict.get("ranges") is not None:
        for rawRange in configDict.get("ranges"):
            rngFeedParams = {
                k: v
                for k, v in rawRange.items()
                if (v is not None and k in rngAcceptableParameters)
            }
            rList += [rangeConfigObj(**rngFeedParams)]
    # same for the outputRange list
    oRList = list()
    oRngAcceptableParameters = [
        i for i in dir(outputRangeObj) if (not i.startswith("_"))
    ]
    if configDict.get("outputRanges") is not None:
        for rawORange in configDict.get("outputRanges"):
            oRngFeedParams = {
                k: v
                for k, v in rawORange.items()
                if (v is not None and k in oRngAcceptableParameters)
            }
            oRList += [outputRangeObj(**oRngFeedParams)]

    mco = mergeConfigObj(
        filename=filename, **mcoFeedParams, ranges=rList, outputRanges=oRList
    )
    return mco


def SDOListFromFiles(
    fnames: List[Path], readConfig: readConfigObj
) -> List[scatteringDataObj]:
    """
    Takes a list of file paths and a read configuration
    and returns the list of scatteringDataObjects read from the individual files
    """

    scatteringDataList = []
    for fname in fnames:
        assert (
            fname.is_file()
        ), f"filename {fname} does not exist. Please supply valid filenames"
        # print(fname)
        scatteringDataList += [scatteringDataObjFromNX(fname, readConfig)]
    return scatteringDataList


if __name__ == "__main__":
    """quick test"""
    so = scatteringDataObjFromNX(
        Path(".")
        / "tests"
        / "data"
        / "20230830"
        / "autoproc"
        / "Thuenemann_ERM-FD-121"
        / "20230830_109_expanded_stacked_processed.nxs"
    )

    mco = mergeConfigObjFromYaml(Path(".", "tests", "mergeConfigExample.yaml"))
    logging.debug(mco)

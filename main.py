#!/usr/bin/env python
# coding: utf-8
# requires at least attrs version == 21.4
from pathlib import Path
import argparse
from sys import platform
import logging
from typing import Optional

from datamerge.readersandwriters import scatteringDataObjFromNX
from datamerge.readersandwriters import mergeConfigObjFromYaml
from datamerge.readersandwriters import outputToNX
from datamerge.mergecore import mergeCore
from datamerge.plotting import plotFigure
import sys


def isMac() -> bool:
    return platform == "darwin"


def getFiles(argDict: dict) -> list:
    """
    Takes the parsed command-line argument dictionary
    and returns the list of scatteringDataObjects read from the individual files
    """
    fnames = argDict["dataFiles"]

    if len(fnames) == 1:
        if fnames[0].is_dir():
            # glob the files from the globkey in Path
            fnames = sorted(fnames[0].glob(argDict["globKey"]))
            logging.info(f"Found the following files to merge: {fnames}")
    assert len(fnames) > 0, "length of filename list to merge is zero, cannot merge."

    scatteringDataList = []
    for fname in fnames:
        assert (
            fname.is_file()
        ), f"filename {fname} does not exist. Please supply valid filenames"
        scatteringDataList += [scatteringDataObjFromNX(fname)]
    return scatteringDataList


if __name__ == "__main__":
    # process input arguments
    parser = argparse.ArgumentParser(
        description="""
            Runs a datamerge binning/rebinning operation from the command line for processed MOUSE data. 
            For this to work, you need to have YAML-formatted configuration files ready. 

            Examples of these configuration files are provided in the examples subdirectory. 

            Released under a GPLv3+ license.
            """
    )
    # TODO: add info about output files to be created ...
    parser.add_argument(
        "-f",
        "--dataFiles",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parent / "testdata" / "quickstartdemo1.csv",
        help="Path to the filenames with the SAXS data. If this is a directory, all *processed.nxs files are globbed",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--globKey",
        type=str,
        default="*processed.nxs",
        help="If filename path is a directory, this will be the glob key to find the files to merge",
        # required=True,
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parent / "test.nxs",
        help="Path to the files to store the datamerge result in",
        # required=True,
    )
    parser.add_argument(
        "-C",
        "--configFile",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parent / "defaults" / "mergeConfig.yaml",
        help="Path to the datamerge configuration (yaml) file",
        # required=True,
    )

    if isMac():
        # on OSX remove automatically provided PID,
        # otherwise argparse exits and the bundle start fails silently
        for i in range(len(sys.argv)):
            if sys.argv[i].startswith("-psn"):  # PID provided by osx
                del sys.argv[i]
    try:
        args = parser.parse_args()
    except SystemExit:
        raise

    # initiate logging (to console stderr for now)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    adict = vars(args)

    try:
        dataList = getFiles(adict)
    except KeyError:
        logging.warning(
            f"The nexus files do not contain fully processed data, skipping. \n used settings: {adict}"
        )
        sys.exit(0)

    m = mergeCore(
        config=mergeConfigObjFromYaml(adict["configFile"]),
        dataList=dataList,
    )
    filteredMDO = m.run()
    # export to the final files
    ofname = Path(adict["outputFile"])
    logging.debug(f"8. Storing result in output file {ofname}")
    outputToNX(ofname=ofname, mco=m.config, mdo=filteredMDO, rangeList=m.ranges)
    # make the plots.
    plotFigure(m, ofname=Path(adict["outputFile"]))
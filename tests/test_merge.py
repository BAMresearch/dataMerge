from datetime import datetime
from pathlib import Path
import pytest
import datamerge as dm
import logging
import sys


def test_integral() -> None:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    def filelistFromArgs(argDict: dict) -> list:
        """
        Takes the parsed command-line argument dictionary
        and returns the list of filenames
        """
        fnames = argDict["dataFiles"]
        if len(fnames) == 1:
            if fnames[0].is_dir():
                # glob the files from the globkey in Path
                fnames = sorted(fnames[0].glob(argDict["globKey"]))
                logging.info(f"Found the following files to merge: {fnames}")
        assert (
            len(fnames) > 0
        ), "length of filename list to merge is zero, cannot merge."
        assert isinstance(fnames, list)
        return fnames

    # normally supplied from the command line:
    adict = {
        "dataFiles": [
            Path(
                "Y:/Measurements/SAXS002/processingCode/dataMerge/tests/data/20220925/autoproc/group_6"
            )
        ],
        "globKey": "*processed.nxs",
        "outputFile": Path("Y:/Measurements/SAXS002/processingCode/dataMerge/automatic.nxs"),
        "configFile": Path(
            "Y:/Measurements/SAXS002/processingCode/dataMerge/tests/mergeConfigExample.yaml"
        ),
    }
    dataList = dm.readersandwriters.SDOListFromFiles(
        filelistFromArgs(adict),
        readConfig=dm.readersandwriters.readConfigObjFromYaml(adict["configFile"]),
    )
    m = dm.mergecore.mergeCore(
        mergeConfig=dm.readersandwriters.mergeConfigObjFromYaml(adict["configFile"]),
        dataList=dataList,
    )
    filteredMDO, supplementaryData = m.run()

    ofname = Path(adict["outputFile"])
    ofname = dm.readersandwriters.outputToNX(
        ofname=ofname,
        mco=m.mergeConfig, 
        mdo=filteredMDO, 
        supplementaryData=supplementaryData,
        rangeList=m.ranges,
        CheckAutomaticName=True,
    )
    dm.plotting.plotFigure(m, ofname=ofname)
    return


def test_readers() -> None:
    rco = dm.readersandwriters.readConfigObjFromYaml(
        Path(".", "tests", "mergeConfigExample.yaml")
    )
    print(rco)
    so = dm.readersandwriters.scatteringDataObjFromNX(
        Path(".")
        / "tests"
        / "data"
        / "20220925"
        / "autoproc"
        / "group_6"
        / "20220925_42_expanded_stacked_processed.nxs",
        readConfig=rco,  # all defaults
    )

    mco = dm.readersandwriters.mergeConfigObjFromYaml(
        Path(".", "tests", "mergeConfigExample.yaml")
    )
    print(mco)
    return

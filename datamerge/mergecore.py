#!/usr/bin/env python
# coding: utf-8
import logging
import time
from attrs import field, define, validators, Factory
import numpy as np
from .findscaling import findScaling, findScaling_noPandas
from .dataclasses import (
    mergeConfigObj,
    mergedDataObj,
    scatteringDataObj,
    rangeConfigObj,
    supplementaryDataObj,
)

from statsmodels.stats.weightstats import DescrStatsW
from typing import List, Optional, Union
from multiprocessing.pool import ThreadPool as Pool
import importlib.util
import sys
import warnings

warnings.filterwarnings("error")


@define
class mergeCore:
    """The core merging function, most of the actual action happens here"""

    mergeConfig: mergeConfigObj = field(
        validator=validators.instance_of(mergeConfigObj)
    )
    # dataList is hardly used, instead the scatteringDataObjs in ranges is used. Depreciate?
    dataList: List[scatteringDataObj] = field(validator=validators.instance_of(list))
    # the following will be constructed in here:
    ranges: list = field(default=Factory(list), validator=validators.instance_of(list))

    preMData: Optional[scatteringDataObj] = field(
        default=None,
        validator=validators.optional(validators.instance_of(scatteringDataObj)),
    )
    mData: mergedDataObj = field(
        default=Factory(mergedDataObj), validator=validators.instance_of(mergedDataObj)
    )
    poolSize: int = field(
        default=50, validator=validators.instance_of(int), converter=int
    )
    supplementaryData: supplementaryDataObj = field(
        default=Factory(supplementaryDataObj),
        validator=validators.instance_of(supplementaryDataObj),
    )

    def constructRanges(self, dataList: list) -> None:
        """constructs a full range list for the complete set of loaded data"""
        assert (
            len(self.ranges) == 0
        ), "constructRanges should only be done starting from an empty range list"
        for ri, dataset in enumerate(dataList):
            self.ranges += [
                rangeConfigObj(
                    rangeId=ri,
                    scatteringData=dataset,
                )
            ]
        return

    def sortRanges(self, reverse: bool = False) -> None:
        """Sorts self.ranges by qMin from smallest to largest (ascending=True)"""
        self.ranges.sort(key=lambda x: x.scatteringData.qMin(), reverse=reverse)
        # reset rangeId
        # rid = 0
        # for drange in self.ranges:
        #     drange.rangeId=rid
        #     rid += 1
        [setattr(drange, "rangeId", rid) for rid, drange in enumerate(self.ranges)]
        return

    def findRangeByConfig(self, config: str, Highlander: bool = True):
        # Returns the range matching a configuration ID. The Highlander option returns a single entry,
        # will return a list, RangeConfigObj or None
        resultList = [
            drange
            for drange in self.ranges
            if drange.scatteringData.configuration == config
        ]
        if len(resultList) == 0:
            # logging.info(
            #     f"Did not find a matching range for {config} specified in the mergeConfig"
            # )
            return None  # Nothing to do, no matching ranges found
        elif (len(resultList) == 1) and Highlander:
            return resultList[0]  # return the one and only index.

        else:
            if not Highlander:
                return resultList
            else:
                logging.debug(
                    f"Found multiple entries matching configuration {config}, but a single entry requested. cannot do anything with this"
                )
                return None  # Failure

    def updateRanges(self, rangesConfigList: list) -> None:
        """
        Gets the ranges list from the configuration yaml, and applies the requested
        range settings to the rangelist in self.ranges.
        Range configuration updates can be specified by rangeId or by a findByConfig ID.
        """
        if len(rangesConfigList) == 0:
            return  # nothing to do
        for rangeConfig in rangesConfigList:
            # find out if specified by rangeId or findByConfig
            if rangeConfig.rangeId == -1:  # find by configuration ID
                assert (
                    rangeConfig.findByConfig is not None
                ), "if rangeId is set to -1, a findByConfig number must be specified"
                resultList = self.findRangeByConfig(
                    config=rangeConfig.findByConfig, Highlander=False
                )
                # assert len(resultList)>=1, 'findByConfig should result in at least one unique datafile. Did not find any matches'
                if resultList is None:
                    logging.debug(
                        f"Did not find a matching range for {rangeConfig.findByConfig=}"
                    )
                    continue  # Nothing to do, no matching ranges found
            elif rangeConfig.rangeId >= 0:  # defined by rangeId
                if rangeConfig.rangeId <= len(self.ranges):
                    resultList = [self.ranges[rangeConfig.rangeId]]  # one-element list
                else:
                    logging.warning(
                        f"Range configuration change requested for rangeId {rangeConfig.rangeId}, but this ID goes beyond the loaded files. skipping"
                    )
                    continue
            else:
                logging.warning(
                    f"{rangeConfig.rangeId=} is not a valid number, must be either -1 or the index of a range"
                )
                continue
            for drange in resultList:
                # update settings for all matching configurations
                [
                    setattr(drange, key, value)
                    for key, value in rangeConfig.items()
                    if (key in dir(drange))
                    and not (key in ["rangeId", "scatteringData"])
                ]
        return

    def rangeObjUpdate(self, rangeObj: rangeConfigObj) -> rangeConfigObj.scatteringData:
        """
        corrections and adjustments  are applied from the rangeobj:
          - scatteringData.mask
          - qMin, qMax
          - scale factor (if applicable)
          - eMin
          - qeMin
        """
        assert rangeObj.scatteringData is not None, logging.error(
            "ScatteringData cannot be none when exporting range object as DataFrame"
        )
        rangeObj.scatteringData.updateScaledMaskedValues(
            maskArray=rangeObj.scatteringData.Mask
            | rangeObj.scatteringData.returnMaskByQRange(
                qMin=rangeObj.qMinPreset, qMax=rangeObj.qMaxPreset
            ),
            scaling=rangeObj.scale,
        )
        rangeObj.scatteringData.ISigma = np.clip(
            rangeObj.scatteringData.ISigma,
            self.mergeConfig.eMin * rangeObj.scatteringData.I,
            None,
        )
        rangeObj.scatteringData.QSigma = np.clip(
            rangeObj.scatteringData.QSigma,
            self.mergeConfig.qeMin * rangeObj.scatteringData.Q,
            None,
        )

        return rangeObj.scatteringData

    def autoScale(self) -> None:
        # if any of the autoscaling dials is set to something else than itself, find the scaling factor between those datasets
        for drange in self.ranges:  # dfn = drange.rangeId, idf = drange
            drange.scale = 1.0  # reset in case of change of heart
            # if self.rangesDf.loc[dfn, 'autoscaletorange'] != dfn:
            if drange.autoscaleToConfig is not None:
                logging.debug(
                    f"autoscaling range: {drange.rangeId} to config {drange.autoscaleToConfig}"
                )
                # find index of range to scale to
                oRange = self.findRangeByConfig(drange.autoscaleToConfig)
                assert (
                    oRange is not None
                ), f"Cannot find a singular matching dataset with configuration {drange.autoscaleToConfig} to scale to, maybe none, maybe too many?"

                # apply the q limit mask to oRange mask: 
                for xRange in [oRange, drange]:
                    xRange.scatteringData.Mask |= xRange.scatteringData.returnMaskByQRange(
                        qMin=xRange.qMinPreset, 
                        qMax=xRange.qMaxPreset
                    )
                # assert drange.autoscaleToRange <= len(
                #     self.ranges
                # ), f"{drange.autoscaleToRange=} must refer to an index within the number of ranges available {len(self.ranges)}"
                # oRange = self.ranges[drange.autoscaleToRange]
                logging.debug(f'trying to find scaling factor between {drange.scatteringData.configuration=} and {oRange.scatteringData.configuration=}')
                fs = findScaling_noPandas(
                    oRange.scatteringData,  # original data, asPandas() returns copy
                    drange.scatteringData,
                    backgroundFit=False,  # just the scaling factor, please, no additional background.
                )
                fs.run()
                drange.scale = float(fs.sc[0])
                logging.info(
                    f"Scaling added to rangeindex: {drange.rangeId}, congfiguration: {drange.scatteringData.configuration} : {float(fs.sc[0])}"
                )
        return

    def concatAllUnmergedData(self) -> None:
        """
        Reads all the datasets in self.dataList, applying masks, limits and scaling factors where necessary.
        All the datasets are then put in a single scatteringDataObj instance in self.preMData,
        ready to be merged.
        """
        # nothing to do
        cfgs = []
        if len(self.ranges) == 1:
            scattering_data = self.rangeObjUpdate(self.ranges[0])
            cfgs += [scattering_data.configuration]
        else:
            with Pool(self.poolSize) as pool:
                scattering_data_per_range = [
                    d for d in pool.imap_unordered(self.rangeObjUpdate, self.ranges)
                ]

            # determine configurations that went in to the concatenated data:
            for sd in scattering_data_per_range:
                if sd.configuration != '-1':
                    cfgs += [sd.configuration]
                if len(sd.configurations) > 0:
                    cfgs += [i for i in sd.configurations]

            try:
                scattering_data = scatteringDataObj(
                    Q=np.concatenate(
                        [i.Q for i in scattering_data_per_range],
                        axis=None,
                        dtype=scattering_data_per_range[0].Q.dtype,
                    ),
                    I=np.concatenate(
                        [i.I for i in scattering_data_per_range],
                        axis=None,
                        dtype=scattering_data_per_range[0].I.dtype,
                    ),
                    ISigma=np.concatenate(
                        [i.ISigma for i in scattering_data_per_range],
                        axis=None,
                        dtype=scattering_data_per_range[0].ISigma.dtype,
                    ),
                    QSigma=np.concatenate(
                        [i.QSigma for i in scattering_data_per_range],
                        axis=None,
                        dtype=scattering_data_per_range[0].QSigma.dtype,
                    ),
                    Mask=np.concatenate(
                        [i.Mask for i in scattering_data_per_range],
                        axis=None,
                        dtype=scattering_data_per_range[0].Mask.dtype,
                    ),
                    # Q= np.array([i.Q for i in scattering_data_per_range ]).flatten(),
                    # I=np.array([i.I for i in scattering_data_per_range ]).flatten(),
                    # ISigma=np.array([i.ISigma for i in scattering_data_per_range ]).flatten(),
                    # QSigma=np.array([i.QSigma for i in scattering_data_per_range ]).flatten(),
                    # Mask=np.array([i.Mask for i in scattering_data_per_range ]).flatten(),
                    # sampleName=scattering_data_per_range[0].sampleName if scattering_data_per_range[0].sampleName is not None else 'na',
                    # sampleOwner=scattering_data_per_range[0].sampleOwner if scattering_data_per_range[0].sampleOwner is not None else 'na',
                    sampleName=scattering_data_per_range[
                        0
                    ].sampleName,  #  not sure how this deals with none
                    sampleOwner=scattering_data_per_range[0].sampleOwner,
                    configuration='-1',  # unspecified
                    configurations=cfgs,
                )
            except np.VisibleDeprecationWarning:  # when would this occur?
                scattering_data = scatteringDataObj(
                    # Q=np.array([t for sd in scattering_data_per_range  for t in sd.Q]),
                    # I=np.array([t for sd in scattering_data_per_range  for t in sd.I]),
                    # ISigma=np.array([t for sd in scattering_data_per_range  for t in sd.ISigma]),
                    # QSigma=np.array([t for sd in scattering_data_per_range  for t in sd.QSigma]),
                    # Mask=np.array([t for sd in scattering_data_per_range  for t in sd.Mask]),
                    Q=np.concatenate(
                        [t for sd in scattering_data_per_range for t in sd.Q],
                        axis=None,
                        dtype=scattering_data_per_range[0].Q.dtype,
                    ),
                    I=np.concatenate(
                        [t for sd in scattering_data_per_range for t in sd.I],
                        axis=None,
                        dtype=scattering_data_per_range[0].I.dtype,
                    ),
                    ISigma=np.concatenate(
                        [t for sd in scattering_data_per_range for t in sd.ISigma],
                        axis=None,
                        dtype=scattering_data_per_range[0].ISigma.dtype,
                    ),
                    QSigma=np.concatenate(
                        [t for sd in scattering_data_per_range for t in sd.QSigma],
                        axis=None,
                        dtype=scattering_data_per_range[0].QSigma.dtype,
                    ),
                    Mask=np.concatenate(
                        [t for sd in scattering_data_per_range for t in sd.Mask],
                        axis=None,
                        dtype=scattering_data_per_range[0].Mask.dtype,
                    ),
                    sampleName=scattering_data_per_range[
                        0
                    ].sampleName,  #  not sure how this deals with none
                    sampleOwner=scattering_data_per_range[0].sampleOwner,
                    configuration='-1',  # unspecified
                    configurations=cfgs,
                )

        self.supplementaryData = supplementaryDataObj(
            configurations=cfgs,
            sampleName=scattering_data.sampleName,  #  not sure how this deals with none
            sampleOwner=scattering_data.sampleOwner,
        )
        # drop empty rows (Q and I are nan)
        nonempty_bin_index = np.argwhere(
            ~(np.isnan(scattering_data.Q) & np.isnan(scattering_data.I))
        ).flatten()
        self.preMData = scatteringDataObj(
            Q=scattering_data.Q[nonempty_bin_index],
            I=scattering_data.I[nonempty_bin_index],
            ISigma=scattering_data.ISigma[nonempty_bin_index],
            QSigma=scattering_data.QSigma[nonempty_bin_index],
            Mask=scattering_data.Mask[nonempty_bin_index],
            sampleName=scattering_data.sampleName,  #  not sure how this deals with none
            sampleOwner=scattering_data.sampleOwner,
            configuration='-1', # unspecified
            configurations=cfgs,
        )
        return

    def sortUnmergedData(self) -> None:
        sorter = np.argsort(self.preMData.Q)
        self.preMData.Q = self.preMData.Q[sorter]
        self.preMData.I = self.preMData.I[sorter]
        self.preMData.ISigma = self.preMData.ISigma[sorter]
        self.preMData.QSigma = self.preMData.QSigma[sorter]
        self.preMData.Mask = self.preMData.Mask[sorter]
        return

    def nonZeroQMin(self) -> float:
        """Returns the smallest nonzero q value for all input ranges for starting the binning at."""
        qMin = np.inf
        for drange in self.ranges:
            if drange.qMinPreset is not None:
                qMin = np.min([qMin, drange.qMinPreset])
            elif drange.scatteringData.qMin() > 0:
                qMin = np.min([qMin, drange.scatteringData.qMin()])
        return qMin

    def nonZeroQMax(self) -> float:
        """Returns the largest non-infinie q value for all input datasets, for ending the binning at."""
        qMax = 0.0
        for drange in self.ranges:
            if drange.qMaxPreset is not None:
                qMax = np.max([qMax, drange.qMaxPreset])
            elif drange.scatteringData.qMax() < np.inf:
                qMax = np.max([qMax, drange.scatteringData.qMax()])
        return qMax

    def createBinEdges(self) -> np.ndarray:
        binEdges = list()
        assert self.mergeConfig.outputRanges is not None, logging.error(
            "at least one output range should be specified"
        )
        for rangeId, outRange in enumerate(self.mergeConfig.outputRanges):
            qEnd = self.nonZeroQMax()
            qStart = (
                outRange.qCrossover
                if (outRange.qCrossover != 0) and (np.isfinite(outRange.qCrossover))
                else self.nonZeroQMin()
            )
            if rangeId < (
                len(self.mergeConfig.outputRanges) - 1
            ):  # set qEnd to the start of the next range
                newQEnd = self.mergeConfig.outputRanges[rangeId + 1].qCrossover
                if np.isfinite(newQEnd):
                    qEnd = newQEnd
            if outRange.QScaling == "log":
                binEdges += [np.geomspace(qStart, qEnd, num=outRange.nbins + 1)]
            else:
                binEdges += [np.linspace(qStart, qEnd, num=outRange.nbins + 1)]

        be = np.concatenate(binEdges, dtype=float)
        # add a little to the end to ensure the last datapoint is included:
        assert len(be > 3), logging.error(
            "number of bins in total range must exceed at least 3.."
        )
        be[-1] = be[-1] + 1e-3 * (be[-1] - be[-2])
        return be

    def SEMw(x, w):
        """
        function adapted from: https://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
        citing: The main reference is this paper, by Donald F. Gatz and Luther Smith, where 3 formula based estimators are compared with bootstrap results. The best approximation to the bootstrap result comes from Cochran (1977):
        side note: this provides rubbish results.
        """
        n = len(w)
        # dummy = [print(f"x: {xi}, {wi}") for xi, wi in zip(x, w)]
        xWbar = np.average(x, weights=w)
        wbar = np.mean(w)
        out = (
            n
            / ((n - 1) * np.sum(w) ** 2)
            * (
                np.sum((w * x - wbar * xWbar) ** 2)
                - 2 * xWbar * np.sum((w - wbar) * (w * x - wbar * xWbar))
                + xWbar**2 * np.sum((w - wbar) ** 2)
            )
        )
        return out

    def mergyMagic(self, binEdges: np.ndarray, calcSEMw: bool = False) -> None:
        # define weighted standard error on the mean:

        self.mData = mergedDataObj(
            Q=np.full(len(binEdges) - 1, np.nan),
            I=np.full(len(binEdges) - 1, np.nan),
            IStd=np.full(len(binEdges) - 1, np.nan),
            ISEM=np.full(len(binEdges) - 1, np.nan),
            ISEMw=np.full(len(binEdges) - 1, np.nan),
            IEPropagated=np.full(len(binEdges) - 1, np.nan),
            ISigma=np.full(len(binEdges) - 1, np.nan),
            QStd=np.full(len(binEdges) - 1, np.nan),
            QSEM=np.full(len(binEdges) - 1, np.nan),
            QSigma=np.full(len(binEdges) - 1, np.nan),
            Mask=np.full(
                len(binEdges) - 1, True, dtype=bool
            ),  # all masked until filled in
            Singles=np.full(
                len(binEdges) - 1, False, dtype=bool
            ),  # Set to True for bins containing just one datapoint
        )
        logging.debug(f"{binEdges=}")
        assert self.preMData is not None, logging.warning(
            "self.preMData cannot be none at the merging step"
        )

        # # set the supplementary information:
        # self.mData.sampleName=self.preMData.sampleName #  not sure how this deals with none
        # self.mData.sampleOwner=self.preMData.sampleOwner
        # self.mData.configurations=self.preMData.configurations

        edgeIndices = np.searchsorted(
            self.preMData.Q, binEdges
        )  # last edge should be slightly outside last
        # we can precalculate the weights for all datapoints:
        if self.mergeConfig.IEWeighting:
            self.preMData.wt = np.abs(
                self.preMData.I / (self.preMData.ISigma**2)
            )  # inverse relative weight per point if desired.
        else:
            self.preMData.wt = np.ones_like(self.preMData.I)  # no datapoint weighting

        def binDfRangeByIndex(binN: int) -> None:
            lowerIndex, upperIndex = edgeIndices[binN], edgeIndices[binN + 1]
            rangeLen = upperIndex - lowerIndex
            if rangeLen == 0:  # nothing in bin
                return
            if rangeLen == 1:  # one datapoint in bin
                # might not be necessary to do this..
                # can't do stats on this:
                self.mData.Q[binN] = float(self.preMData.Q[lowerIndex:upperIndex][0])
                self.mData.I[binN] = float(self.preMData.I[lowerIndex:upperIndex][0])
                self.mData.IStd[binN] = float(
                    self.preMData.ISigma[lowerIndex:upperIndex][0]
                )
                self.mData.ISEM[binN] = float(
                    self.preMData.ISigma[lowerIndex:upperIndex][0]
                )
                self.mData.ISEMw[binN] = float(
                    self.preMData.ISigma[lowerIndex:upperIndex][0]
                )
                self.mData.IEPropagated[binN] = float(
                    self.preMData.ISigma[lowerIndex:upperIndex][0]
                )
                self.mData.ISigma[binN] = float(
                    self.preMData.ISigma[lowerIndex:upperIndex][0]
                )
                self.mData.QStd[binN] = float(
                    self.preMData.QSigma[lowerIndex:upperIndex][0]
                )
                self.mData.QSEM[binN] = float(
                    self.preMData.QSigma[lowerIndex:upperIndex][0]
                )
                self.mData.QSigma[binN] = float(
                    self.preMData.QSigma[lowerIndex:upperIndex][0]
                )
                self.mData.Singles[binN] = True
                self.mData.Mask[binN] = False
                return

            else:  # multiple datapoints in bin
                name = "binstats"
                # check if cython modules are available
                if (spec := importlib.util.find_spec(name)) is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    if self.preMData.I.dtype == "float64":
                        self.mData.I[binN] = module.weighted_mean(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.IStd[binN] = module.std_ddof(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        # following suggestion regarding V1/V2 from: https://groups.google.com/forum/#!topic/medstats/H4SFKPBDAAM
                        self.mData.ISEM[binN] = module.sem(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )

                        if calcSEMw:
                            self.mData.ISEMw[binN] = module.sem_weighted(
                                self.preMData.I[lowerIndex:upperIndex],
                                self.preMData.wt[lowerIndex:upperIndex],
                            )
                    else:
                        self.mData.I[binN] = module.weighted_mean_sp(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.IStd[binN] = module.std_ddof_sp(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.ISEM[binN] = module.sem_sp(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )

                        if calcSEMw:
                            self.mData.ISEMw[binN] = module.sem_weighted_sp(
                                self.preMData.I[lowerIndex:upperIndex],
                                self.preMData.wt[lowerIndex:upperIndex],
                            )

                    if self.preMData.ISigma.dtype == "float64":
                        self.mData.ISigma[binN] = module.sigma(
                            self.preMData.ISigma[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                    else:
                        self.mData.ISigma[binN] = module.sigma_sp(
                            self.preMData.ISigma[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                    self.mData.IEPropagated[binN] = module.propagated_error(
                        self.mData.ISEM[binN],
                        self.mData.ISigma[binN],
                        self.mData.I[binN],
                        self.mergeConfig.eMin,
                    )

                    if (
                        self.preMData.Q.dtype == "float64"
                        and self.preMData.wt.dtype == "float64"
                    ):
                        self.mData.Q[binN] = module.weighted_mean(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.QStd[binN] = module.std_ddof(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.QSEM[binN] = module.sem(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                    elif (
                        self.preMData.Q.dtype == "float32"
                        and self.preMData.wt.dtype == "float32"
                    ):
                        self.mData.Q[binN] = module.weighted_mean_sp(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.QStd[binN] = module.std_ddof_sp(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                        self.mData.QSEM[binN] = module.sem_sp(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )
                    elif self.preMData.Q.dtype == "float32":
                        self.mData.Q[binN] = module.weighted_mean_sp(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex].astype("float32"),
                        )
                        self.mData.QStd[binN] = module.std_ddof_sp(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex].astype("float32"),
                        )
                        self.mData.QSEM[binN] = module.sem_sp(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex].astype("float32"),
                        )
                    else:
                        self.mData.Q[binN] = module.weighted_mean(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex].astype("float64"),
                        )
                        self.mData.QStd[binN] = module.std_ddof(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex].astype("float64"),
                        )
                        self.mData.QSEM[binN] = module.sem(
                            self.preMData.Q[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex].astype("float64"),
                        )

                    self.mData.QSigma[binN] = module.propagated_error(
                        self.mData.QSEM[binN],
                        -0.1,
                        self.mData.Q[binN],
                        self.mergeConfig.eMin,
                    )  # instead of second value which was I sigma in the above call of the func, set to negative
                    self.mData.Mask[binN] = False
                else:
                    # exploit the DescrStatsW package from statsmodels
                    DSI = DescrStatsW(
                        self.preMData.I[lowerIndex:upperIndex],
                        weights=self.preMData.wt[lowerIndex:upperIndex],
                    )
                    DSQ = DescrStatsW(
                        self.preMData.Q[lowerIndex:upperIndex],
                        weights=self.preMData.wt[lowerIndex:upperIndex],
                    )
                    self.mData.Q[binN] = DSQ.mean
                    self.mData.I[binN] = DSI.mean
                    self.mData.ISigma[binN] = (
                        np.sqrt(
                            (
                                (
                                    self.preMData.wt[lowerIndex:upperIndex]
                                    * self.preMData.ISigma[lowerIndex:upperIndex]
                                )
                                ** 2
                            ).sum()
                        )
                        / self.preMData.wt[lowerIndex:upperIndex].sum()
                    )
                    self.mData.IStd[binN] = DSI.std
                    # following suggestion regarding V1/V2 from: https://groups.google.com/forum/#!topic/medstats/H4SFKPBDAAM
                    self.mData.ISEM[binN] = DSI.std * np.sqrt(
                        (self.preMData.wt[lowerIndex:upperIndex] ** 2).sum()
                        / (self.preMData.wt[lowerIndex:upperIndex].sum()) ** 2
                    )
                    if calcSEMw:
                        self.mData.ISEMw[binN] = SEMw(
                            self.preMData.I[lowerIndex:upperIndex],
                            self.preMData.wt[lowerIndex:upperIndex],
                        )  # adds considerable time, and we're not using it at the mo.
                    self.mData.IEPropagated[binN] = np.max(
                        [
                            self.mData.ISEM[binN],
                            self.mData.ISigma[binN],
                            DSI.mean * self.mergeConfig.eMin,
                        ]
                    )
                    self.mData.QStd[binN] = DSQ.std
                    self.mData.QSEM[binN] = DSQ.std * np.sqrt(
                        (self.preMData.wt[lowerIndex:upperIndex] ** 2).sum()
                        / (self.preMData.wt[lowerIndex:upperIndex].sum()) ** 2
                    )
                    self.mData.QSigma[binN] = np.max(
                        [self.mData.QSEM[binN], DSQ.mean * self.mergeConfig.qeMin]
                    )
                    self.mData.Mask[binN] = False
            return

        # separate:
        with Pool(self.poolSize) as pool:
            _ = [
                i
                for i in pool.imap_unordered(
                    binDfRangeByIndex, range(len(binEdges) - 1)
                )
            ]

        return

    def returnMaskedOutput(
        self, maskMasked: bool = True, maskSingles: bool = False
    ) -> mergedDataObj:
        newMDO = mergedDataObj()
        assert self.mData.Mask is not None, logging.warning(
            "Mask array must be filled at this point"
        )
        mask = np.zeros(shape=self.mData.Mask.shape, dtype=bool)
        if maskMasked:
            mask += self.mData.Mask
        if maskSingles:
            assert self.mData.Singles is not None, logging.warning(
                "Singles array must be filled at this point"
            )
            mask += self.mData.Singles
        keyList = ["Q"]  # Q must be set first
        keyList += [key for key in newMDO.keys() if not key in keyList]

        assert mask.sum() < len(mask), logging.error(
            "Cannot output anything, all values masked"
        )
        for key in keyList:
            setattr(newMDO, key, self.mData[key][~mask])
        return newMDO

    def run(self) -> mergedDataObj:
        # config is already read, and the raw data has been loaded into a list of scatteringData objects
        # construct initial list of ranges
        starttime = time.time()
        # logging.info("1. constructing ranges. t=0")
        self.constructRanges(self.dataList)
        # sort by qMin so that index 0 is the one with the smallest qMin
        # logging.info(f"2. sorting ranges. t={time.time() - starttime}")
        self.sortRanges()
        [
            logging.debug(
                f"rID: {dr.rangeId}, has SDL: {dr.scatteringData is not None}, qMin: {dr.scatteringData.qMin()}"
            )
            for dr in self.ranges
        ]
        # update ranges with custom configuration if necessary
        if self.mergeConfig.ranges is not None:
            # logging.info(f"2.1 updating ranges. t={time.time() - starttime}")
            logging.debug(f"{self.mergeConfig.ranges=}")
            self.updateRanges(self.mergeConfig.ranges)
        # determine scaling factors
        # logging.info(f"3. applying autoscaling, t={time.time() - starttime}")
        self.autoScale()
        # just checking it makes it to here.
        o = [
            f"{dr.rangeId}({dr.scatteringData.configuration}): {dr.scale}"
            for dr in self.ranges
        ]
        # logging.info(f"AutoScale done, scaling factors: {o}")
        # logging.debug(f"{self.mergeConfig.outputRanges=}")
        # do I need to resort the data by Q? I don't think so... was part of the original though.
        # read all the data into a single dataframe, taking care of scaling, clipping and masking
        # logging.info(f"4. concatenating original data, t={time.time() - starttime}")
        self.concatAllUnmergedData()
        # Sort for posterity
        # logging.info(f"5. Sorting unmerged data, t={time.time() - starttime}")
        self.sortUnmergedData()
        # apply mergyMagic to the list of q Edges.
        # logging.info(f"6. creating bin edges, t={time.time() - starttime}")
        binEdges = self.createBinEdges()
        # logging.info(f"6. merging within bin edges, t={time.time() - starttime}")
        # this is the bottleneck... not surprising but still.
        self.mergyMagic(binEdges=binEdges)
        # filter result
        # logging.info(f"7. filtering out invalid points from merged data with {self.mergeConfig.maskMasked=} and {self.mergeConfig.maskSingles=}, t={time.time() - starttime}" )
        filteredMDO = self.returnMaskedOutput(
            maskMasked=self.mergeConfig.maskMasked,
            maskSingles=self.mergeConfig.maskSingles,
        )
        # logging.info(f"7.1 done filtering, t={time.time() - starttime}")

        return filteredMDO, self.supplementaryData

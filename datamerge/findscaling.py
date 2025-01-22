#!/usr/bin/env python
# coding: utf-8

"""
Overview:
========
This tool finds the scaling factor to bring the second curve in line with the first.
The Q-values are NOT expected to match

Required input arguments:
    *Q1*: Q-vector of the first dataset
    *I1*: intensity of the first dataset
    *E1*: relative intensity uncertainty of the first dataset
    *Q2*: Q-vector of the first dataset
    *I2*: intensity of the second dataset
    *E2*: relative intensity uncertainty of the second dataset
Optional input arguments: 
    *backgroundFit*: Boolean indicating whether or not to fit the background,
        Default: True
"""

__author__ = "Brian R. Pauw"
__contact__ = "brian@stack.nl"
__license__ = "GPLv3+"
__date__ = "2016/04/15"
__status__ = "beta"

# correctionBase contains the __init__ function for these classes
from pathlib import Path
from typing import Optional
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
import logging
import pandas as pd
from attrs import define, validators, field, cmp_using
import numpy as np
import copy 

from datamerge.dataclasses import scatteringDataObj

def object_copy_converter(value):
    # Create a deepcopy to ensure the copy is independent of the original object
    return copy.deepcopy(value)

@define
class findScaling_noPandas(object):
    """
    new version of findScaling, modified to take scatteringDataObj instead of pd.DataFrame. 
    """
    # make sure we're working on copies. 
    dataset1: scatteringDataObj = field(validator=validators.instance_of(scatteringDataObj), converter = object_copy_converter)
    dataset2: scatteringDataObj = field(validator=validators.instance_of(scatteringDataObj), converter = object_copy_converter)

    backgroundFit: bool = field(default=True, validator=validators.instance_of(bool))
    doInterpolate: bool = field(default=True, validator=validators.instance_of(bool))

    # internal
    Mask: Optional[np.ndarray]=field(default=None, validator=validators.optional(validators.instance_of(np.ndarray)), init=False)

    sc: np.ndarray = field(
        default=np.array([1, 0], dtype=float),
        validator=validators.instance_of(np.ndarray),
        eq=cmp_using(eq=np.array_equal),
    )

    def run(self) -> None:
        # check Q
        logging.debug('running findScaling_noPandas')
        if self.dataset2.Q.shape != self.dataset1.Q.shape:
            logging.debug('Q vectors are not the same shape, interpolating...')
            self.doInterpolate = True
        elif (self.dataset2.Q != self.dataset1.Q).any():
            logging.debug("nonequal Q vectors, interpolating...")
            self.doInterpolate = True

        logging.debug(f'Q limits of datasets before interpolation: {self.dataset1.qMinNonMasked()=:0.02f} {self.dataset1.qMaxNonMasked()=:0.02f}, {self.dataset2.qMinNonMasked()=:0.02f} {self.dataset2.qMaxNonMasked()=:0.02f}')
        # we only need to match the overlapping range: 
        overlappingQLimits = (
            np.maximum(self.dataset1.qMinNonMasked(), self.dataset2.qMinNonMasked()), 
            np.minimum(self.dataset1.qMaxNonMasked(), self.dataset2.qMaxNonMasked()))
        self.dataset1.Mask |= self.dataset1.returnMaskByQRange(overlappingQLimits[0], overlappingQLimits[1])
        self.dataset2.Mask |= self.dataset2.returnMaskByQRange(overlappingQLimits[0], overlappingQLimits[1])
        self.dataset1.updateScaledMaskedValues(maskArray=self.dataset1.Mask, scaling=1.0)
        self.dataset2.updateScaledMaskedValues(maskArray=self.dataset2.Mask, scaling=1.0)
        # print(f'{self.dataset1=}, {self.dataset2=}')
        if self.doInterpolate:
            self.dataset2 = self.interpolate(
                dataset=self.dataset2, interpQ=self.dataset1.Q
            )
        logging.debug(f'Q limits of datasets after interpolation: {self.dataset1.qMinNonMasked()=:0.02f} {self.dataset1.qMaxNonMasked()=:0.02f}, {self.dataset2.qMinNonMasked()=:0.02f} {self.dataset2.qMaxNonMasked()=:0.02f}')
        self.Mask = np.zeros(self.dataset1.Q.shape, dtype=bool)  # none masked
        self.Mask |= self.dataset1.Mask
        self.Mask |= self.dataset2.Mask
        self.sc = self.scale()

        # we omit the generation of this for speed purposes. 
        # self.dataset2Scaled = self.dataset2.copy()
        # self.dataset2Scaled.I *= self.sc[0]
        # self.dataset2Scaled.I += self.sc[1]
        # self.dataset2Scaled.ISigma *= self.sc[0]
        # self.dataset2Scaled.Mask = self.Mask
        # calculate curve deviation:
        # self.dev = (self.dataset1.I - (self.dataset2.I*self.sc[0] + self.sc[1]) ) / self.dataset1.I

        return

    def scale(self) -> np.ndarray:
        # match the curves
        iMask = np.invert(self.Mask)
        if sum(iMask)==0: 
            logging.warning('No overlapping data found for scaling datasets, leaving scaling at 1.0')
            return np.array((1.0, 0.0), dtype=float)
        sc = np.zeros(2)
        sc[1] = self.dataset1.I[iMask].min() - self.dataset2.I[iMask].min()
        sc[0] = (self.dataset1.I[iMask] / self.dataset2.I[iMask]).mean()
        sc, _ = leastsq(self.csqr, sc)
        if not self.backgroundFit:
            sc[1] = 0.0

        _ = self.csqrV1(sc)
        return sc

    def interpolate(
        self, dataset: scatteringDataObj = None, interpQ: np.ndarray = None
    ) -> scatteringDataObj:
        """interpolation function that interpolates provided dataset, returning a Pandas dataframe
        with Q, I, IError and Mask fields. the provided interpQ values are attempted to be kept,
        although values outside interpQ will be filled with nan, and Mask set to True there."""

        # interpolator (linear) to equalize Q.
        fI = interp1d(dataset.Q, dataset.I, kind="linear", bounds_error=False)
        fE = interp1d(
            dataset.Q, dataset.ISigma, kind="linear", bounds_error=False
        )

        logging.debug(f'{dataset.Q=}, {dataset.I=}, {dataset.ISigma=}, {interpQ=}')
        # not a full copy!
        dst = scatteringDataObj(
            Q = interpQ,
            I = fI(interpQ), # initialize as nan
            ISigma = fE(interpQ),
            Mask = np.invert(np.isfinite(fI(interpQ)) & np.isfinite(fE(interpQ))),  # none masked that are finite
        )
        logging.debug(f'{dst=}')

        return dst

    def csqr(self, sc, useMask=True):
        # csqr to be used with scipy.optimize.leastsq
        if useMask:
            # mask = (np.invert(self.dataset1["Mask"]) & np.invert(self.dataset2["Mask"]))
            mask = np.invert(self.Mask)
        else:
            mask = np.ones(self.dataset1.I.shape, dtype="bool")
        I1 = self.dataset1.I[mask]
        E1 = self.dataset1.ISigma[mask]
        I2 = self.dataset2.I[mask]
        E2 = self.dataset2.ISigma[mask]
        if not self.backgroundFit:
            bg = 0.0
        else:
            bg = sc[1]
        return (I1 - sc[0] * I2 - bg) / (np.sqrt(E1**2 + E2**2))

    def csqrV1(self, sc, useMask=True):
        # complete reduced chi-squared calculation
        if useMask:
            # mask = (np.invert(self.dataset1["Mask"]) & np.invert(self.dataset2["Mask"]))
            mask = np.invert(self.Mask)
        else:
            mask = np.ones(self.dataset1["I"].shape, dtype="bool")
        I1 = self.dataset1.I[mask]
        E1 = self.dataset1.ISigma[mask]
        I2 = self.dataset2.I[mask] * sc[0] + sc[1]
        E2 = self.dataset2.ISigma[mask] * sc[0]

        return sum(((I1 - I2) / (np.sqrt(E1**2 + E2**2))) ** 2) / np.size(I1)


@define
class findScaling(object):
    dataset1: pd.DataFrame = field()

    @dataset1.validator
    def check_dset1(instance, attribute, value):
        assert isinstance(value, pd.DataFrame)
        for key in ["Q", "I", "ISigma"]:
            assert key in value.keys(), f"required {key=} is missing from dataset1"

    dataset2: pd.DataFrame = field(validator=validators.instance_of(pd.DataFrame))

    @dataset2.validator
    def check_dset2(instance, attribute, value):
        assert isinstance(value, pd.DataFrame)
        for key in ["Q", "I", "ISigma"]:
            assert key in value.keys(), f"required {key=} is missing from dataset1"

    backgroundFit: bool = field(default=True, validator=validators.instance_of(bool))
    doInterpolate: bool = field(default=True, validator=validators.instance_of(bool))
    sc: np.ndarray = field(
        default=np.array([1, 0], dtype=float),
        validator=validators.instance_of(np.ndarray),
        eq=cmp_using(eq=np.array_equal),
    )

    def run(self) -> None:
        # check Q
        if self.dataset2["Q"].shape != self.dataset1["Q"].shape:
            self.doInterpolate = True
        elif (self.dataset2["Q"] != self.dataset1["Q"]).any():
            logging.warning("nonequal Q vectors, interpolating...")
            self.doInterpolate = True

        if self.doInterpolate:
            self.dataset2 = self.interpolate(
                dataset=self.dataset2, interpQ=self.dataset1["Q"]
            )

        self.Mask = np.zeros(self.dataset1["Q"].shape, dtype=bool)  # none masked
        if "Mask" in self.dataset1:
            self.Mask |= self.dataset1["Mask"]
        if "Mask" in self.dataset2:
            self.Mask |= self.dataset2["Mask"]
        self.sc = self.scale(
            dataset1=self.dataset1[np.invert(self.Mask)],
            dataset2=self.dataset2[np.invert(self.Mask)],
        )

        self.dataset2Scaled = self.dataset2.copy()
        self.dataset2Scaled["I"] *= self.sc[0]
        self.dataset2Scaled["I"] += self.sc[1]
        self.dataset2Scaled["IError"] *= self.sc[0]
        self.dataset2Scaled["Mask"] = self.Mask
        # calculate curve deviation:
        self.dev = (self.dataset1["I"] - self.dataset2Scaled["I"]) / self.dataset1["I"]

        return

    def scale(self) -> np.ndarray:
        # match the curves
        sc = np.zeros(2)
        sc[1] = self.dataset1["I"].min() - self.dataset2["I"].min()
        sc[0] = (self.dataset1["I"] / self.dataset2["I"]).mean()
        sc, _ = leastsq(self.csqr, sc)
        if not self.backgroundFit:
            sc[1] = 0.0

        _ = self.csqrV1(sc)
        return sc

    def interpolate(
        self, dataset: pd.DataFrame = None, interpQ: np.ndarray = None
    ) -> pd.DataFrame:
        """interpolation function that interpolates provided dataset, returning a Pandas dataframe
        with Q, I, IError and Mask fields. the provided interpQ values are attempted to be kept,
        although values outside interpQ will be filled with nan, and Mask set to True there."""

        dst = pd.DataFrame()
        dst["Q"] = interpQ
        dst["I"] = np.full(interpQ.shape, np.nan)  # initialize as nan
        dst["IError"] = np.full(interpQ.shape, np.nan)
        dst["Mask"] = np.zeros(interpQ.shape, dtype=bool)  # none masked

        # interpolator (linear) to equalize Q.
        fI = interp1d(dataset["Q"], dataset["I"], kind="linear", bounds_error=False)
        fE = interp1d(
            dataset["Q"], dataset["IError"], kind="linear", bounds_error=False
        )

        # interpolate, rely on Mask to deliver final limits
        dst["I"] = fI(interpQ)
        dst["IError"] = fE(interpQ)

        # extra mask clip based on I or IError values:
        dst["Mask"] |= dst["I"].isnull()
        dst["Mask"] |= dst["IError"].isnull()

        # dst[np.invert(dst["Mask"])] # not masked values

        # return interpolated dataset
        # self.dataset2 = dst
        return dst

    def csqr(self, sc, useMask=True):
        # csqr to be used with scipy.optimize.leastsq
        if useMask:
            # mask = (np.invert(self.dataset1["Mask"]) & np.invert(self.dataset2["Mask"]))
            mask = np.invert(self.Mask)
        else:
            mask = np.ones(self.dataset1["I"].shape, dtype="bool")
        I1 = self.dataset1["I"][mask]
        E1 = self.dataset1["IError"][mask]
        I2 = self.dataset2["I"][mask]
        E2 = self.dataset2["IError"][mask]
        if not self.backgroundFit:
            bg = 0.0
        else:
            bg = sc[1]
        return (I1 - sc[0] * I2 - bg) / (np.sqrt(E1**2 + E2**2))

    def csqrV1(self, sc, useMask=True):
        # complete reduced chi-squared calculation
        if useMask:
            # mask = (np.invert(self.dataset1["Mask"]) & np.invert(self.dataset2["Mask"]))
            mask = np.invert(self.Mask)
        else:
            mask = np.ones(self.dataset1["I"].shape, dtype="bool")
        I1 = self.dataset1["I"][mask]
        E1 = self.dataset1["IError"][mask]
        I2 = self.dataset2["I"][mask] * sc[0] + sc[1]
        E2 = self.dataset2["IError"][mask] * sc[0]

        return sum(((I1 - I2) / (np.sqrt(E1**2 + E2**2))) ** 2) / np.size(I1)

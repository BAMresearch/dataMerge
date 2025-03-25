#!/usr/bin/env python
# coding: utf-8

"""
Overview:
========
Datamerge's dataclasses are defined here. 
"""

__author__ = "Brian R. Pauw"
__contact__ = "brian@stack.nl"
__license__ = "GPLv3+"
__date__ = "2022/10/18"
__status__ = "beta"

from attrs import Factory
from attrs import define, validators, field, cmp_using, fields
from pathlib import Path
import numpy as np
from typing import List, Optional, Any, NoReturn, Union
from collections.abc import Iterable


def QChecker(instance, attribute, value):
    assert isinstance(value, np.ndarray)
    assert (
        value.ndim == 1
    ), f"{attribute=} can only have 1 dimension (2D not implemented yet)"


def scalingChecker(instance, attribute, value):
    assert value in [
        "log",
        "linear",
    ], f"{attribute=} should be either set to 'log' or 'linear'"


def IChecker(instance, attribute, value):
    if value is None:
        return
    assert isinstance(value, np.ndarray)
    assert (
        value.shape == instance.Q.shape
    ), f"Shape of {attribute.name=} with value {value} must match shape of Q: {instance.Q.shape}"
    assert (
        value.ndim == 1
    ), f"{attribute=} can only have 1 dimension (2D not implemented yet)"


# Mixin class for making a dict-like object out of an attrs class
# from: https://github.com/python-attrs/attrs/issues/879
class gimmeItems:  # used to be MutableMappingMixin(MutableMapping)
    """Mixin class to make attrs classes quack like a dictionary (well,
    technically a mutable mapping). ONLY use this with attrs classes.

    Provides keys(), values(), and items() methods in order to be
    dict-like in addition to MutableMapping-like. Also provides pop(),
    but it just raises a TypeError :)
    """

    __slots__ = ()  # May as well save on memory?

    def __iter__(self) -> Iterable:
        for ifield in fields(self.__class__):
            yield ifield.name

    def __len__(self) -> int:
        return len(fields(self.__class__))

    def __getitem__(self, k: str) -> Any:
        """
        Adapted from:
        https://github.com/python-attrs/attrs/issues/487#issuecomment-660727537
        """
        try:
            return self.__getattribute__(k)
        except AttributeError as exc:
            raise KeyError(str(exc)) from None

    def __delitem__(self, v: str) -> NoReturn:
        raise TypeError("Cannot delete fields for attrs classes.")

    def __setitem__(self, k: str, v: Any) -> None:
        self.__setattr__(k, v)

    def pop(self, key, default=None) -> NoReturn:
        raise TypeError("Cannot pop fields from attrs classes.")

    def keys(self) -> Iterable:
        return self.__iter__()

    def values(self) -> Iterable:
        for key in self.__iter__():
            yield self.__getattribute__(key)

    def items(self) -> Iterable:
        for key in self.__iter__():
            yield key, self.__getattribute__(key)


# end copy


@define
class scatteringDataObj(gimmeItems):
    """
    Object that carries the necessary 1D data from each measurement file with (methods to get the) metadata.
    This will be used in the merging process.

    Note: Masks can be set, but are not automatically applied.
    Note2: Unit conversions are not implemented yet.
    """

    Q: np.ndarray = field(
        # default=np.array([0], dtype=float),
        validator=QChecker,
        eq=cmp_using(eq=np.array_equal),
        converter=np.array,
    )
    I: np.ndarray = field(
        # default=np.array([0], dtype=float),
        validator=IChecker,
        eq=cmp_using(eq=np.array_equal),
        converter=np.array,
    )
    ISigma: np.ndarray = field(
        # default=np.array([0], dtype=float),
        validator=validators.optional(IChecker),
        eq=cmp_using(eq=np.array_equal),
        converter=np.array,
    )
    wt: Optional[np.ndarray] = field(
        default=None,
        validator=validators.optional(IChecker),
        eq=cmp_using(eq=np.array_equal),
    )
    Mask: Optional[np.ndarray] = field(
        default=None,
        validator=validators.optional(IChecker),
        eq=cmp_using(eq=np.array_equal),
        # converter=np.array,
    )
    QSigma: Optional[np.ndarray] = field(
        default=None,
        validator=validators.optional(IChecker),
        eq=cmp_using(eq=np.array_equal),
    )
    configuration: int = field(
        default=-1, validator=validators.instance_of(int), converter=int
    )
    configurations: List[
        int
    ] = field(  # in case it's a merged dataset containing multiple configurations
        default=Factory(list), validator=validators.instance_of(list)
    )
    filename: Optional[Path] = field(
        default=None,
        validator=validators.optional(validators.instance_of(Path)),
    )
    sampleName: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        converter=str,
    )
    sampleOwner: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        converter=str,
    )
    IUnits: str = field(
        default="1/(m sr)",
        validator=validators.instance_of(str),
        converter=str,
    )
    Qunits: str = field(
        default="1/nm",
        validator=validators.instance_of(str),
        converter=str,
    )

    def __attrs_post_init__(self):
        """Make sure we set a mask, all zeros if not specified"""
        if self.Mask is None:
            self.Mask = np.zeros(self.Q.shape, dtype=bool)
        if (
            self.QSigma is None
        ):  # I expect this to be most of the cases, in which it gets set to qeMin*Q
            self.QSigma = np.zeros(self.Q.shape, dtype=float)

    def dataLen(self) -> int:
        """Return the length of the data array"""
        return len(self.Q)

    def updateScaledMaskedValues(
        self, maskArray: Optional[np.ndarray] = None, scaling: float = 1.0
    ) -> None:
        """
        Updates the attributes of the class with some scaling and mask (if present)
        """
        if maskArray is None:
            self.Q = self.Q.copy()
            self.QSigma = self.QSigma.copy()
            self.I = self.I * scaling
            self.ISigma = self.ISigma * scaling
            self.Mask = self.Mask
        elif isinstance(maskArray, np.ndarray):
            assert (
                maskArray.shape == self.Q.shape
            ), "Mask array supplied to scatteringDataObj.asPandas must conform to the data shape"
            self.Q = self.Q[~maskArray]
            self.QSigma = self.QSigma[~maskArray]
            self.I = self.I[~maskArray] * scaling
            self.ISigma = self.ISigma[~maskArray] * scaling
            self.Mask = self.Mask[~maskArray]
        else:
            assert False, "Not a valid mask"

    def qMin(self) -> float:
        """Return the minimum Q"""
        return self.Q.min()

    def qMax(self) -> float:
        """Return the maximum Q"""
        return self.Q.max()

    def qRange(self) -> List[float]:
        """Return the Q range"""
        return list([self.qMin(), self.qMax()])

    def returnMaskByQRange(
        self, qMin: Optional[float] = None, qMax: Optional[float] = None
    ) -> np.ndarray:
        if qMin is None:
            qMin = self.qMin()
        if qMax is None:
            qMax = self.qMax()
        return (self.Q < qMin) | (self.Q > qMax)

    # def updateMaskByQRange(self, maskByQRange: np.ndarray) -> None:
    #     """Updates the internal data mask with the q range mask. Not sure if useful"""
    #     self.Mask = np.array(self.Mask, dtype=bool) | maskByQRange
    #     return


@define
class rangeConfigObj(gimmeItems):
    """
    Defines a single range - for a single dataset that goes into the merge.
    """

    rangeId: int = field(
        default=-1,
        validator=validators.instance_of(int),
        converter=int,
    )
    scatteringData: Optional[scatteringDataObj] = field(
        default=None,
        validator=validators.optional(validators.instance_of(scatteringDataObj)),
    )
    qMinPreset: Optional[float] = field(
        default=None,
        validator=validators.optional(
            [validators.ge(0), validators.instance_of(float)]
        ),
    )
    qMaxPreset: Optional[float] = field(
        default=None,
        validator=validators.optional(
            [validators.ge(0), validators.instance_of(float)]
        ),
    )
    autoscaleToConfig: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        # converter=int,
    )
    scale: float = field(
        default=1.0,
        validator=[validators.ge(0), validators.instance_of(float)],
        # converter=float,
    )
    findByConfig: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        # converter=int,
    )


@define
class outputRangeObj(gimmeItems):
    """
    Config carrying the settings for the spacing of the output bins within
    a given range. Multiple ranges can be specified so that you can set
    different binning for e.g. SAXS ranges and WAXS ranges.
    """

    outputRangeId: int = field(
        default=0, validator=validators.instance_of(int), converter=int
    )
    qCrossover: float = field(
        default=np.inf, validator=validators.instance_of(float), converter=float
    )
    nbins: int = field(
        default=200, validator=validators.instance_of(int), converter=int
    )
    QScaling: str = field(
        default="log",
        validator=[validators.instance_of(str), scalingChecker],
        converter=str,
    )


@define
class HDFDefaultsObj(gimmeItems):
    """In case the preset HDF5 paths are empty, defaults can be used for some non-critical items"""

    sampleName: str = field(
        default="", validator=validators.instance_of(str), converter=str
    )
    sampleOwner: str = field(
        default="", validator=validators.instance_of(str), converter=str
    )
    configuration: int = field(
        default=1, validator=validators.instance_of(int), converter=int
    )
    IUnits: str = field(
        default="1/(m sr)", validator=validators.instance_of(str), converter=str
    )
    QUnits: str = field(
        default="1/nm", validator=validators.instance_of(str), converter=str
    )


@define
class HDFPathsObj(gimmeItems):
    """
    Config carrying the HDF5 path locations for reading datafiles.
    """

    Q: str = field(
        default="/entry/result/Q", validator=validators.instance_of(str), converter=str
    )
    I: str = field(
        default="/entry/result/I", validator=validators.instance_of(str), converter=str
    )
    ISigma: str = field(
        default="/entry/result/ISigma",
        validator=validators.instance_of(str),
        converter=str,
    )
    QSigma: Optional[str] = field(
        default="/entry/result/QSigma",
        validator=validators.instance_of(str),
        converter=str,
    )
    sampleName: str = field(
        default="/entry1/sample/name",
        validator=validators.instance_of(str),
        converter=str,
    )
    sampleOwner: str = field(
        default="/entry1/sample/sampleowner",
        validator=validators.instance_of(str),
        converter=str,
    )
    configuration: str = field(
        default="/entry1/instrument/confuration",
        validator=validators.instance_of(str),
        converter=str,
    )


@define
class readConfigObj(gimmeItems):
    """
    Object that carries information on how to read the datafiles
    """

    hdfPaths: HDFPathsObj = field(
        default=Factory(HDFPathsObj),
        validator=validators.instance_of(HDFPathsObj),
    )

    hdfDefaults: HDFDefaultsObj = field(
        default=Factory(HDFDefaultsObj),
        validator=validators.instance_of(HDFDefaultsObj),
    )


@define
class mergeConfigObj(gimmeItems):
    """
    Object that carries the merge configuration information with read methods
    This will be used to configure the merging process
    """

    filename: Optional[Path] = field(
        default=None,
        validator=validators.optional(validators.instance_of(Path)),
        converter=Path,
    )

    eMin: float = field(
        default=0.01,
        validator=[validators.ge(0), validators.instance_of(float)],
        converter=float,
    )
    qeMin: float = field(
        default=0.01,
        validator=[validators.ge(0), validators.instance_of(float)],
        converter=float,
    )
    outputRanges: Optional[List[outputRangeObj]] = field(
        default=None,
        validator=validators.optional(validators.instance_of(list)),
    )
    ranges: Optional[List[rangeConfigObj]] = field(
        default=None,
        validator=validators.optional(validators.instance_of(list)),
    )
    outputIUnits: str = field(
        default="1/(m sr)",
        validator=validators.instance_of(str),
        converter=str,
    )
    outputQUnits: str = field(
        default="1/nm",
        validator=validators.instance_of(str),
        converter=str,
    )
    IEWeighting: bool = field(
        default=True,
        validator=validators.instance_of(bool),
        converter=bool,
    )
    maskMasked: bool = field(
        default=True,
        validator=validators.instance_of(bool),
        converter=bool,
    )
    maskSingles: bool = field(
        default=False,
        validator=validators.instance_of(bool),
        converter=bool,
    )

    def maxRange(self) -> int:
        if self.ranges is None:
            return -1
        return len(self.ranges)


@define
class supplementaryDataObj(gimmeItems):
    """
    sample information
    """

    # supplementary information
    configurations: List[
        int
    ] = field(  # in case it's a merged dataset containing multiple configurations
        default=Factory(list), validator=validators.instance_of(list)
    )
    sampleName: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        converter=str,
    )
    sampleOwner: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        converter=str,
    )


@define
class mergedDataObj(gimmeItems):
    """
    Object that carries the merged/binned data
    This will be written / construted in the merging process, since it's an iterative procedure, all are optional.

    """

    # DIAB: datapoints in a bin
    Q: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(QChecker)
    )  # scattering vector
    I: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # scattering intensity or cross-section
    IStd: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # standard deviation of the DIAB
    ISEM: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # standard error on the mean of the DIAB
    ISEMw: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # weighted SEM
    IEPropagated: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # propagated uncertanties from the original DIAB
    ISigma: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # final overall uncertainty estimator
    QStd: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # Standard deviation in Q of the DIAB
    QSEM: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # standard error on the mean of the same
    QSigma: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # final overall Q uncertainty estimator
    Mask: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # any masked bins (probably only for empty bins)
    Singles: Optional[np.ndarray] = field(
        default=None, validator=validators.optional(IChecker)
    )  # bin value computed from a single DIAB only

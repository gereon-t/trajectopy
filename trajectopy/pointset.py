import copy
from typing import Union

import numpy as np
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection


class PointSetError(Exception):
    pass


class PointSet:
    """Class representing a point set


    The pointset class can hold one or multiple 3d positions
    with a corresponding EPSG code for datum information.
    When creating a pointset instance, a transformation pipeline
    is created based on the passed points, which can transform
    the points into a local system tangent to the ellipsoid (grs80).
    Such a local datum is represented within this class with an
    EPSG code of 0 and is mainly suitable for local calculations
    within the pointset.

    All transformations, including the local transformation, are
    carried out using pyproj, a python toolbox for projections and
    transformations.
    """

    def __init__(
        self,
        xyz: Union[np.ndarray, list],
        epsg: int = 0,
        local_transformer: Transformer = None,
        init_local_transformer: bool = True,
        epsg_local_cart: int = 4936,
        epsg_local_geod: int = 4937,
    ) -> None:
        """Initialize PointSet and create local transformer

        If a pointset is initialized directly with an EPSG code of 0,
        such a local transformer cannot be constructed, and a
        transformation into other EPSG codes is therefore not possible.
        Use this setting, if you dont have any information about the
        datum of the passed points.
        However, if the local transformer is already known, it can be
        provided during the initialization of the pointset using the
        local_transformer variable.

        Args:
            xyz (np.ndarray): 1- / 2- dimensional numpy array
                              containing the coordinated of the
                              input positions
            epsg (int, optional): EPSG code of the datum of the input
                                  positions. Defaults to 0.
            local_transformer (Transformer, optional): pyproj transformer
                                                       that describes the
                                                       transformation to
                                                       a local coordinate
                                                       system.
                                                       Defaults to None.
            init_local_transformer (bool, optional): Specifies if a local
                                                     transformer should be
                                                     initialized.
                                                     Defaults to True.
            epsg_local_cart (int, optional): EPSG code of the earth-centered
                                             datum that is used to construct
                                             the local transformation pipeline.
                                             In a first step, the coordinates
                                             are transformed to this coordinate
                                             frame. In this coordinate frame
                                             they are reduced by their mean
                                             position.
                                             Defaults to 4936.
            epsg_local_geod (int, optional): In the final step of the local
                                             transformation pipeline, the
                                             positions reduced by their mean
                                             are rotated into a local system
                                             tangent to the ellipsoid.
                                             The ellipsoid is defined by this
                                             parameter using an EPSG code.
                                             Both EPSG codes, epsg_local_cart
                                             and epsg_local_geod should refer
                                             to the same datum (here ETRS89).
                                             Defaults to 4937.

        Raises:
            PointSetError: Gets raised, if input xyz is not a numpy array
        """

        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz)

        if (len(xyz.shape) == 1 and len(xyz) != 3) or (len(xyz.shape) == 2 and xyz.shape[1] != 3):
            raise PointSetError("Input must be nx3 or 3xNone array!")

        self.xyz = xyz
        self.epsg = epsg
        self.local_transformer = local_transformer
        self.epsg_local_geod = epsg_local_geod
        self.epsg_local_cart = epsg_local_cart
        self.pipeline_str = ""

        if self.epsg == 0:
            return

        # Create CRS and local transformer
        if local_transformer or not init_local_transformer:
            return

        self.build_local_transformer()

    def build_local_transformer(self):
        # geodetic
        llh_geod = self.to_epsg(target_epsg=self.epsg_local_geod, inplace=False)

        # mean latitude and longitude
        mean_lat = np.deg2rad(np.mean(llh_geod.x))
        mean_lon = np.deg2rad(np.mean(llh_geod.y))

        # geocentric cartesian
        xyz_cart = self.to_epsg(target_epsg=self.epsg_local_cart, inplace=False)

        # mean position
        mean_pos = np.mean(xyz_cart.xyz, axis=0)

        # transformation matrix
        t_local = np.array(
            [
                [-np.sin(mean_lon), np.cos(mean_lon), 0],
                [
                    -np.sin(mean_lat) * np.cos(mean_lon),
                    -np.sin(mean_lat) * np.sin(mean_lon),
                    np.cos(mean_lat),
                ],
                [
                    np.cos(mean_lat) * np.cos(mean_lon),
                    np.cos(mean_lat) * np.sin(mean_lon),
                    np.sin(mean_lat),
                ],
            ]
        )

        # define local transformation
        self.pipeline_str = (
            f"+proj=pipeline "
            f"+step +proj=affine "
            f"+xoff={-mean_pos[0]} +yoff={-mean_pos[1]} +zoff={-mean_pos[2]} "
            f"+step +proj=affine "
            f"+s11={t_local[0,0]} +s12={t_local[0,1]} +s13={t_local[0,2]} "
            f"+s21={t_local[1,0]} +s22={t_local[1,1]} +s23={t_local[1,2]} "
            f"+s31={t_local[2,0]} +s32={t_local[2,1]} +s33={t_local[2,2]}"
        )

        self.local_transformer = Transformer.from_pipeline(proj_pipeline=self.pipeline_str)

    def __str__(self) -> str:
        return f"EPSG: {self.epsg}\nCoordinates:\n{str(self.xyz)}"

    def __len__(self) -> int:
        """Returns number of points"""
        return len(self.xyz)

    def copy(self) -> "PointSet":
        """Deep copy"""
        return copy.deepcopy(self)

    def __get_column(self, idx: int) -> Union[int, float, np.ndarray]:
        """Internal method to extract a column from xyz

        This method will return one column specified by idx from the
        xyz array.
        It distinguishes between the number of points and returns
        either a single float or a 1-d numpy array.

        Args:
            idx (int): index of the desired column

        Returns:
            Union[int, float, np.ndarray]: Value / array of that column
        """
        return self.xyz[:, idx] if len(self.xyz) > 1 else self.xyz[0][idx]

    def __set_column(self, v: Union[int, float, np.ndarray], idx: int) -> None:
        """Internal method to change a column of xyz

        This method will set a column of xyz to v.

        Args:
            v (Union[int, float, np.ndarray]): Value(s) with which the
                                               column is to be filled.
                                               Either a single value or
                                               an array of exactly the
                                               same length as xyz.
            idx (int): column index in xyz
        """
        self.xyz[:, idx] = v

    @property
    def xyz(self) -> np.ndarray:
        """xyz property returning the points within the pointset

        Returns:
            np.ndarray: 2-dimensional numpy array
        """
        return self.__xyz

    @xyz.setter
    def xyz(self, xyz: np.ndarray):
        """Sets xyz array

        Takes care that xyz is always a 2-dimensional numpy array

        Args:
            xyz (np.ndarray): points used to replace the current points

        Raises:
            PointSetError: Gets raised, if input xyz is not a numpy array
        """
        if not isinstance(xyz, np.ndarray):
            raise PointSetError("Input must be numpy array!")
        self.__xyz = xyz if xyz.ndim > 1 else xyz[None, :]

    @property
    def x(self) -> Union[int, float, np.ndarray]:
        """x property

        The x/y/z properties will either return a one-dimensional numpy
        array or a single float / int depending on whether there is
        more than one point in the pointset

        Returns:
            Union[int, float, np.ndarray]
        """
        return self.__get_column(idx=0)

    @property
    def y(self) -> Union[int, float, np.ndarray]:
        """y property

        The x/y/z properties will either return a one-dimensional numpy
        array or a single float / int depending on whether there is
        more than one point in the pointset

        Returns:
            Union[int, float, np.ndarray]
        """
        return self.__get_column(idx=1)

    @property
    def z(self) -> Union[int, float, np.ndarray]:
        """z property

        The x/y/z properties will either return a one-dimensional numpy
        array or a single float / int depending on whether there is
        more than one point in the pointset

        Returns:
            Union[int, float, np.ndarray]
        """
        return self.__get_column(idx=2)

    @x.setter
    def x(self, x: Union[int, float, np.ndarray]) -> None:
        """x setter

        This method will set the x value(s) to some input value(s)

        Args:
            x (Union[int, float, np.ndarray]): Either a single value or
                                               an array of exactly the
                                               same length as xyz.
        """
        self.__set_column(v=x, idx=0)

    @y.setter
    def y(self, y: Union[int, float, np.ndarray]) -> None:
        """y setter

        This method will set the y value(s) to some input value(s)

        Args:
            y (Union[int, float, np.ndarray]): Either a single value or
                                               an array of exactly the
                                               same length as xyz.
        """
        self.__set_column(v=y, idx=1)

    @z.setter
    def z(self, z: Union[int, float, np.ndarray]) -> None:
        """z setter

        This method will set the z value(s) to some input value(s)

        Args:
            z (Union[int, float, np.ndarray]): Either a single value or
                                               an array of exactly the
                                               same length as xyz.
        """
        self.__set_column(v=z, idx=2)

    @property
    def crs(self) -> CRS:
        """Coordinate Reference System

        Returns:
            CRS: pyproj CRS object that represents the current
            coordinate system
        """
        return None if self.epsg == 0 else CRS.from_epsg(code=self.epsg)

    def to_epsg(self, target_epsg: int, inplace: bool = True) -> "PointSet":
        """Performs a coordinate transformation using a target crs

        This method will construct the required pyproj transformer and
        applies it in order to transform the pointset to the target
        ESPG code.

        Args:
            target_epsg (int): EPSG code of target CRS
            inplace (bool, optional): perform transformation in place.
                                      Defaults to True.

        Raises:

            PointSetError: Gets raised if it is not possible to recover
                           from a local datum since local transformer
                           is unknown

        Returns:
            PointSet: transformed pointset
        """
        pointset = self if inplace else self.copy()

        # 0) already there
        if pointset.epsg == target_epsg:
            return pointset

        # currently in unknown frame
        if pointset.epsg == 0 and pointset.local_transformer is None:
            raise PointSetError("Unable to recover from local frame since definition is unknown!")

        # 1) from non-local to local
        if target_epsg == 0:
            # geocentric cartesian (ITRF2014)
            pointset.to_epsg(target_epsg=pointset.epsg_local_cart)

            # apply local transformer
            x, y, z = pointset.local_transformer.transform(xx=pointset.x, yy=pointset.y, zz=pointset.z)

            pointset.xyz = np.c_[x, y, z]
            pointset.epsg = target_epsg

            return pointset

        # 2) from local to non-local (intermediate step)
        if pointset.epsg == 0:
            # undo local transformation
            x, y, z = pointset.local_transformer.transform(
                xx=pointset.x,
                yy=pointset.y,
                zz=pointset.z,
                direction=TransformDirection.INVERSE,
            )
            xyz = np.c_[x, y, z]
            crs = CRS.from_epsg(pointset.epsg_local_cart)
        else:
            xyz = pointset.xyz
            crs = pointset.crs

        # 3) from non-local to non-local
        transformer = Transformer.from_crs(crs, CRS.from_epsg(code=target_epsg))

        x, y, z = transformer.transform(xx=xyz[:, 0], yy=xyz[:, 1], zz=xyz[:, 2])

        pointset.xyz = np.c_[x, y, z]
        pointset.epsg = target_epsg

        return pointset

    def to_local(self, inplace: bool = True) -> "PointSet":
        """Transform pointset to a local frame tangential to the
           (grs80) ellipsoid

        This is equivalent to an transformation to an EPSG of 0

        Args:
            inplace (bool, optional): perform transformation in place.
                                      Defaults to True.

        Returns:
            PointSet: 2-dimensional PointSet containing xyz of the
                        transformed points
        """
        return self.to_epsg(target_epsg=0, inplace=inplace)

    def mean(self, inplace: bool = False) -> "PointSet":
        """Computes the mean of all points within the pointset

        Args:
            inplace (bool, optional): if true, the pointset gets
                                      replaced by a single mean
                                      position. Defaults to False.

        Returns:
            PointSet: Contains the mean position
        """
        mean_xyz = np.mean(self.xyz, axis=0)

        if inplace:
            self.xyz = mean_xyz
        return PointSet(xyz=mean_xyz, epsg=self.epsg, local_transformer=self.local_transformer)

    def round_to(self, prec: float) -> "PointSet":
        """Rounds all points to a given precision

        Args:
            prec (float): desired rounding precision

        Returns:
            PointSet: Contains the rounded positions
        """
        rounded = self.copy()
        rounded.xyz = np.round(rounded.xyz / prec) * prec
        return rounded

    def __add__(self, other: "PointSet") -> "PointSet":
        """Plus operator"""
        xyz_add = self.xyz + other.xyz
        return PointSet(xyz=xyz_add, epsg=self.epsg, local_transformer=self.local_transformer)

    def __sub__(self, other: "PointSet") -> "PointSet":
        """Minus operator"""
        xyz_sub = self.xyz - other.xyz
        return PointSet(xyz=xyz_sub, epsg=self.epsg, local_transformer=self.local_transformer)

    def __key(self):
        """Key generation used for hash generation"""
        return (
            self.epsg,
            self.xyz[:, 0] @ self.xyz[:, 0],
            self.xyz[:, 1] @ self.xyz[:, 1],
            self.xyz[:, 2] @ self.xyz[:, 2],
            self.pipeline_str,
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        """Equality"""
        if isinstance(other, PointSet):
            return self.__key() == other.__key()
        raise NotImplementedError(f"Cannot compare {type(self)} with {type(other)}!")

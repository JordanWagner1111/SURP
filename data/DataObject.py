import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS, FITSFixedWarning
from pkg_resources import resource_filename

warnings.filterwarnings("ignore", category=FITSFixedWarning)


class DataObject:

    def get_atomic_map_hdu(self):
        """
        Returns: atomic map Header Data Unit (hdu) and World Coordinate System (wcs)

        """
        filename = resource_filename(__name__, "FITS/atomic.fits")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        hdu = fits.open(filename)[0]

        return hdu

    # ===================================================================================
    def get_co_map_hdu(self):
        """
        Returns: co map Header Data Unit (hdu) and World Coordinate System (wcs)

        """
        filename = resource_filename(__name__, "FITS/ico.regrid.smooth.fits")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        hdu = fits.open(filename)[0]

        return hdu

    # ===================================================================================
    def get_atomic_map_wcs(self):
        wcs = WCS(self.get_atomic_map_hdu().header)

        return wcs

    # ===================================================================================
    def get_co_map_wcs(self):
        wcs = WCS(self.get_co_map_hdu().header)

        return wcs

    # ===================================================================================
    def get_rsg_coord_df(self):
        table = pd.read_csv("data/tsv/rsg_df.tsv", sep="\t")

        return table

    # ===================================================================================
    def get_wr_coord_df(self):
        table = pd.read_csv("data/tsv/wr_df.tsv", sep="\t")

        return table

    # ===================================================================================
    def get_star_pixel_array(self, star_type: str):
        if star_type == "RSGs":
            coord_df = self.get_rsg_coord_df()
            RA = coord_df["RAJ2000"]
            Dec = coord_df["DEJ2000"]
            sky_coord_unit = ("hourangle", "deg")
        elif star_type == "WRs":
            coord_df = self.get_wr_coord_df()
            RA = coord_df["RA"]
            Dec = coord_df["Dec"]
            sky_coord_unit = ("deg", "deg")
        else:
            raise ValueError(
                "Expected 'RSGs' or 'WRs' for star_type, got '{}'".format(star_type)
            )

        sky_coord = SkyCoord(RA, Dec, unit=sky_coord_unit)
        x, y = self.get_atomic_map_wcs().world_to_pixel(sky_coord)
        i_component = np.asarray(x, dtype="int")
        j_component = np.asarray(y, dtype="int")

        return i_component, j_component

    # ===================================================================================
    def stars_on_gas_map_array(self, hdu, star_type: str, map_type: str):
        """
        Filters and returns the pixel coordinates of stars on the gas map. It computes
        the pixel coordinates of the specified star type using the World Coordinate System (WCS)
        of the selected gas map. The function then filters out any coordinates that fall
        outside the boundaries of the gas map.

        Args:
            hdu (astropy.io.fits.PrimaryHDU): The primary HDU containing the gas map data.
            star_type (str): The type of stars to be processed, either "RSGs" or "WRs".

        Returns:
            tuple: Two numpy arrays, `i_component` and `j_component`, containing the valid
                pixel coordinates of the specified star type on the gas map.

        Raises:
            ValueError: If `star_type` is not "RSGs" or "WRs".
        """
        if star_type == "RSGs":
            i_component, j_component = self.get_star_pixel_array(star_type="RSGs")
        elif star_type == "WRs":
            i_component, j_component = self.get_star_pixel_array(star_type="WRs")
        else:
            raise ValueError(
                "Expected 'RSGs' or 'WRs' for star_type, got '{}'".format(star_type)
            )

        i_component = np.array(i_component)
        j_component = np.array(j_component)

        # Create a boolean mask for valid coordinates
        if map_type == "Atomic":
            valid_mask = (i_component < hdu.data.shape[0]) & (
                j_component < hdu.data.shape[1]
            )
        elif map_type == "CO":
            valid_mask = (i_component < hdu.data[0, 0].shape[0]) & (
                j_component < hdu.data[0, 0].shape[1]
            )
        else:
            raise ValueError(
                "Expected 'Atomic' or 'CO' for map_type, got '{}'".format(map_type)
            )

        # Use the mask to filter the coordinates
        i_component = i_component[valid_mask]
        j_component = j_component[valid_mask]

        return i_component, j_component

    # ===================================================================================

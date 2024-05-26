import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import skycoord_to_pixel
from regions import PolygonSkyRegion, Regions

from data.DataObject import DataObject

warnings.filterwarnings("ignore", category=FITSFixedWarning)


class Region(DataObject):

    # ===================================================================================
    def get_co_map_data(self):
        comap = fits.open("data/FITS/ico.regrid.smooth.fits")
        comapdata = comap[0].data[0, 0]

        return comapdata

    def get_co_map_header(self):
        comap = fits.open("data/FITS/ico.regrid.smooth.fits")
        comapheader = comap[0].header

        return comapheader

    # ===================================================================================
    def star_coords_inside_gas_region(self, wcs, star_type):
        """
        Returns: 2-D np.array of pixel coordinates
        """
        if star_type == "RSGs":
            star_coords = SkyCoord(
                self.get_rsg_coord_df()["RAJ2000"],
                self.get_rsg_coord_df()["DEJ2000"],
                unit=("hourangle", "deg"),
            )
        elif star_type == "WRs":
            star_coords = SkyCoord(
                self.get_wr_coord_df()["RA"],
                self.get_wr_coord_df()["Dec"],
                unit=("deg", "deg"),
            )
        elif star_type == "SNRs":
            star_coords = SkyCoord(
                self.get_snr_coord_df()["RAJ2000"],
                self.get_snr_coord_df()["DEJ2000"],
                unit=("hourangle", "deg"),
            )
        else:
            raise ValueError(
                f"Expected 'RSGs', 'WRs', or 'SNRs' for star_type, got '{star_type}'"
            )
        star_pix = np.asarray(skycoord_to_pixel(star_coords, wcs), dtype=int)
        comapdata = self.get_co_map_data()
        lencox, lencoy = comapdata.shape
        # Only keep RSG pixels that fall inside the CO map area
        rsgpix_insideco = np.array(
            [[i, j] for i, j in star_pix.T if ((0 <= j < lencox) and (0 <= i < lencoy))]
        ).T

        return rsgpix_insideco

    # ===================================================================================
    def get_sky_region(self):
        sky_region = Regions.read("data/reg/ACA_region.reg", format="ds9")

        return sky_region

    # ===================================================================================
    def draw_gas_map_inside_sky_region(self, wcs):
        vertices = SkyCoord(
            [
                (23.6754846, 30.8179815),
                (23.4167034, 30.8648173),
                (23.3110882, 30.5247383),
                (23.5741699, 30.479219),
            ],
            unit="deg",
            frame="fk5",
        )
        region = PolygonSkyRegion(vertices=vertices)
        # region = RectangleSkyRegion(center=center_sky, width=778.045*u.arcsec, height=885.116*u.arcsec, angle=13.715856*u.deg)
        pixel_region = region.to_pixel(wcs)

        return pixel_region

    # ===================================================================================
    def star_coords_inside_aca_region(self, wcs, star_type):
        if star_type == "RSGs":
            star_coords = SkyCoord(
                self.get_rsg_coord_df()["RAJ2000"],
                self.get_rsg_coord_df()["DEJ2000"],
                unit=("hourangle", "deg"),
            )
        elif star_type == "WRs":
            star_coords = SkyCoord(
                self.get_wr_coord_df()["RA"],
                self.get_wr_coord_df()["Dec"],
                unit=("deg", "deg"),
            )
        elif star_type == "SNRs":
            star_coords = SkyCoord(
                self.get_snr_coord_df()["RAJ2000"],
                self.get_snr_coord_df()["DEJ2000"],
                unit=("hourangle", "deg"),
            )
        else:
            raise ValueError(
                f"Expected 'RSGs', 'WRs', or 'SNRs' for star_type, got '{star_type}'"
            )

        arr_is_inside = self.get_sky_region()[0].contains(star_coords, wcs)
        rsgs_inside_box = star_coords[arr_is_inside]

        return rsgs_inside_box

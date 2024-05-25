import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from pkg_resources import resource_filename


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

    def get_atomic_map_wcs(self):
        wcs = WCS(self.get_atomic_map_hdu().header)

        return wcs

    def get_rsg_coord_table(self):
        table = pd.read_csv("data/tsv/asu.tsv", sep="\t")

        return table

    def rsg_pixel_array(self):
        RA = self.get_rsg_coord_table()["RAJ2000"]
        Dec = self.get_rsg_coord_table()["DEJ2000"]

        sky_hourangle = SkyCoord(RA, Dec, unit=("hourangle", "deg"))
        sky_deg = SkyCoord(sky_hourangle.ra.deg, sky_hourangle.dec.deg, unit=("deg"))
        x, y = self.get_atomic_map_wcs().world_to_pixel(sky_deg)
        i_component = np.asarray(x, dtype="int")
        j_component = np.asarray(y, dtype="int")

        return i_component, j_component

    def rsgs_atomic_map_array(self, hdu):
        i_component, j_component = self.rsg_pixel_array()

        filterout = []
        for idx in range(len(i_component)):
            filterout.append(
                i_component[idx] >= hdu.data.shape[0]
                or j_component[idx] >= hdu.data.shape[1]
            )
        new_i_coordinate = []
        new_j_coordinate = []
        for idx in range(len(filterout)):
            if not filterout[idx]:
                new_i_coordinate.append(i_component[idx])
                new_j_coordinate.append(j_component[idx])
            else:
                pass
        i_component = np.array(new_i_coordinate)
        j_component = np.array(new_j_coordinate)

        return i_component, j_component

    def plot_rsg_cdf(
        self,
        hdu,
        stars_on_map_array,
        title: str,
        xlabel: str,
        ylabel: str,
        gas_label: str,
    ):
        plt.hist(
            stars_on_map_array,
            bins=np.logspace(0, 4, 1000),
            zorder=2,
            histtype="step",
            cumulative=True,
            density=True,
            label="RSGs",
        )
        plt.hist(
            hdu.data.flatten(),
            bins=np.logspace(0, 4, 1000),
            histtype="step",
            cumulative=True,
            density=True,
            label=gas_label,
        )
        plt.xscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(30, 2000)
        plt.legend(loc="upper left")

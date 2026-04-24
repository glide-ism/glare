import calendar
import datetime
from pathlib import Path

import cupy as cp
import numpy as np
import pytz
import xarray as xr
from pysolar.solar import get_altitude, get_azimuth


class SolarPotential:
    """
    GPU-accelerated solar potential calculator using CUDA ray tracing.

    For a given DEM and geographic location, computes per-pixel solar
    potential accounting for terrain self-shadowing and incidence angle.
    """

    def __init__(
        self,
        dem: xr.Dataset,
        latitude: float,
        longitude: float,
        kernel_path: str = None,
        grid_resolution: float = 90.0,
        step_size: float = 1.0,
        timezone: str = "America/Anchorage",
    ):
        """
        Parameters
        ----------
        dem : xr.Dataset
            Dataset with an 'elevation' variable on an (y, x) grid.
        latitude : float
            Representative latitude for solar angle calculations.
        longitude : float
            Representative longitude for solar angle calculations.
        kernel_path : str, optional
            Path to the azimuth_trace CUDA kernel source file. If not provided,
            defaults to the bundled kernel in glare/cuda/azimuth_trace.cu.
        grid_resolution : float
            Grid spacing in meters, used for gradient and zenith calculations.
        step_size : float
            Ray marching step size passed to the CUDA kernel.
        timezone : str
            Timezone string for solar position calculations.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.grid_resolution = cp.float32(grid_resolution)
        self.step_size = cp.float32(step_size)
        self.timezone = pytz.timezone(timezone)

        self.z = cp.array(dem.elevation.values, dtype=cp.float32)
        self.ny, self.nx = self.z.shape

        dZdy, dZdx = cp.gradient(self.z, grid_resolution)
        self.dZdx = dZdx
        self.dZdy = -dZdy

        # Resolve kernel path: use bundled kernel if not provided
        if kernel_path is None:
            module_dir = Path(__file__).parent
            kernel_path = module_dir / "cuda" / "azimuth_trace.cu"
        else:
            kernel_path = Path(kernel_path)

        # Load CUDA kernel
        with open(kernel_path, "r") as f:
            kernel_code = f.read()
        kernels = cp.RawModule(code=kernel_code)
        self.kernel = kernels.get_function("azimuth_trace")
        self.block_size = (16, 16)
        self.grid_size = (self.nx // 16 + 1, self.ny // 16 + 1)

    def _run_shadow_kernel(self, azimuth_deg: float):
        """
        Run the ray tracing kernel for a given solar azimuth.

        Returns
        -------
        max_zenith : cp.ndarray
            Maximum terrain zenith angle along each ray, shape (ny, nx).
        """
        max_zenith = cp.zeros(self.z.shape, dtype=cp.float32)
        max_j = cp.zeros(self.z.shape, dtype=cp.uint32)
        max_i = cp.zeros(self.z.shape, dtype=cp.uint32)

        j_basis = cp.float32(np.sin(np.deg2rad(azimuth_deg)))
        i_basis = -cp.float32(np.cos(np.deg2rad(azimuth_deg)))

        self.kernel(
            self.grid_size,
            self.block_size,
            (max_zenith, max_j, max_i, self.z, j_basis, i_basis,
             self.step_size, self.nx, self.ny),
        )
        return max_zenith

    def compute_zenith_deg(self, azimuth_deg: float) -> cp.ndarray:
        """
        Terrain horizon zenith angle in degrees for the given solar azimuth.

        Parameters
        ----------
        azimuth_deg : float
            Solar azimuth angle in degrees.

        Returns
        -------
        cp.ndarray, shape (ny, nx)
        """
        max_zenith = self._run_shadow_kernel(azimuth_deg)
        return cp.rad2deg(cp.arctan(max_zenith / self.grid_resolution))

    def compute_shadow_mask(self, altitude_deg: float, azimuth_deg: float) -> cp.ndarray:
        """
        Soft shadow mask based on solar altitude vs terrain horizon zenith.

        Parameters
        ----------
        altitude_deg : float
            Solar altitude angle in degrees.
        azimuth_deg : float
            Solar azimuth angle in degrees.

        Returns
        -------
        cp.ndarray, shape (ny, nx), values in [0, 1]
        """
        zenith_deg = self.compute_zenith_deg(azimuth_deg)
        z_i = altitude_deg - zenith_deg
        return 1.0 / (1.0 + cp.exp(-z_i / 0.1))

    def compute_incidence(self, altitude_deg: float, azimuth_deg: float) -> cp.ndarray:
        """
        Cosine of the solar incidence angle on the terrain surface.
        Negative values (back-facing slopes) are clamped to zero.

        Parameters
        ----------
        altitude_deg : float
            Solar altitude angle in degrees.
        azimuth_deg : float
            Solar azimuth angle in degrees.

        Returns
        -------
        cp.ndarray, shape (ny, nx), values in [0, 1]
        """
        sin_phi = cp.sin(cp.deg2rad(cp.float32(azimuth_deg)))
        cos_phi = cp.cos(cp.deg2rad(cp.float32(azimuth_deg)))
        sin_alpha = cp.sin(cp.deg2rad(cp.float32(altitude_deg)))
        cos_alpha = cp.cos(cp.deg2rad(cp.float32(altitude_deg)))

        incidence = (
            -self.dZdx * sin_phi * cos_alpha
            - self.dZdy * cos_phi * cos_alpha
            + sin_alpha
        ) / cp.sqrt(self.dZdx ** 2 + self.dZdy ** 2 + 1)

        incidence = cp.maximum(incidence, 0.0)
        return incidence

    def compute_accumulated_sunlight_hours(
        self, year: int, month: int
    ) -> cp.ndarray:
        """
        Mean daily accumulated sunlight hours for each pixel over a month.

        Parameters
        ----------
        year : int
        month : int
            1-indexed month number.

        Returns
        -------
        cp.ndarray, shape (ny, nx)
        """
        num_days = calendar.monthrange(year, month)[1]
        accumulated = cp.zeros(self.z.shape, dtype=cp.float32)

        for day in range(1, num_days + 1):
            for hour in range(24):
                date = datetime.datetime(
                    year, month, day, hour, 0, 0, tzinfo=self.timezone
                )
                altitude = get_altitude(self.latitude, self.longitude, date)
                azimuth = get_azimuth(self.latitude, self.longitude, date)
                shadow_mask = self.compute_shadow_mask(altitude, azimuth)
                accumulated += shadow_mask

        accumulated /= num_days
        return accumulated

    def compute_monthly_solar_potential(self, year: int, normalized=True) -> cp.ndarray:
        """
        Mean daily solar potential for each pixel for all 12 months.

        Parameters
        ----------
        year : int
        normalized: bool - if True, then returns solar potential in dimensionless 
                                    scale representing intensity relative to continuous 
                                    orthogonal sunlight.  

        Returns
        -------
        cp.ndarray, shape (12, ny, nx)
        """
        result = cp.zeros((12, self.ny, self.nx), dtype=cp.float32)

        for month in range(1, 13):
            num_days = calendar.monthrange(year, month)[1]
            monthly = cp.zeros(self.z.shape, dtype=cp.float32)

            for day in range(1, num_days + 1):
                print(f"Calculating month {month}, day {day}")
                for hour in range(24):
                    date = datetime.datetime(
                        year, month, day, hour, 0, 0, tzinfo=self.timezone
                    )
                    altitude = get_altitude(self.latitude, self.longitude, date)
                    azimuth = get_azimuth(self.latitude, self.longitude, date)

                    shadow_mask = self.compute_shadow_mask(altitude, azimuth)
                    incidence = self.compute_incidence(altitude, azimuth)
                    monthly += incidence * shadow_mask

            if normalized:
                factor = num_days * 24
            else:
                factor = 1
            result[month - 1] = monthly / factor

        return result


    def compute_hourly_solar_potential(self, year: int, normalized=True) -> cp.ndarray:
        """
        Mean daily solar potential for each pixel averaged over each hour for all 12 months.

        Parameters
        ----------
        year : int
        normalized: bool - if True, then returns solar potential in dimensionless 
                                    scale representing intensity relative to continuous 
                                    orthogonal sunlight.  

        Returns
        -------
        cp.ndarray, shape (12, 24, ny, nx)
        """
        result = cp.zeros((12, 24, self.ny, self.nx), dtype=cp.float32)

        for month in range(1, 13):
            num_days = calendar.monthrange(year, month)[1]
            hourly = cp.zeros((24,self.ny,self.nx), dtype=cp.float32)

            for day in range(1, num_days + 1):
                print(f"Calculating month {month}, day {day}")
                for hour in range(24):
                    date = datetime.datetime(
                        year, month, day, hour, 0, 0, tzinfo=self.timezone
                    )
                    altitude = get_altitude(self.latitude, self.longitude, date)
                    azimuth = get_azimuth(self.latitude, self.longitude, date)

                    shadow_mask = self.compute_shadow_mask(altitude, azimuth)
                    incidence = self.compute_incidence(altitude, azimuth)
                    hourly[hour] += incidence * shadow_mask
            
            if normalized:
                factor = num_days
            else:
                factor = 1
            result[month - 1] = hourly / factor
        
        return result

    def compute_solar_potential_fourier_decomposition(self, year: int, normalized=True) -> cp.ndarray:
        """
        Carrying around 12 x 24 fields representing hourly insolation (which we need to 
        properly modulate melt over diurnal temperature scales) is expensive.  We instead
        compute the first three pixelwise components of the fourier series averaged over 
        each month, thus we only store 3 (12,ny,nx) fields instead of 24.  
        
        Parameters
        ----------
        year : int
        normalized: bool - if True, then returns solar potential in dimensionless 
                                    scale representing intensity relative to continuous 
                                    orthogonal sunlight.  

        Returns
        -------
        3 x cp.ndarray, shape (12, ny, nx)
        """
        result = self.compute_hourly_solar_potential(year=year,normalized=normalized)
        ny,nx = result.shape[2],result.shape[3]

        c0 = cp.zeros((12, ny, nx),dtype=cp.float32)
        cc = cp.zeros((12, ny, nx),dtype=cp.float32)
        cs = cp.zeros((12, ny, nx),dtype=cp.float32)
        
        for h in range(24):
            slc = result[:, h, :, :]
            c0 += slc
            cc += slc * cp.cos(2*np.pi*h/24)
            cs += slc * cp.sin(2*np.pi*h/24)
        c0 /= 24
        cc /= 12  # 2/24
        cs /= 12

        return c0,cc,cs

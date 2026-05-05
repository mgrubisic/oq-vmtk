Intensity Measure Calculation
##############################

The ``imcalculator`` class computes intensity measures (IMs) from an acceleration
time series. Supported IMs include response spectra, peak ground motion parameters
(PGA, PGV, PGD), spectral accelerations, average spectral acceleration (AvgSA),
Arias Intensity, Cumulative Absolute Velocity (CAV), significant duration, the
filtered incremental velocity (FIV3), and orientation-independent spectral
acceleration (RotDxx).

Minimal example
---------------

.. code-block:: python

   import numpy as np
   from openquake.vmtk.imcalculator import imcalculator

   # Acceleration time series (g) and time step (s)
   acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
   dt = 0.005

   im = imcalculator(acc, dt)
   pga, pgv, pgd = im.get_amplitude_ims()
   sa_T1 = im.get_sa(period=1.0)
   avg_sa = im.get_saavg(period=0.6)
   ai = im.get_arias_intensity()
   t595 = im.get_significant_duration()

   print(f"PGA = {pga:.3f} g, Sa(1.0) = {sa_T1:.3f} g, AI = {ai:.3f} m/s")

The reference values produced by this snippet are tabulated in
:doc:`validation` and asserted by ``test_imcalculator.py``. A full
end-to-end workflow is provided in the
``IntensityMeasureProcessing`` notebook
(see :doc:`demos` and :doc:`examples`).

.. toctree::

   imc/init
   imc/get_spectrum
   imc/get_sa
   imc/get_saavg
   imc/get_saavg_user_defined
   imc/get_velocity_displacement_history
   imc/get_amplitude_ims
   imc/get_arias_intensity
   imc/get_cav
   imc/get_significant_duration
   imc/get_FIV3
   imc/get_rotdxx

References
----------

1. Cordova, P.P., Deierlein, G.G., Mehanny, S.S., and Cornell, C.A. (2000). "Development of
   a two-parameter seismic intensity measure and probabilistic assessment procedure" in
   *Proceedings of the 2nd US–Japan Workshop on Performance-Based Earthquake Engineering
   Methodology for RC Building Structures* (Sapporo, Hokkaido, 2000).

2. Eads, L., Miranda, E., and Lignos, D.G. (2015). "Average spectral acceleration as an
   intensity measure for collapse risk assessment", *Earthquake Engineering and Structural Dynamics*,
   44, 2057–2073. doi: 10.1002/eqe.2575.

3. Kempton, J.J., and Stewart J.P. (2006). "Prediction equations for significant duration
   of earthquake ground motions considering site and near-source effects", *Earthquake Spectra*,
   22(4), 985-1013.

4. Arias, A. (1970). "A measure of earthquake intensity", in *Seismic Design for Nuclear
   Power Plants* (R.J. Hansen, ed.). The MIT Press, Cambridge, MA. 438-483.

5. Dávalos, H. and Miranda, E. (2019). "Filtered incremental velocity: A novel approach
   in intensity measures for seismic collapse estimation." *Earthquake Engineering & Structural Dynamics*,
   48(12), 1384–1405. DOI: 10.1002/eqe.3205.

6. Boore, D.M. (2010). "Orientation-independent, nongeometric-mean measures of seismic
   intensity from two horizontal components of motion." *Bulletin of the Seismological
   Society of America*, 100(4), 1830–1835. DOI: 10.1785/0120090400.

7. Trifunac, M.D. and Brady, A.G. (1975). "A study on the duration of strong earthquake
   ground motion." *Bulletin of the Seismological Society of America*, 65(3), 581–626.

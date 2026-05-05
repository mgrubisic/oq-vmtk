Visualisation
#############

The ``plotter`` class creates publication-quality plots for all ``oq-vmtk``
analysis outputs. It covers modal shape plots, cloud/MCA scatter plots, IDA fan
plots, MSA stripe plots, fragility curves, vulnerability functions, Storey Loss
Function outputs, and animated NRHA responses. All plots share a consistent style
(fonts, line widths, colour schemes) and can optionally be saved to disk.

Minimal example
---------------

.. code-block:: python

   import numpy as np
   from openquake.vmtk.plotter import plotter

   pl = plotter()

   # Plot a lognormal fragility curve
   intensities = np.geomspace(0.05, 3.0, 50)
   from scipy.stats import lognorm
   poes = lognorm(s=0.6, scale=0.35).cdf(intensities)

   pl.plot_fragility_from_mca(
       intensities=intensities,
       poes=[poes],
       labels=["DS1"],
       xlabel="Sa(T1) [g]",
       export_path=None,
   )

Every demo notebook under :doc:`demos` exercises the relevant ``plotter``
methods on real outputs.

.. toctree::

   plo/plot_modes
   plo/animate_spo
   plo/animate_cpo
   plo/animate_nrha
   plo/plot_mca_analysis
   plo/plot_ida_analysis
   plo/plot_msa_analysis
   plo/plot_fragility_from_mca
   plo/plot_fragility_from_ida
   plo/plot_fragility_from_msa
   plo/plot_slf_model
   plo/plot_vulnerability_function

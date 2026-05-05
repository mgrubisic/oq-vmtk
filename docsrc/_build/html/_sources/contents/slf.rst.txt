Storey Loss Function Generation
################################

The ``slf_generator`` class generates Storey Loss Functions (SLFs) from fragility,
consequence, and quantity data. SLFs link expected repair loss at a storey to the
Engineering Demand Parameter (EDP). A Monte Carlo approach is used to sample damage
states and associated repair costs across a user-defined inventory of damageable
components.

Minimal example
---------------

The example below uses the inventory CSVs bundled with the ``oq-vmtk``
demos at ``demos/StoreyLossFunctionGeneration/in/``.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from openquake.vmtk.slfgenerator import slfgenerator

   # Drift-sensitive nonstructural component inventory (FEMA P-58 derived)
   inventory_psd = pd.read_csv(
       "demos/StoreyLossFunctionGeneration/in/inventory_psd.csv"
   )

   model = slfgenerator(
       component_data=inventory_psd,
       edp="PSD",
       edp_range=np.linspace(0.001, 0.10, 100),
       grouping_flag=True,
       conversion=1.0,
       realizations=500,
       replacement_cost=1.0,
       regression=None,           # auto-select best fit
   )
   slf, cache = model.generate()

The full notebook (which also generates the acceleration-sensitive SLF and
plots both) is bundled at
``demos/StoreyLossFunctionGeneration/StoreyLossFunctionGeneration.ipynb`` —
see :doc:`demos` and :doc:`examples`.

.. toctree::

   slf/init
   slf/generate

References
----------

1. Ramirez, C. and Miranda, E., (2009) "Building-specific loss estimation methods and tools for simplified
   performance-based earthquake engineering", John A. Blume Earthquake Engineering Center, Stanford University.

2. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Story loss functions for seismic design and assessment:
   Development of tools and application," Earthquake Spectra 2021; 37(4): 2813-2839.
   DOI: 10.1177/87552930211023523.

3. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Development of a Python-Based Storey Loss Function
   Generator," COMPDYN 2021 - 8th International Conference on Computational Methods in Structural Dynamics
   and Earthquake Engineering, 2021. DOI: 10.7712/120121.8659.18567.

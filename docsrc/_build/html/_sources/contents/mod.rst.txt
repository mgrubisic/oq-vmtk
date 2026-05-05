Model Building and Analysis
###########################

The ``modeller`` class builds and analyses stick-and-mass MDOF structural models
using the OpenSees framework. Each storey is represented by a Pinching4 hysteretic
spring. Supported analysis types include gravity analysis, modal analysis, static
pushover (SPO), and nonlinear time-history analysis (NRHA).

Minimal example
---------------

.. code-block:: python

   import numpy as np
   from openquake.vmtk.modeller import modeller

   # 2-storey stick-and-mass model
   m = modeller(
       number_storeys=2,
       storey_heights=[3.0, 3.0],
       floor_masses=[120.0, 100.0],          # tonnes
       storey_drifts=np.array([              # (nst, n_capacity_points)
           [0.000, 0.005, 0.020, 0.060],
           [0.000, 0.005, 0.020, 0.060],
       ]),
       storey_forces=np.array([
           [0.0, 250.0, 320.0, 200.0],
           [0.0, 220.0, 280.0, 180.0],
       ]),
       degradation=True,
   )
   m.compile_model()
   m.do_gravity_analysis()
   periods, mode_shapes = m.do_modal_analysis(num_modes=2)

End-to-end workflows (pushover, NRHA, cloud analysis) are demonstrated in the
``ModelCompilation``, ``PushoverAnalysis`` and ``NonlinearTimeHistoryAnalysis``
notebooks — see :doc:`demos` and :doc:`examples`.

.. toctree::

   mod/init
   mod/compile_model
   mod/plot_model
   mod/do_gravity_analysis
   mod/do_modal_analysis
   mod/do_spo_analysis
   mod/do_nrha_analysis

References
----------

1. Minjie, Zhu. McKenna, F. and Scott, M.H. (2018). "OpenSeesPy: Python library for the OpenSees finite element
   framework", *SoftwareX*, Volume 7, 2018, Pages 6-11, ISSN 2352-7110,
   https://doi.org/10.1016/j.softx.2017.10.009.

SLF Generation
==============

.. automethod:: openquake.vmtk.slf_generator.slf_generator.generate

.. admonition:: Theoretical Background

   A Storey Loss Function (SLF) links the expected repair loss at a given storey to the
   Engineering Demand Parameter (EDP) at that storey (Ramirez & Miranda, 2009;
   Shahnazaryan et al., 2021).

   **Component loss model**

   For each damageable component :math:`c` in performance group :math:`g`, the repair
   cost :math:`\ell_c` is a random variable that depends on the damage state
   :math:`DS_c`. Given the EDP level :math:`x`, the expected cost contribution is:

   .. math::

      E[\ell_c \mid x] = \sum_{i=1}^{n_{DS}}
        \mu_{c,i} \cdot P(DS_c = i \mid x)

   where :math:`\mu_{c,i}` is the mean repair cost for damage state :math:`i` and
   :math:`P(DS_c = i \mid x)` is derived from the component fragility functions via:

   .. math::

      P(DS_c = i \mid x) =
        P(DS_c \geq i \mid x) - P(DS_c \geq i+1 \mid x)

   **Monte Carlo sampling**

   Damage states are sampled via Monte Carlo simulation across a user-defined EDP range.
   For :math:`N` realisations, each realisation draws a damage state for every component
   at every EDP level. Component costs are summed within each performance group to obtain
   the total group loss per realisation.

   **Regression and SLF fitting**

   The empirical loss-EDP cloud (from all Monte Carlo realisations) is fitted by
   regression to one of several functional forms (Weibull, Papadopoulos, Generalised
   Pareto, or Lognormal). The fitted curve is the SLF for the group:

   .. math::

      \hat{\ell}_g(x) = f_{\boldsymbol{\theta}}(x)

   where :math:`\boldsymbol{\theta}` are the regression parameters estimated by
   minimising the sum of squared residuals between the empirical median loss and the
   fitted curve.

   Empirical statistics (median, 16th and 84th percentiles) across realisations are
   also retained in the cache for uncertainty characterisation.



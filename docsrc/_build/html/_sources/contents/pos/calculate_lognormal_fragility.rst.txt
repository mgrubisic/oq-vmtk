Lognormal Fragility Functions
=============================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_lognormal_fragility

.. admonition:: Theoretical Background


   The lognormal fragility function is the most widely used model in earthquake
   engineering. It expresses the probability of exceeding a damage state as a
   lognormal cumulative distribution function (CDF) of the intensity measure.

   **Fragility model**

   .. math::

      P(\text{DS} \geq ds_i \mid \text{IM}) =
      \Phi\!\left(\frac{\ln(\text{IM}/\theta_i)}{\beta_{\text{total}}}\right)

   where:

   - :math:`\Phi(\cdot)` is the standard normal CDF,
   - :math:`\theta_i` is the median IM capacity (the IM level at which there is a 50 %
     probability of exceeding damage state :math:`i`),
   - :math:`\beta_{\text{total}}` is the total logarithmic standard deviation,
     combining all sources of uncertainty:

   .. math::

      \beta_{\text{total}} = \sqrt{\beta_{\text{r2r}}^2 + \beta_{\text{b2b}}^2 + \beta_{\text{DS}}^2}

   with :math:`\beta_{\text{r2r}}` the record-to-record variability,
   :math:`\beta_{\text{b2b}}` the building-to-building modelling uncertainty,
   and :math:`\beta_{\text{DS}}` the uncertainty in the damage-state threshold.



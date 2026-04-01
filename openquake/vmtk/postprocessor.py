import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats, optimize
from scipy.optimize import curve_fit
from scipy.stats import norm, lognorm
from scipy.interpolate import interp1d
from statsmodels.miscmodels.ordinal_model import OrderedModel


class postprocessor:

    """
    Class for post-processing results of nonlinear time-history analysis,
    including fragility and vulnerability analysis.

    This class provides methods to compute fragility functions, perform
    cloud and multiple stripe analyses, and calculate vulnerability
    functions and average annual losses. It supports various fragility
    fitting methods, including lognormal, probit, logit, and ordinal
    models. The class also includes functionality to handle uncertainty
    and variability in the analysis.

    Methods
    -------
    calculate_lognormal_fragility(
        theta, sigma_record2record, sigma_build2build=0.30,
        sigma_ds=0.30,
        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Computes the probability of exceeding a damage state using a
        lognormal cumulative distribution function (CDF).

    calculate_rotated_fragility(
        percentile, theta, sigma_record2record,
        sigma_build2build=0.30, sigma_ds=0.30,
        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Calculates a rotated fragility function based on a lognormal
        CDF, adjusting the median intensity to align with a specified
        target percentile.

    calculate_glm_fragility(
        imls, edps, damage_thresholds,
        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
        fragility_method='logit')
        Computes non-parametric fragility functions using Generalized
        Linear Models (GLM) with either a Logit or Probit link function.

    calculate_ordinal_fragility(
        imls, edps, damage_thresholds,
        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Fits an ordinal (cumulative) probit model to estimate fragility
        curves for different damage states.

    process_mca_results(
        imls, edps, damage_thresholds, lower_limit, censored_limit,
        sigma_build2build=0.3, sigma_ds=0.3,
        intensities=np.geomspace(0.05, 10, 50), n_bootstrap=200,
        random_seed=None, fragility_rotation=False,
        rotation_percentile=0.10, fragility_method='lognormal')
        Postprocess Modified Cloud Analysis (MCA) results: fits the
        probabilistic seismic demand model and derives fragility functions
        from pre-computed NLTHA outputs.

    process_msa_results(
        imls, edps, damage_thresholds, sigma_build2build=0.3,
        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
        fragility_rotation=False, rotation_percentile=0.10)
        Postprocess Multiple Stripe Analysis (MSA) results: maximum
        likelihood estimation for fragility curve fitting from
        pre-computed NLTHA stripe outputs.

    calculate_vulnerability_function(
        poes, consequence_model, cov_consequence=None,
        uncertainty=True, method=None,
        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Compute a vulnerability function by convolving fragility
        functions with a consequence (damage-to-loss) model.

    calculate_average_annual_damage_probability(
        fragility_array, hazard_array, return_period=1,
        max_return_period=5000)
        Calculate the Average Annual Damage State Probability (AADP)
        based on fragility and hazard curves.

    calculate_average_annual_loss(
        vulnerability_array, hazard_array, return_period=1,
        max_return_period=5000)
        Calculate the Average Annual Loss (AAL) based on vulnerability
        and hazard curves.

    """

    def __init__(self):
        pass

    def calculate_lognormal_fragility(self,
                                      theta,
                                      sigma_record2record,
                                      sigma_build2build=0.30,
                                      sigma_ds=0.30,
                                      intensities=np.round(
                                          np.geomspace(0.05, 10.0, 50),
                                          3)):
        """
        Computes the probability of exceeding a damage state using a
        lognormal cumulative distribution function (CDF).

        Parameters
        ----------
        theta : float
            The median seismic intensity corresponding to an EDP-based
            damage threshold.

        sigma_record2record : float
            The logarithmic standard deviation representing
            record-to-record variability.

        sigma_build2build : float, optional
            The logarithmic standard deviation representing
            building-to-building (or model) variability.
            Default value is 0.30.

        sigma_ds : float, optional
            The logarithmic standard deviation representing uncertainty
            in damage-state thresholds. Default value is 0.30.

        intensities : array-like, optional
            The set of intensity measure (IM) levels for which
            exceedance probabilities will be computed. Default is a
            geometric sequence from 0.05 to 10.0 with 50 points.

        Returns
        -------
        poes : numpy.ndarray
            An array of exceedance probabilities corresponding to each
            intensity measure in `intensities`.

        References
        -----
        1) Baker JW. Efficient Analytical Fragility Function Fitting
        Using Dynamic Structural Analysis. Earthquake Spectra.
        2015;31(1):579-599. doi:10.1193/021113EQS025M

        2) Singhal A, Kiremidjian AS. Method for probabilistic
        evaluation of seismic structural damage. Journal of Structural
        Engineering 1996; 122: 1459-1467.
        DOI:10.1061/(ASCE)0733-9445(1996)122:12(1459)

        3) Lallemant, D., Kiremidjian, A., and Burton, H. (2015),
        Statistical procedures for developing earthquake damage
        fragility curves. Earthquake Engng Struct. Dyn., 44, 1373-1389.
        doi: 10.1002/eqe.2522.

        4) Bird JF, Bommer JJ, Bray JD, Sancio R, Spence RJS. Comparing
        loss estimation with observed damage in a zone of ground failure:
        a study of the 1999 Kocaeli Earthquake in Turkey. Bulletin of
        Earthquake Engineering 2004; 2: 329-360.
        DOI: 10.1007/s10518-004-3804-0

        """

        # Calculate the total uncertainty
        beta_total = np.sqrt(sigma_record2record**2 +
                             sigma_build2build**2 + sigma_ds**2)

        # Compute exceedance probabilities for each intensity level
        return lognorm.cdf(intensities, s=beta_total, loc=0, scale=theta)

    def calculate_rotated_fragility(self,
                                    percentile,
                                    theta,
                                    sigma_record2record,
                                    sigma_build2build=0.30,
                                    sigma_ds=0.30,
                                    intensities=np.round(
                                        np.geomspace(0.05, 10.0, 50),
                                        3)):
        """
        Calculates a rotated fragility function based on a lognormal
        cumulative distribution function (CDF), adjusting the median
        intensity to align with a specified target percentile.

        This function modifies the median intensity based on the desired
        target percentile and total uncertainty (considering both
        record-to-record variability and modeling variability). The
        resulting rotated fragility curve represents the damage
        exceedance probabilities for a range of intensity measure
        levels, as defined by the lognormal distribution.

        Parameters
        ----------
        percentile : float
            The target percentile for fragility function rotation. This
            value corresponds to the desired percentile (e.g., 0.2
            corresponds to the 20th percentile of the fragility curve).
            The curve is adjusted such that this percentile aligns with
            the calculated fragility function.

        theta : float
            The median seismic intensity corresponding to the edp-based
            damage threshold.

        sigma_record2record : float
            The uncertainty associated with record-to-record variability
            in the seismic records used to derive the fragility.

        sigma_build2build : float, optional, default=0.30
            The uncertainty associated with modeling variability between
            different buildings or building types.

        sigma_ds : float, optional
            The logarithmic standard deviation representing uncertainty
            in damage-state thresholds. Default value is 0.30.

        intensities : array-like, optional,
            default=np.round(np.geomspace(0.05, 10.0, 50), 3)
            A list or array of intensity measure levels at which to
            evaluate the fragility function, typically representing
            seismic intensity levels (e.g., spectral acceleration). The
            default is a geometric space ranging from 0.05 to 10.0.

        Returns
        -------
        theta_prime : float
            The new median intensity after the rotation based on the
            specified percentile.

        beta_total : float
            The total standard deviation of the lognormal distribution,
            calculated from both record-to-record and building-to-
            building (modelling) uncertainties.

        poes : array-like
            The probabilities of exceedance (fragility values)
            corresponding to the input intensity measure levels. This is
            the lognormal CDF evaluated at the given intensities with
            the rotated median and combined uncertainty.

        References
        ----------
        1) Porter, K. (2017), "When Addressing Epistemic Uncertainty in
        a Lognormal Fragility Function, How Should One Adjust the
        Median?", Proceedings of the 16th World Conference on Earthquake
        Engineering (16WCEE), Santiago, Chile.

        """

        # Calculate combined log standard deviation (total uncertainty)
        beta_total = np.sqrt(sigma_record2record**2 +
                             sigma_build2build**2 + sigma_ds**2)

        # Adjust the median intensity based on the target percentile
        theta_prime = theta * \
            np.exp(-stats.norm.ppf(percentile) *
                   (beta_total - sigma_record2record))

        # Return rotated lognormal CDF for the given intensities
        return (theta_prime, beta_total,
                stats.lognorm(
                    s=beta_total, scale=theta_prime).cdf(intensities))

    def calculate_glm_fragility(self,
                                imls,
                                edps,
                                damage_thresholds,
                                intensities=np.round(
                                    np.geomspace(0.05, 10.0, 50), 3),
                                fragility_method='logit'):
        """
        Computes non-parametric fragility functions using Generalized
        Linear Models (GLM) with either a Logit or Probit link function.

        Parameters
        ----------
        imls : array-like
            Intensity Measure Levels (IMLs) corresponding to each
            observation.

        edps : array-like
            Engineering Demand Parameters (EDPs) representing structural
            response values.

        damage_thresholds : array-like
            List of thresholds defining different damage states.

        intensities : array-like, optional
            Intensity measure values at which probabilities of
            exceedance (PoEs) are evaluated. Defaults to
            np.round(np.geomspace(0.05, 10.0, 50), 3).

        fragility_method : str, optional
            Specifies the GLM model to be used for fragility function
            fitting. Options:

            - ``'logit'`` (default): Uses a logistic regression model.
            - ``'probit'``: Uses a probit regression model.

        Returns
        -------
        poes : ndarray
            A 2D array where each column represents the probability of
            exceeding a specific damage state at each intensity level.

        References
        ----------
        1) Charvet, I., Ioannou, I., Rossetto, T., Suppasri, A., and
        Imamura, F.: Empirical fragility assessment of buildings
        affected by the 2011 Great East Japan tsunami using improved
        statistical models, Nat. Hazards, 73, 951-973, 2014.

        2) Lahcene, E., Ioannou, I., Suppasri, A., Pakoksung, K.,
        Paulik, R., Syamsidik, S., Bouchette, F., and Imamura, F.:
        Characteristics of building fragility curves for seismic and
        non-seismic tsunamis: case studies of the 2018 Sunda Strait,
        2018 Sulawesi-Palu, and 2004 Indian Ocean tsunamis, Nat.
        Hazards Earth Syst. Sci., 21, 2313-2344,
        https://doi.org/10.5194/nhess-21-2313-2021, 2021.

        3) Lallemant, D., Kiremidjian, A., and Burton, H. (2015),
        Statistical procedures for developing earthquake damage
        fragility curves. Earthquake Engng Struct. Dyn., 44, 1373-1389.
        doi: 10.1002/eqe.2522.

        4) Jalayer, F., Ebrahamian, H., Trevlopoulos, K., and Bradley,
        B. (2023). Empirical tsunami fragility modelling for
        hierarchical damage levels. Natural Hazards and Earth System
        Sciences, 23(2), 909-931.
        https://doi.org/10.5194/nhess-23-909-2023

        """

        # Create probabilities of exceedance array
        poes = np.zeros((len(intensities), len(damage_thresholds)))

        for ds, current_threshold in enumerate(damage_thresholds):

            # Count exceedances
            exceedances = [1 if edp > damage_thresholds[ds]
                           else 0 for edp in edps]

            # Assemble dict: log of IMs and binary damage state labels
            data = {'IM': np.log(imls),
                    'Damage': exceedances}

            # Create DataFrame
            df = pd.DataFrame(data)

            # Add a constant for the intercept term
            X = sm.add_constant(df['IM'])
            y = df['Damage']

            if fragility_method.lower() == 'probit':

                # Fit the Probit GLM model
                probit_model = sm.GLM(y, X, family=sm.families.Binomial(
                    link=sm.families.links.Probit()))
                probit_results = probit_model.fit()

                # Generate a range of IM values for plotting
                log_IM_range = np.log(intensities)
                X_range = sm.add_constant(log_IM_range)

                # Predict probabilities using the Probit GLM model
                poes[:, ds] = probit_results.predict(X_range)

            elif fragility_method.lower() == 'logit':

                # Fit the Logit GLM model
                logit_model = sm.GLM(y, X, family=sm.families.Binomial(
                    link=sm.families.links.Logit()))
                logit_results = logit_model.fit()

                # Generate a range of IM values for plotting
                log_IM_range = np.log(intensities)
                X_range = sm.add_constant(log_IM_range)

                # Predict probabilities using the Probit GLM model
                poes[:, ds] = logit_results.predict(X_range)

        return poes

    def calculate_ordinal_fragility(self,
                                    imls,
                                    edps,
                                    damage_thresholds,
                                    intensities=np.round(
                                        np.geomspace(0.05, 10.0, 50),
                                        3)):
        """
        Fits an ordinal (cumulative) probit model to estimate fragility
        curves for different damage states.

        This function estimates the probability of exceeding various
        damage states using an ordinal regression model based on observed
        Engineering Demand Parameters (EDPs) and corresponding Intensity
        Measure Levels (IMLs).

        Parameters
        ----------
        imls : array-like
            Intensity measure levels corresponding to the observed EDPs.

        edps : array-like
            Engineering Demand Parameters (EDPs) representing structural
            responses.

        damage_thresholds : array-like
            Damage state thresholds for classifying exceedance levels.

        intensities : array-like, optional
            Intensity measure levels for which fragility curves are
            evaluated (default: np.geomspace(0.05, 10.0, 50)).

        Returns
        -------
        poes : numpy.ndarray
            A 2D array of exceedance probabilities (CDF values) for each
            intensity level. Shape: (len(intensities),
            len(damage_thresholds) + 1), where the last column represents
            the probability of exceeding the highest damage state.

        References
        -----
        1) Lallemant, D., Kiremidjian, A., and Burton, H. (2015),
        Statistical procedures for developing earthquake damage fragility
        curves. Earthquake Engng Struct. Dyn., 44, 1373-1389.
        doi: 10.1002/eqe.2522.

        2) Nguyen, M. and Lallemant, D. (2022), Order Matters: The
        Benefits of Ordinal Fragility Curves for Damage and Loss
        Estimation. Risk Analysis, 42: 1136-1148.
        https://doi.org/10.1111/risa.13815

        """

        # Create probabilities of exceedance array
        # +1 to include the highest damage state
        poes = np.zeros((len(intensities), len(damage_thresholds) + 1))

        # Initialize damage state assignments
        damage_states = np.zeros(len(edps), dtype=int)

        # Loop over each EDP and determine the highest exceeded damage state
        for i, edp in enumerate(edps):
            # Indices where EDP exceeds thresholds
            exceeded = np.where(edp > damage_thresholds)[0]
            # Assign highest exceeded state (0-based)
            damage_states[i] = exceeded[-1] + 1 if exceeded.size > 0 else 0

        # Assemble DataFrame containing log(IM) and damage state assignment
        df = pd.DataFrame({'IM': np.log(imls), 'Damage State': damage_states})

        # Fit the Cumulative Probit Model
        X_ordinal = df[['IM']]
        y_ordinal = df['Damage State']

        # Create and fit the OrderedModel
        ordinal_model = OrderedModel(y_ordinal, X_ordinal, distr='probit')
        ordinal_results = ordinal_model.fit(
            method='bfgs', disp=False)  # Silent optimization

        # Generate log-transformed IM values for prediction
        log_IM_range = np.log(intensities)
        X_range_ordinal = pd.DataFrame({'IM': log_IM_range})

        # Predict probabilities for each damage state (PMF)
        # Shape: (len(intensities), num_damage_states)
        pmf_values = ordinal_results.predict(X_range_ordinal)

        # Convert PMF to CDF (exceedance probabilities) by cumulative sum
        # Cumulative sum along damage state axis
        poes = 1 - np.cumsum(pmf_values, axis=1)

        return poes.values

    def process_mca_results(self,
                            imls,
                            edps,
                            damage_thresholds,
                            lower_limit,
                            censored_limit,
                            sigma_build2build=0.3,
                            sigma_ds=0.3,
                            intensities=np.geomspace(0.05, 10, 50),
                            n_bootstrap=200,
                            random_seed=None,
                            fragility_rotation=False,
                            rotation_percentile=0.10,
                            fragility_method='lognormal'):
        """
        Perform a Modified Cloud Analysis (MCA) to derive seismic
        fragility functions. This method extends classical cloud analysis
        by incorporating logistic regression to account for structural
        collapse cases and using bootstrapping to ensure statistical
        stability. It supports lognormal, ordinal, and GLM-based
        fragility fitting.

        Parameters
        ----------
        imls : array_like
            Intensity Measure Levels (e.g., Sa, AvgSA) from the cloud.

        edps : array_like
            Engineering Demand Parameters (e.g., maximum interstory
            drift) from the cloud.

        damage_thresholds : list of float
            The demand-based thresholds defining the onset of different
            damage states.

        lower_limit : float
            The EDP value below which data is ignored for regression
            (demand is considered negligible for damage).

        censored_limit : float
            The "Collapse" threshold. EDP values above this are treated
            as collapse instances in the logistic regression.

        sigma_build2build : float, optional
            Additional modeling uncertainty (building-to-building
            variability). Default is 0.3.

        sigma_ds : float, optional
            The logarithmic standard deviation representing uncertainty
            in damage-state thresholds. Default value is 0.30.

        intensities : np.array, optional
            The seismic intensity range over which to evaluate the
            fragility functions. Default is a geometric space from
            0.05 to 10.

        n_bootstrap : int, optional
            Number of bootstrap samples to draw for statistical
            stability. Default is 200.

        random_seed : int, optional
            Seed for reproducibility of the bootstrap sampling.
            Default is None.

        fragility_rotation : float, optional
            Parameter for rotating fragility functions around a specific
            percentile to adjust for target reliability. Default is 0.1.

        fragility_method : {'lognormal', 'ordinal', 'probit', 'logit'},
            optional. The methodology used to fit the fragility functions.
            Default is 'lognormal'.

        Returns
        -------
        cloud_dict : dict
            A nested dictionary with keys: ``'cloud inputs'`` (imls, edps,
            lower_limit, upper_limit, damage_thresholds); ``'fragility'``
            (fragility_method, intensities, poes, medians,
            sigma_record2record, sigma_build2build, sigma_ds, betas_total);
            ``'regression'`` (b1, b0, sigma, fitted_x, fitted_y — lognormal
            only); ``'bootstraps'`` (b1, a, sigma_rr, alpha0, alpha1,
            poes_all of shape n_bootstrap x n_IM x n_DS+1 — lognormal only);
            ``'raw_data'`` (im_nc, edp_nc, im_c — lognormal only).

        Notes
        -----
        The 'lognormal' method implements a dual-regression approach:
        1.  **Linear Regression**: Performed in log-log space on
            non-collapse data (log(EDP) = log(a) + b * log(IM)).
        2.  **Logistic Regression**: Used to predict P(C|IM).
        3.  **Total Fragility**: P(DS|IM) = P(DS|NC,IM)*P(NC|IM) +
            P(C|IM).
        """

        def cond_fragility(x, a, b):
            """
            Helper function that fits the conditioned fragility
            functions (i.e., considering collapse and non-collapse
            cases).
            """
            return (1-np.exp(-a*x))**b

        def prepare_mca_data(imls, edps, collapse_limit, bootstrap=True):
            """
            Helper function that standardizes the cloud input parameters
            and splits IMs and EDPs into collapse and non-collapse cases.
            Then implements bootstrapping to ensure stability in Logistic
            Regression.
            """
            # Ensure inputs are arrays
            imls = np.array(imls)
            edps = np.array(edps)
            n_total = len(imls)

            # Identify indicies where collapse occurs in the original dataset
            mask_coll_ori = edps > collapse_limit
            im_coll_ori = imls[mask_coll_ori]
            edp_coll_ori = edps[mask_coll_ori]
            npt_ori = len(im_coll_ori)

            # Bootstrap sampling
            if bootstrap:
                idx_boot = np.random.randint(0, n_total, size=n_total)
                sample_im = imls[idx_boot]
                sample_edp = edps[idx_boot]
            else:
                sample_im = imls.copy()
                sample_edp = edps.copy()

            # Standardization
            npts_sample_coll = np.sum(sample_edp > collapse_limit)
            target_min = math.ceil(0.5*npt_ori)
            # Stabilize Logistic Regression if bootstrap has few collapses
            if npts_sample_coll < target_min and npt_ori > 0:
                add_count = int(target_min - npts_sample_coll)
                # Resample from original collapse points (1D)
                extra_indices = np.random.choice(
                    len(im_coll_ori), size=add_count, replace=True)

                # Use 1D indexing: im_coll_ori[extra_indices]
                sample_im = np.concatenate(
                    [sample_im, im_coll_ori[extra_indices]])
                sample_edp = np.concatenate(
                    [sample_edp, edp_coll_ori[extra_indices]])

            # Final split
            is_coll = sample_edp > collapse_limit  # Collapse mask

            im_nc = sample_im[~is_coll]  # Non-collapse IMs
            edp_nc = sample_edp[~is_coll]  # Non-collapse EDPs
            im_c = sample_im[is_coll]   # Collapse IMs

            return im_nc, edp_nc, im_c

        # Compute exceedance probabilities using the specified fragility method
        if fragility_method in ['probit', 'logit']:

            # Get the probabilities of exceedance
            poes = self.calculate_glm_fragility(
                imls, edps, damage_thresholds,
                fragility_method=fragility_method)

            # Compute lognormal equivalent fragility parameters
            # Equivalent median intensities
            thetas = [np.interp(0.50, poes[:, ds], intensities)
                      for ds in range(len(damage_thresholds))]
            sigmas_record2record = [
                np.abs(0.50 * (
                    np.log(np.interp(0.84, poes[:, ds], intensities))
                    - np.log(np.interp(
                        0.16, poes[:, ds], intensities))))
                for ds in range(len(damage_thresholds))
            ]  # Equivalent record-to-record variability
            # Modelling uncertainty
            sigmas_build2build = np.full(
                len(damage_thresholds), sigma_build2build)
            # Uncertainty in DS thresholds
            sigmas_ds = np.full(len(damage_thresholds), sigma_ds)
            betas_total = [
                np.sqrt(
                    sigma_record2record**2
                    + sigma_build2build**2
                    + sigma_ds**2)
                for sigma_record2record, sigma_build2build, sigma_ds
                in zip(
                    sigmas_record2record,
                    sigmas_build2build,
                    sigmas_ds)
            ]  # Total dispersion

            # Create the dictionary
            cloud_dict = {
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {'imls': imls,
                                 'edps': edps,
                                 'lower_limit': None,
                                 'upper_limit': None,
                                 'damage_thresholds': damage_thresholds},

                # Add a nested dictionary for fragility functions parameters
                'fragility': {'fragility_method': fragility_method.lower(),
                              'intensities': intensities,
                              'poes': poes,
                              'medians': thetas,
                              'sigma_record2record': sigmas_record2record,
                              'sigma_build2build': sigmas_build2build,
                              'sigma_ds': sigmas_ds,
                              'betas_total': betas_total},

                # Add a nested dictionary for regression coefficients
                'regression': {'b1': None,   # Store 'b1' coefficient
                               'b0': None,   # Store 'b0' coefficient
                               'sigma': None,   # Store 'sigma' value
                               'fitted_x': None,   # Store the fitted x-values
                               'fitted_y': None}   # Store the fitted y-values
            }

            return cloud_dict

        elif fragility_method.lower() == 'ordinal':

            # Compute exceedance probabilities via ordinal fragility
            poes = self.calculate_ordinal_fragility(
                imls, edps, damage_thresholds)

            # Compute lognormal equivalent fragility parameters
            # Equivalent median intensities
            thetas = [np.interp(0.50, poes[:, ds], intensities)
                      for ds in range(len(damage_thresholds))]
            sigmas_record2record = [
                np.abs(0.50 * (
                    np.log(np.interp(0.84, poes[:, ds], intensities))
                    - np.log(np.interp(
                        0.16, poes[:, ds], intensities))))
                for ds in range(len(damage_thresholds))
            ]  # Equivalent record-to-record variability
            # Modelling uncertainty
            sigmas_build2build = np.full(
                len(damage_thresholds), sigma_build2build)
            # Uncertainty in DS thresholds
            sigmas_ds = np.full(len(damage_thresholds), sigma_ds)
            betas_total = [
                np.sqrt(
                    sigma_record2record**2
                    + sigma_build2build**2
                    + sigma_ds**2)
                for sigma_record2record, sigma_build2build, sigma_ds
                in zip(
                    sigmas_record2record,
                    sigmas_build2build,
                    sigmas_ds)
            ]  # Total dispersion

            # Create the dictionary
            cloud_dict = {
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {
                    'imls': imls,
                    'edps': edps,
                    'lower_limit': None,
                    'upper_limit': None,
                    'damage_thresholds': damage_thresholds},

                # Fragility functions parameters
                'fragility': {
                    'fragility_method': fragility_method.lower(),
                    'intensities': intensities,
                    'poes': poes,
                    'medians': thetas,
                    'sigma_record2record': sigmas_record2record,
                    'sigma_build2build': sigmas_build2build,
                    'sigma_ds': sigmas_ds,
                    'betas_total': betas_total},

                # Add a nested dictionary for regression coefficients
                'regression': {'b1': None,   # Store 'b1' coefficient
                               'b0': None,   # Store 'b0' coefficient
                               'sigma': None,   # Store 'sigma' value
                               'fitted_x': None,   # Store the fitted x-values
                               'fitted_y': None}   # Store the fitted y-values
            }

            return cloud_dict

        elif fragility_method.lower() == 'lognormal':

            # Initialise seed for reproducibility
            if random_seed is not None:
                np.random.seed(random_seed)

            # Ensure inputs are in the right format
            imls, edps = np.asarray(imls), np.asarray(edps)

            # Storage for exceedance probabilities and regression parameters
            n_ds = len(damage_thresholds)
            n_im = len(intensities)
            # +1 to store the "collapse" fragility in the last index
            poes_s = np.zeros((n_bootstrap, n_im, n_ds + 1))
            a_s, b_s, sig_s = np.zeros(n_bootstrap), np.zeros(
                n_bootstrap), np.zeros(n_bootstrap)
            al0_s, al1_s = np.zeros(n_bootstrap), np.zeros(n_bootstrap)

            # Bootstrapping loop
            for i in range(n_bootstrap):

                # Prepare bootstrap samples
                im_nc_b, edp_nc_b, im_c_b = prepare_mca_data(imls,
                                                             edps,
                                                             censored_limit,
                                                             bootstrap=True)

                # Cloud regression on non-collapse cases above lower_limit
                mask_lower = edp_nc_b >= lower_limit
                ln_im, ln_edp = np.log(im_nc_b[mask_lower]), np.log(
                    edp_nc_b[mask_lower])  # log-log transform
                b = (
                    np.sum(
                        (ln_im - ln_im.mean())
                        * (ln_edp - ln_edp.mean()))
                    / np.sum((ln_im - ln_im.mean()) ** 2)
                )  # b-parameter
                # Calculate the a-parameter
                a = np.exp(ln_edp.mean()-b*ln_im.mean())
                # Apply the regression to get the mean
                res = ln_edp - np.log(a*im_nc_b[mask_lower]**b)
                # Get the standard error
                sig = np.linalg.norm(res)/np.sqrt(len(res)-2)

                # Store cloud analysis coefficients for this bootstrap
                a_s[i], b_s[i], sig_s[i] = a, b, sig

                # Do logistic regression to account for the collapse cases
                y_logit = np.concatenate(
                    [np.zeros(len(im_nc_b)), np.ones(len(im_c_b))])
                x_logit = sm.add_constant(
                    np.log(np.concatenate([im_nc_b, im_c_b])))
                logit_mod = sm.GLM(
                    y_logit, x_logit,
                    family=sm.families.Binomial()).fit(disp=0)

                # Store logistic regression coefficients for this bootstrap
                al0_s[i], al1_s[i] = logit_mod.params

                # Calculate the probabilities of exceedance
                p_collapse = logit_mod.predict(sm.add_constant(
                    np.log(intensities)))  # The probability of collapse
                # The cloud regression
                mu_ln = np.log(a * intensities**b)
                # Total uncertainty (b2b + DS threshold inflated)
                sig_total = np.sqrt(sig**2 + sigma_build2build**2+sigma_ds**2)

                # Loop over damage states
                for ds in range(n_ds):
                    # Calculate the non-collapse fragility functions
                    poe_nc = 1 - \
                        norm.cdf(
                            np.log(damage_thresholds[ds]),
                            loc=mu_ln, scale=sig_total)
                    # Calculate the conditional fragility functions P(NC, IM|C)
                    poes_s[i, :, ds] = poe_nc * (1-p_collapse) + p_collapse

                # Store the collapse fragility
                poes_s[i, :, -1] = p_collapse

            # Storage for mean statistics and lognormal CDF parameters
            poes_mean = poes_s.mean(axis=0)
            poes_fitted = np.zeros_like(poes_mean)
            params_a, params_b = np.zeros(n_ds+1), np.zeros(n_ds+1)
            medians = np.zeros(n_ds+1)
            betas_total = np.zeros(n_ds+1)
            sigmas_ds = np.full(len(damage_thresholds), sigma_ds)

            # Loop over damage states and collapse
            for ds in range(n_ds+1):
                # Fit functional form
                try:
                    popt, _ = curve_fit(cond_fragility,
                                        intensities,
                                        poes_mean[:, ds],
                                        bounds=((0, 0), (np.inf, np.inf)))
                    params_a[ds], params_b[ds] = popt

                except Exception as e:
                    raise RuntimeError(
                        f'ERROR! Curve fitting failed for DS '
                        f'{ds}: {e}')

                # Calculate the fitted probabilities
                poes_fitted[:, ds] = cond_fragility(intensities,
                                                    params_a[ds],
                                                    params_b[ds])

                # Interpolate for lognormal equivalents at 16%, 50%, 84%
                f_interp = interp1d(
                    poes_fitted[:, ds], intensities,
                    bounds_error=False, fill_value='extrapolate')
                medians[ds] = f_interp(0.5)
                im16 = f_interp(0.16)
                im84 = f_interp(0.84)

                # Calculate the uncertainty
                if im16 > 0 and im84 > im16:
                    betas_total[ds] = np.log(im84/im16)/2
                else:
                    betas_total[ds] = np.nan

            # Recalculate the lognormal fragility functions
            for ds in range(n_ds+1):

                if fragility_rotation:
                    fragility_method = (
                        f'lognormal - rotated around the '
                        f'{rotation_percentile}th percentile')
                    (medians[ds],
                     betas_total[ds],
                     poes_fitted[:, ds]) = (
                        self.calculate_rotated_fragility(
                            rotation_percentile,
                            medians[ds],
                            betas_total[ds],
                            sigma_build2build=0.0,
                            sigma_ds=0.0))
                else:
                    poes_fitted[:, ds] = (
                        self.calculate_lognormal_fragility(
                            medians[ds],
                            betas_total[ds],
                            sigma_build2build=0.0,
                            sigma_ds=0.0))

            # Final cleanup: ensure fragility functions are not crossing
            # Work backwards from Collapse to DS1 to ensure PoE(DS_i) is always
            # >= PoE(Collapse) and PoE(DS_i) >= PoE(DS_i+1)
            for i in range(n_ds-1, -1, -1):
                poes_fitted[:, i] = np.maximum(
                    poes_fitted[:, i], poes_fitted[:, i+1])

            # Store everything in dedicated dictionary
            is_collapse = edps >= censored_limit
            is_nc_plot = (~is_collapse) & (edps >= lower_limit)

            # Create the dictionary
            cloud_dict = {
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {
                    'imls': imls,
                    'edps': edps,
                    'lower_limit': lower_limit,
                    'upper_limit': censored_limit,
                    'damage_thresholds': damage_thresholds},

                # Fragility functions parameters
                'fragility': {
                    'fragility_method': fragility_method.lower(),
                    'intensities': intensities,
                    'poes': poes_fitted,
                    'medians': medians,
                    'sigma_record2record': sig_s.mean(),
                    'sigma_build2build': sigma_build2build,
                    'sigma_ds': sigmas_ds,
                    'betas_total': betas_total},

                # Regression coefficients
                'regression': {
                    'b1': b_s.mean(),
                    'b0': np.log(a_s.mean()),
                    'sigma': sig_s.mean(),
                    'alpha0': al0_s.mean(),
                    'alpha1': al1_s.mean(),
                    'fitted_x': np.log(intensities),
                    'fitted_y': (np.log(a_s.mean())
                                 + b_s.mean() * np.log(intensities))},

                # Bootstrap iteration results
                'bootstraps': {
                    'b1': b_s,
                    'a': a_s,
                    'sigma_rr': sig_s,
                    'alpha0': al0_s,
                    'alpha1': al1_s,
                    'poes_all': poes_s},

                'raw_data': {'im_nc': imls[is_nc_plot],
                             'edp_nc': edps[is_nc_plot],
                             'im_c': imls[is_collapse]}
            }

            return cloud_dict

    # ---------------------------------------------------------------
    # POSTPROCESS MULTIPLE STRIPE ANALYSIS RESULTS
    # ---------------------------------------------------------------
    def process_msa_results(self,
                            imls,
                            edps,
                            damage_thresholds,
                            sigma_build2build=0.3,
                            sigma_ds=0.3,
                            intensities=np.round(
                                np.geomspace(0.05, 10.0, 50), 3),
                            fragility_rotation=False,
                            rotation_percentile=0.10):
        """
        Perform maximum likelihood estimation (MLE) for fragility curve
        fitting following a multiple stripe analysis. This method
        calculates the fragility function by fitting to the provided
        intensity measure levels (IMLs) and engineering demand parameters
        (EDPs) "stripes", with the option to rotate the fragility curve
        around a target percentile.

        The method is useful for deriving fragility functions by
        determining the probability of exceedance for various damage
        states based on the provided data.

        Parameters
        ----------
        imls : list or array
            A list or array of intensity measure levels (IMLs)
            representing the seismic intensity levels used for sampling
            the fragility functions.

        edps : list or array
            A list or array of engineering demand parameters (EDPs),
            which describe the structural response to seismic events.
            Examples include maximum interstorey drifts, maximum peak
            floor acceleration, or top displacements.

        damage_thresholds : list
            A list of EDP-based damage thresholds that correspond to
            different levels of structural damage, such as slight,
            moderate, extensive, and complete. These thresholds help
            categorize the severity of damage based on EDP values.

        sigma_build2build : float, optional, default=0.3
            The building-to-building variability or modeling uncertainty.
            It accounts for differences in performance between buildings
            with similar characteristics due to random variations or
            model uncertainties.

        sigma_ds : float, optional
            The logarithmic standard deviation representing uncertainty
            in damage-state thresholds. Default value is 0.30.

        intensities : array, optional, default=np.geomspace(0.05, 10.0, 50)
            An array of intensity measure levels over which the fragility
            function will be sampled. By default, this is a logarithmic
            space ranging from 0.05 to 10.0, with 50 sample points.

        fragility_rotation : bool, optional, default=False
            A boolean flag that determines whether or not to rotate the
            fragility curve about a given percentile. If `True`, the
            fragility curve will be adjusted based on the specified
            `rotation_percentile`.

        rotation_percentile : float, optional, default=0.10
            The target percentile (between 0 and 1) around which the
            fragility function will be rotated. A value of 0.10
            corresponds to rotating the curve to the 10th percentile.

        Returns
        -------
        msa_dict : dict
            A nested dictionary with the following top-level keys:

            **'msa_inputs'** : dict
                Inputs passed to the analysis.

                - ``'imls'`` : array — intensity measure levels
                  used for each stripe.
                - ``'edps'`` : array — engineering demand
                  parameters recorded at each stripe.
                - ``'damage_thresholds'`` : list — EDP values
                  defining each damage state boundary.
                - ``'sigma_build2build'`` : float — building-to-
                  building modelling uncertainty.
                - ``'sigma_ds'`` : float — uncertainty in the
                  damage-state threshold definition.
                - ``'is_rotated'`` : bool — whether fragility
                  rotation was applied.

            **'fragility'** : dict
                MLE-fitted fragility function results.

                - ``'fragility_method'`` : str — always 'mle'.
                - ``'intensities'`` : array — IM levels at which
                  PoEs are evaluated.
                - ``'poes'`` : ndarray, shape (n_IM, n_DS) —
                  probabilities of exceedance per damage state.
                - ``'medians'`` : list, length n_DS — median IM
                  (theta) for each damage state.
                - ``'sigma_record2record'`` : list, length n_DS —
                  record-to-record dispersion per damage state.
                - ``'betas_total'`` : list, length n_DS — total
                  logarithmic standard deviation per damage state,
                  combining record-to-record, building-to-building,
                  and damage-state threshold uncertainties.

            **'metadata'** : dict
                Stripe-level summary information.

                - ``'stripe_levels'`` : array — mean IM value for
                  each stripe.
                - ``'observed_fractions'`` : list of lists,
                  shape (n_DS, n_stripes) — observed fraction of
                  ground-motion records exceeding each damage
                  threshold at each stripe level.

        Notes
        -----
        This method fits the fragility curve using MLE, which minimizes
        the difference between observed and predicted exceedance
        probabilities. The option for fragility curve rotation allows for
        adjusting the curve to better match the expected percentile of
        damage occurrence, offering greater flexibility in representing
        the fragility of the structure.
        """

        # Convert to numpy arrays
        imls = np.array(imls)
        edps = np.array(edps)

        if intensities is None:
            intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)

        # Handle IM levels: Extract stripe values (one per column)
        stripe_imls = np.mean(imls, axis=0) if imls.ndim > 1 else imls
        num_stripes = len(stripe_imls)
        num_gmrs_per_stripe = np.array(
            [len(edps[:, i]) for i in range(num_stripes)])

        def negative_log_likelihood(params, x, n, z):
            """
            Negative Log-Likelihood for Binomial Distribution.
            Fits the aleatory (record-to-record) parameters to the data.
            """
            theta, beta_r2r = params

            # Probability of exceedance based on lognormal CDF
            # We use the standard normal CDF on the log-transformed data
            p = stats.norm.cdf(np.log(x / theta) / beta_r2r)

            # Numerical stability: clip p to avoid log(0) in binomial logpmf
            p = np.clip(p, 1e-15, 1 - 1e-15)

            # log-likelihood of observing z exceedances in n trials
            log_f = stats.binom.logpmf(z, n, p)
            return -np.sum(log_f)

        # Storage lists
        thetas = []
        sigmas_record2record = []
        betas_total = []

        # Iterate through each Damage State threshold
        for threshold in damage_thresholds:
            # Count exceedances per stripe (z values)
            num_exc = np.array([np.sum(edps[:, i] >= threshold)
                               for i in range(num_stripes)])

            # Initial guess: theta = median of stripes, beta = 0.4
            initial_guess = [np.median(stripe_imls), 0.4]

            # Bounds: prevent median=0 and keep beta in physical range
            bounds = optimize.Bounds([0.001 * np.min(stripe_imls), 0.05],
                                     [10.0 * np.max(stripe_imls), 1.5])

            # Optimize the aleatory fit
            sol = optimize.minimize(
                negative_log_likelihood,
                initial_guess,
                args=(stripe_imls, num_gmrs_per_stripe, num_exc),
                bounds=bounds,
                method='L-BFGS-B'
            )

            t_val = sol.x[0]  # Fitted Median
            s_val = sol.x[1]  # Fitted Beta (Record-to-Record)

            # Combine uncertainties AFTER fitting using SRSS
            b_tot = np.sqrt(s_val**2 + sigma_build2build**2 + sigma_ds**2)

            thetas.append(t_val)
            sigmas_record2record.append(s_val)
            betas_total.append(b_tot)

        # Calculate Fragility Curves (POEs) over the full intensity range
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        for i in range(len(damage_thresholds)):
            if fragility_rotation:
                # Assuming this helper exists in your class
                poes[:, i] = self.calculate_rotated_fragility(
                    thetas[i],
                    rotation_percentile,
                    sigmas_record2record[i],
                    sigma_build2build=sigma_build2build,
                    sigma_ds=sigma_ds
                )
            else:
                # Standard lognormal PoE
                poes[:, i] = stats.norm.cdf(
                    np.log(intensities / thetas[i]) / betas_total[i])

        # Formatting Output
        # observed_fractions: list of arrays (one array per DS)
        msa_dict = {
            'msa_inputs': {
                'imls': imls,
                'edps': edps,
                'damage_thresholds': damage_thresholds,
                'sigma_build2build': sigma_build2build,
                'sigma_ds': sigma_ds,
                'is_rotated': fragility_rotation
            },
            'fragility': {
                'fragility_method': 'mle',
                'intensities': intensities,
                'poes': poes,
                'medians': thetas,
                'sigma_record2record': sigmas_record2record,
                'betas_total': betas_total
            },
            'metadata': {
                'stripe_levels': stripe_imls,
                'observed_fractions': [
                    [np.sum(edps[:, j] >= thresh) / num_gmrs_per_stripe[j]
                     for j in range(num_stripes)]
                    for thresh in damage_thresholds
                ]
            }
        }

        return msa_dict

    # ---------------------------------------------------------------
    # POSTPROCESS INCREMENTAL DYNAMIC ANALYSIS RESULTS
    # ---------------------------------------------------------------
    def process_ida_results(self,
                            ansys_dict,
                            im_matrix,
                            damage_thresholds,
                            edp_key,
                            sigma_build2build=0.3,
                            sigma_ds=0.3,
                            intensities=np.round(
                                np.geomspace(0.05, 10.0, 50), 3),
                            edp_range=np.linspace(0.00, 0.05, 101),
                            fragility_rotation=False,
                            rotation_percentile=0.10):
        """
        Perform fragility function fitting and statistical processing on
        Incremental Dynamic Analysis (IDA) results.

        This method processes raw IDA data by interpolating individual
        record response curves to a continuous Engineering Demand
        Parameter (EDP) range. It accounts for "flatlining" (global
        dynamic instability) using Maximum Likelihood Estimation (MLE)
        for censored data to estimate the fragility parameters (median
        and dispersion) for multiple damage states. It also supports
        fragility curve rotation around a target percentile to account
        for modeling uncertainties.

        Parameters
        ----------
        ansys_dict : dict
            A dictionary containing the structural response data with keys:

            - ``'max_peak_drift_list'`` or ``'max_peak_accel_list'``: a list
              where each entry maps Scale Factors (SF) to peak drift or
              acceleration values.
            - ``'sf_matrix'``: a 2D numpy array (n_records × max_runs)
              of scale factors used for each analysis run.

        im_matrix : numpy.ndarray
            A 2D array of intensity measure levels corresponding to the
            ground motion records and number of runs carried out in IDA.

        damage_thresholds : list of float
            A list of EDP-based damage thresholds (e.g., interstorey
            drift ratios) defining different limit states (e.g., Slight,
            Moderate, Extensive, Collapse).

        edp_key : str, optional,
            default='max_peak_drift_list' (other: 'max_peak_accel_list')
            The key in `ansys_dict` used to retrieve the engineering
            demand parameter data.

        sigma_build2build : float, optional, default=0.3
            The modeling uncertainty or building-to-building variability.
            This is combined with the record-to-record variability to
            calculate total fragility dispersion.

        sigma_ds : float, optional
            The logarithmic standard deviation representing uncertainty
            in damage-state thresholds. Default value is 0.30.

        intensities : numpy.ndarray, optional,
            default=np.geomspace(0.05, 10.0, 50)
            The array of intensity measure levels over which the final
            fragility functions (POEs) will be sampled.

        edp_range : numpy.ndarray, optional,
            default=np.linspace(0.00, 0.05, 101) (0% to 5% drift)
            The array of engineering demand parameters over which the
            IDA curves will be evaluated.

        fragility_rotation : bool, optional, default=False
            Flag to determine if the fragility curves should be rotated
            around a specific percentile to adjust for modeling bias or
            target reliability levels.

        rotation_percentile : float, optional, default=0.10
            The target percentile (0.0 to 1.0) around which the fragility
            curve rotation is anchored if `fragility_rotation` is True.

        Returns
        -------
        ida_dict : dict
            A nested dictionary with the following top-level keys:

            **'ida_inputs'** : dict
                Raw IDA data and analysis configuration.

                - ``'target_edps'`` : array — continuous EDP axis
                  over which IM values are interpolated.
                - ``'raw_curves'`` : list of dicts, one per record,
                  each containing ``'im'`` (sorted IM array) and
                  ``'edp'`` (monotonic EDP array).
                - ``'damage_thresholds'`` : list — EDP values that
                  define each damage state boundary.
                - ``'im_matrix'`` : array — IM values applied to
                  each record at each run.
                - ``'n_records'`` : int — number of ground-motion
                  records in the analysis.
                - ``'im_max_analyzed'`` : float — maximum IM level
                  reached across all records and runs.

            **'fragility'** : dict
                MLE-fitted fragility parameters and PoEs.

                - ``'fragility_method'`` : str — always
                  'mle_ida_censored'.
                - ``'intensities'`` : array — IM levels at which
                  PoEs are evaluated.
                - ``'poes'`` : ndarray, shape (n_IM, n_DS) —
                  probabilities of exceedance per damage state.
                - ``'medians'`` : list, length n_DS — median IM
                  (theta) for each damage state.
                - ``'sigma_record2record'`` : list, length n_DS —
                  record-to-record dispersion per damage state,
                  estimated via MLE on censored capacities.
                - ``'sigma_build2build'`` : list, length n_DS —
                  building-to-building modelling uncertainty.
                - ``'sigma_ds'`` : list, length n_DS — uncertainty
                  in the damage-state threshold definition.
                - ``'betas_total'`` : list, length n_DS — total
                  logarithmic standard deviation per damage state.
                - ``'rotation_active'`` : bool — whether fragility
                  rotation around a percentile was applied.
                - ``'rotation_percentile'`` : float or None — the
                  percentile used for rotation, or None if inactive.

            **'stats'** : dict
                Statistical IDA curves across all records.

                - ``'fitted_edps'`` : array — EDP axis shared with
                  ``'target_edps'`` in ``'ida_inputs'``.
                - ``'median_im'`` : array — median IM across all
                  records at each EDP level (50th percentile curve).
                - ``'p16_im'`` : array — 16th percentile IM across
                  records at each EDP level.
                - ``'p84_im'`` : array — 84th percentile IM across
                  records at each EDP level.

        Notes
        -----
        The method uses a log-likelihood minimization approach to handle
        records that do not reach a specific damage threshold within the
        analyzed range (right-censored data), ensuring the fragility
        curves remain statistically robust even near collapse.
        """

        drifts_data = ansys_dict[edp_key]
        sf_matrix = ansys_dict['sf_matrix']
        n_records = len(drifts_data)

        # Define Continuous EDP range (X-axis for the statistical lines)
        im_at_edp_matrix = []
        raw_curves = []

        for i in range(n_records):
            rec_ims, rec_edps = [], []
            for j, sf in enumerate(sf_matrix[i, :]):
                if not np.isnan(sf) and sf in drifts_data[i]:
                    rec_ims.append(im_matrix[i, j])
                    rec_edps.append(drifts_data[i][sf])

            if len(rec_ims) > 1:
                # Sort by IM (not EDP) to preserve analysis sequence
                idx = np.argsort(rec_ims)
                sorted_ims = np.array(rec_ims)[idx]
                sorted_edps = np.array(rec_edps)[idx]

                # Handle structural resurrection
                # Ensures that once a limit state is reached, it stays reached.
                monotonic_edps = np.maximum.accumulate(sorted_edps)

                raw_curves.append({'im': sorted_ims, 'edp': monotonic_edps})

                # Interpolate IM = f(EDP)
                # We interpolate against the monotonic EDPs to find the FIRST
                # occurrence of each threshold.
                # Use fill_value=np.nan for points beyond the analyzed range
                # to allow the MLE to handle censoring correctly.
                f_im_cap = interp1d(monotonic_edps, sorted_ims,
                                    bounds_error=False, fill_value=np.nan)

                im_at_edp_matrix.append(f_im_cap(edp_range))
            else:
                im_at_edp_matrix.append(np.full_like(edp_range, np.nan))

        im_at_edp_matrix = np.array(im_at_edp_matrix)

        # Handle collapse cases
        # Replace NaNs that occur AFTER a record has flatlined with the max IM
        # achieved for that record to represent the capacity 'ceiling'.
        for i in range(n_records):
            # Find where the record stops having data
            mask = ~np.isnan(im_at_edp_matrix[i, :])
            if np.any(mask):
                last_valid_idx = np.where(mask)[0][-1]
                last_valid_im = im_at_edp_matrix[i, last_valid_idx]

                # Fill the remaining EDP range with the collapse IM
                im_at_edp_matrix[i, last_valid_idx+1:] = last_valid_im

        # Calculate percentiles
        # Now that collapse region is filled, no All-NaN slices
        # unless a record never started at all.
        median_ida_im = np.nanmedian(im_at_edp_matrix, axis=0)
        p16_ida_im = np.nanpercentile(im_at_edp_matrix, 16, axis=0)
        p84_ida_im = np.nanpercentile(im_at_edp_matrix, 84, axis=0)

        # Fragility Fitting (MLE for Censored Data)
        im_max = np.nanmax(im_matrix)
        thetas = []
        sigmas_rec2rec = []
        sigmas_build2build = []
        sigmas_ds = []

        for threshold in damage_thresholds:
            thresh_idx = np.argmin(np.abs(edp_range - threshold))
            capacities = im_at_edp_matrix[:, thresh_idx]
            num_collapsed = np.sum(~np.isnan(capacities))

            if num_collapsed == n_records:
                ln_cap = np.log(capacities)
                theta = np.exp(np.mean(ln_cap))
                beta_rec = np.std(ln_cap, ddof=1)
            else:
                def log_likelihood(params):
                    t, b = params
                    if t <= 0 or b <= 0:
                        return 1e10
                    collapsed = capacities[~np.isnan(capacities)]
                    term1 = np.sum(np.log(stats.norm.pdf(
                        (np.log(collapsed)-np.log(t))/b)/(collapsed*b)))
                    term2 = ((n_records - num_collapsed) * np.log(
                        1 - stats.norm.cdf(
                            (np.log(im_max) - np.log(t)) / b)))
                    return -(term1 + term2)

                sol = optimize.minimize(
                    log_likelihood, [im_max, 0.4], method='Nelder-Mead')
                theta, beta_rec = sol.x[0], sol.x[1]

            thetas.append(theta)
            sigmas_rec2rec.append(beta_rec)
            sigmas_build2build.append(sigma_build2build)
            sigmas_ds.append(sigma_ds)

        # Generate Probabilities of Exceedance with Rotation Option
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        betas_total = []

        for i, threshold in enumerate(damage_thresholds):
            theta = thetas[i]
            beta_rec = sigmas_rec2rec[i]

            if fragility_rotation:
                # Combined uncertainty isn't a simple SRSS in rotation,
                # but we report total for consistency in dict
                betas_total.append(
                    np.sqrt(beta_rec**2 + sigma_build2build**2 + sigma_ds**2))
                (_, _, poes[:, i]) = (
                    self.calculate_rotated_fragility(
                        theta,
                        rotation_percentile,
                        beta_rec,
                        sigma_build2build,
                        sigma_ds,
                        intensities))
            else:
                beta_total = np.sqrt(
                    beta_rec**2 + sigma_build2build**2 + sigma_ds**2)
                betas_total.append(beta_total)
                poes[:, i] = self.calculate_lognormal_fragility(
                    theta, beta_total)

        # 5. Construct the final nested dictionary (Cloud-style)
        ida_dict = {
            'ida_inputs': {
                'target_edps': edp_range,
                'raw_curves': raw_curves,
                'damage_thresholds': damage_thresholds,
                'im_matrix': im_matrix,
                'n_records': n_records,
                'im_max_analyzed': im_max
            },

            'fragility': {
                'fragility_method': 'mle_ida_censored',
                'intensities': intensities,
                'poes': poes,
                'medians': thetas,
                'sigma_record2record': sigmas_rec2rec,
                'sigma_build2build': sigmas_build2build,
                'sigma_ds': sigmas_ds,
                'betas_total': betas_total,
                'rotation_active': fragility_rotation,
                'rotation_percentile': (
                    rotation_percentile if fragility_rotation
                    else None)
            },

            'stats': {
                'fitted_edps': edp_range,
                'median_im': median_ida_im,
                'p16_im': p16_ida_im,
                'p84_im': p84_ida_im
            }
        }

        return ida_dict

    def calculate_vulnerability_function(self,
                                         poes,
                                         consequence_model,
                                         cov_consequence=None,
                                         uncertainty=True,
                                         method=None,
                                         intensities=np.round(
                                             np.geomspace(
                                                 0.05, 10.0, 50),
                                             3)):
        """
        Compute a vulnerability function (mean loss ratio and associated
        uncertainty) by convolving fragility functions with a consequence
        (damage-to-loss) model.

        The expected loss ratio is computed as the convolution of mutually
        exclusive damage-state probabilities with damage-to-loss ratios.
        Uncertainty in the loss ratio conditional on intensity measure
        level (Loss | IM) can be computed either explicitly using the law
        of total variance or via an empirical Silva-type envelope.

        Parameters
        ----------
        poes : ndarray, shape (n_IM, n_DS)
            Probabilities of exceedance of each damage state conditional
            on the intensity measure level (P[DS >= k | IM]). Damage
            states must be ordered from least to most severe.

        consequence_model : array-like, length n_DS
            Mean damage-to-loss ratios associated with each damage state.
            Values must lie in the interval [0, 1].

        cov_consequence : array-like, length n_DS, optional
            Coefficient of variation of the damage-to-loss ratio for each
            damage state. Required when ``method="explicit"``. Each entry
            represents the conditional uncertainty of loss given the
            damage state.

        uncertainty : bool, optional
            Flag indicating whether to compute uncertainty (coefficient of
            variation) of the loss ratio conditional on IM. If False, the
            COV column is still returned and filled with zeros.
            Default is True.

        method : {"explicit", "silva"}, optional
            Method used to compute uncertainty when ``uncertainty=True``.

            - "explicit" (default):
              Computes uncertainty using the law of total variance,
              accounting for both damage-state mixing and uncertainty
              within each damage state. Requires ``cov_consequence``.

            - "silva":
              Computes uncertainty using a Silva-type empirical envelope
              based only on the mean loss ratio.

            If ``uncertainty=True`` and ``method=None``, the method
            defaults to "explicit".

        intensities : ndarray, optional
            Intensity measure levels corresponding to the rows of
            ``poes``. Default is a geometric sequence between 0.05
            and 10.0.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with the following columns:

            - ``IML``  : Intensity measure level
            - ``Loss`` : Expected loss ratio at the given IML
            - ``COV``  : Coefficient of variation of the loss ratio
              at the given IML

            The ``COV`` column is always returned. If
            ``uncertainty=False``, it contains zeros.

        Raises
        ------
        Exception
            If the dimensions of ``poes``, ``consequence_model``, or
            ``cov_consequence`` are inconsistent, or if
            ``method="explicit"`` is selected without providing
            ``cov_consequence``.

        Notes
        -----
        For the explicit uncertainty method, the variance of the loss
        ratio is computed using the law of total variance:

        Var(LR | IM) = sum_k p_k [ sigma_k^2 + (mu_k - mu)^2 ]

        where:
            - p_k    is the probability of being in damage state k
              given IM,
            - mu_k   is the mean loss ratio for damage state k,
            - sigma_k^2 is the variance of the loss ratio within
              damage state k,
            - mu     is the mean loss ratio at the given IM.

        This formulation is consistent with performance-based earthquake
        engineering (PBEE) frameworks and produces physically meaningful,
        IM-dependent uncertainty.
        """

        def calculate_sigma_silva(loss):
            """
            Helper function to calculate uncertainty in the loss
            estimates based on the method proposed in Silva (2019),
            which incorporates the sigma (standard deviation) for loss
            ratios within seismic vulnerability functions.

            This method computes the sigma loss ratio for expected loss
            ratios and also estimates the parameters of a beta
            distribution (coefficients a and b), which describe the
            uncertainty and variability in the loss estimates. The
            formula used is derived from seismic vulnerability research.

            Parameters:
            -----------
            loss : list or array
                A list or array of expected loss ratios. The expected
                loss ratio represents the proportion of the building's
                value that is expected to be lost due to an earthquake
                event, ranging from 0 to 1.

            Returns:
            --------
            sigma_loss_ratio : list or array
                The calculated uncertainty (sigma) associated with the
                mean loss ratio for each input loss value. The sigma
                loss ratio represents the variability of the loss
                estimates and is computed based on the loss ratios
                provided.

            a_beta_dist : list or array
                The coefficient 'a' of the beta distribution for each
                loss ratio. This parameter represents the shape of the
                beta distribution and is used to model the uncertainty
                in the loss estimates.

            b_beta_dist : list or array
                The coefficient 'b' of the beta distribution for each
                loss ratio. This parameter also represents the shape of
                the beta distribution, complementing the coefficient 'a'
                to fully describe the distribution's behavior.

            References:
            ----------
            1) Silva, V. (2019) "Uncertainty and correlation in seismic
            vulnerability functions of building classes." Earthquake
            Spectra. DOI: 10.1193/013018eqs031m.

            """
            sigma_loss_ratio = np.where(
                loss == 1e-8, 1e-8, np.where(
                    loss == 1, 1,
                    0.5 * np.sqrt(
                        loss * (-0.7 - 2 * loss
                                + np.sqrt(6.8 * loss + 0.5)))))
            a_beta_dist = np.zeros(loss.shape)
            b_beta_dist = np.zeros(loss.shape)

            return sigma_loss_ratio, a_beta_dist, b_beta_dist

        # Default behavior
        if uncertainty and method is None:
            method = "explicit"

        # Consistency checks
        n_im, n_ds = poes.shape
        if len(consequence_model) != n_ds:
            raise Exception(
                'ERROR! Mismatch between fragility and consequence models!')

        if len(intensities) != n_im:
            raise Exception(
                'ERROR! Mismatch between number of IMLs and fragility models!')

        if uncertainty and method == "explicit":
            if cov_consequence is None:
                raise Exception(
                    'ERROR! Explicit uncertainty method requires '
                    'cov_consequence.')
            if len(cov_consequence) != n_ds:
                raise Exception(
                    'ERROR! Length of cov_consequence must match '
                    'consequence_model.')

        # Convert to arrays
        mu_k = np.asarray(consequence_model, dtype=float)
        if cov_consequence is not None:
            cov_k = np.asarray(cov_consequence, dtype=float)
            var_k = (cov_k * mu_k) ** 2
        else:
            var_k = np.zeros_like(mu_k)

        # Damage-state probabilities from POEs
        p_ds = np.zeros_like(poes)

        for j in range(n_ds):
            if j == n_ds - 1:
                p_ds[:, j] = poes[:, j]
            else:
                p_ds[:, j] = poes[:, j] - poes[:, j + 1]

        # Mean loss ratio (convolution)
        loss = np.dot(p_ds, mu_k)

        # Initialize COV
        cov = np.zeros_like(loss)
        if uncertainty:
            if method.lower() == "explicit":
                # Law of total variance
                diff = mu_k - loss[:, None]
                var_loss = np.sum(p_ds * (var_k + diff ** 2), axis=1)
                cov = np.sqrt(var_loss) / (loss + 1e-12)
            elif method.lower() == "silva":
                # Semi-empirical derivatoin
                for i, mu in enumerate(loss):
                    sigma_loss_ratio, _, _ = calculate_sigma_silva(mu)
                    cov[i] = np.min(
                        [sigma_loss_ratio / mu,
                         0.90 * np.sqrt(mu * (1 - mu)) / mu])
            else:
                raise Exception(f"ERROR! Unknown uncertainty method: {method}")

        # Output
        df = pd.DataFrame({'IML': intensities,
                           'Loss': loss,
                           'COV': cov})

        return df

    def calculate_average_annual_damage_probability(self,
                                                    fragility_array,
                                                    hazard_array,
                                                    return_period=1,
                                                    max_return_period=5000):
        """
        Calculate the Average Annual Damage State Probability (AADP)
        based on fragility and hazard curves.

        This function estimates the average annual probability of damage
        states occurring over a given return period, using the fragility
        curve (which relates intensity measure levels to damage state
        probabilities) and the hazard curve (which relates intensity
        measure levels to annual rates of exceedance).

        The calculation integrates the product of the fragility function
        and the hazard curve over the specified range of intensity measure
        levels, accounting for the return period and a maximum return
        period threshold.

        Parameters
        ----------
        fragility_array : 2D array
            A 2D array where the first column contains intensity measure
            levels, and the second column contains the corresponding
            probabilities of exceedance for each intensity level.

        hazard_array : 2D array
            A 2D array where the first column contains intensity measure
            levels, and the second column contains the annual rates of
            exceedance (i.e., the probability of exceedance per year)
            for each intensity level.

        return_period : float, optional, default=1
            The return period used to scale the hazard rate. This is the
            time span (in years) over which the average annual damage
            probability is calculated. A typical value is 1 year, but
            longer periods can be used for multi-year assessments.

        max_return_period : float, optional, default=5000
            The maximum return period threshold used to filter out very
            low hazard rates. The hazard curve is truncated to include
            only intensity levels with exceedance rates above this
            threshold.

        Returns
        -------
        average_annual_damage_probability : float
            The average annual damage state probability, calculated by
            integrating the product of the fragility function and the
            hazard curve over the given intensity measure levels.

        """

        # Ensure arrays are sorted by Intensity (Column 0)
        hazard_array = hazard_array[hazard_array[:, 0].argsort()]
        fragility_array = fragility_array[fragility_array[:, 0].argsort()]

        # Filter hazard based on return period threshold
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]

        # Need at least 2 points to calculate a rate difference
        if len(hazard_array) < 2:
            return 0.0

        # Compute midpoints and rate of occurrences (|d_lambda|)
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        # abs ensures positive probability mass regardless of sort order
        rate_occ = np.abs(np.diff(hazard_array[:, 1])) / return_period

        # Define fragility curve with dynamic upper boundary
        # We assume Probability=0 at IM=0 and Probability=1 at high IM
        upper_im_bound = max(20.0, fragility_array[:, 0].max() * 1.5)
        curve_imls = np.hstack(
            ([0.0], fragility_array[:, 0], [upper_im_bound]))
        curve_ordinates = np.hstack(([0.0], fragility_array[:, 1], [1.0]))

        # Interpolate and Integrate
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)

        # Result: Sum of (Probability of Damage * Frequency of Occurrence)
        return np.dot(interpolated_values, rate_occ)

    def calculate_average_annual_loss(self,
                                      vulnerability_array,
                                      hazard_array,
                                      return_period=1,
                                      max_return_period=5000):
        """
        Calculate the Average Annual Loss Ratio (AALR) based on
        vulnerability and hazard curves.

        This function estimates the average loss ratio occurring over a
        given return period (typically annual where return_period = 1),
        using the vulnerability curve (which relates intensity measure
        levels to an expected loss ratio) and the hazard curve (which
        relates intensity measure levels to annual rates of exceedance).

        The calculation integrates the product of the vulnerability
        function and the hazard curve over the specified range of
        intensity measure levels, accounting for the return period and
        a maximum return period threshold.

        Parameters
        ----------
        vulnerability_array : 2D array
            A 2D array where the first column contains intensity measure
            levels, and the second column contains the corresponding
            expected loss ratios for each intensity level.

        hazard_array : 2D array
            A 2D array where the first column contains intensity measure
            levels, and the second column contains the annual rates of
            exceedance (i.e., the probability of exceedance per year)
            for each intensity level.

        return_period : float, optional, default=1
            The return period used to scale the hazard rate. This is the
            time span (in years) over which the average annual damage
            probability is calculated. A typical value is 1 year, but
            longer periods can be used for multi-year assessments.

        max_return_period : float, optional, default=5000
            The maximum return period threshold used to filter out very
            low hazard rates. The hazard curve is truncated to include
            only intensity levels with exceedance rates above this
            threshold.

        Returns
        -------
        average_annual_loss_ratio : float
            The average annual loss ratio, calculated by integrating
            the product of the vulnerability function and the hazard
            curve over the given intensity measure levels.

        """
        # Ensure hazard data is sorted by Intensity Measure (IM)
        hazard_array = hazard_array[hazard_array[:, 0].argsort()]
        vulnerability_array = vulnerability_array[
            vulnerability_array[:, 0].argsort()]

        # Filter hazard based on max return period (min frequency threshold)
        min_rate_threshold = return_period / max_return_period
        hazard_filtered = hazard_array[hazard_array[:, 1]
                                       >= min_rate_threshold]

        if len(hazard_filtered) < 2:
            return 0.0

        # Calculate midpoints (mean IMs) and the change in rates (d_lambda)
        # abs difference gives the rate of occurrence for each interval
        mean_imls = (hazard_filtered[:-1, 0] + hazard_filtered[1:, 0]) / 2
        rate_occ = np.abs(np.diff(hazard_filtered[:, 1])) / return_period

        # Prepare vulnerability curve for interpolation
        # Anchoring the curve at IM=0 (Loss=0) and a high IM cap (Loss=1.0)
        v_im = vulnerability_array[:, 0]
        v_loss = vulnerability_array[:, 1]

        curve_imls = np.concatenate(
            ([0.0], v_im, [max(20.0, v_im.max() * 1.5)]))
        curve_ordinates = np.concatenate(([0.0], v_loss, [1.0]))

        # Interpolate vulnerability at the hazard midpoints
        interpolated_losses = np.interp(mean_imls, curve_imls, curve_ordinates)

        # Final Integration (Dot product of Losses and Probabilities)
        average_annual_loss = np.dot(interpolated_losses, rate_occ)

        return average_annual_loss

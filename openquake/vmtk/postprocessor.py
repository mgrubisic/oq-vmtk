import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats, optimize
from scipy.optimize import curve_fit
from scipy.stats import norm, lognorm
from scipy.interpolate import interp1d
from statsmodels.miscmodels.ordinal_model import OrderedModel

class postprocessor():

    """
    Class for post-processing results of nonlinear time-history analysis, including fragility and vulnerability analysis.

    This class provides methods to compute fragility functions, perform cloud and multiple stripe analyses,
    and calculate vulnerability functions and average annual losses. It supports various fragility fitting
    methods, including lognormal, probit, logit, and ordinal models. The class also includes functionality
    to handle uncertainty and variability in the analysis.

    Methods
    -------
    calculate_lognormal_fragility(theta, sigma_record2record, sigma_build2build=0.30, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Computes the probability of exceeding a damage state using a lognormal cumulative distribution function (CDF).

    calculate_rotated_fragility(theta, percentile, sigma_record2record, sigma_build2build=0.30, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Calculates a rotated fragility function based on a lognormal CDF, adjusting the median intensity to align with a specified target percentile.

    calculate_glm_fragility(imls, edps, damage_thresholds, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_method='logit')
        Computes non-parametric fragility functions using Generalized Linear Models (GLM) with either a Logit or Probit link function.

    calculate_ordinal_fragility(imls, edps, damage_thresholds, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Fits an ordinal (cumulative) probit model to estimate fragility curves for different damage states.

    do_modified_cloud_analysis(imls, edps, damage_thresholds, lower_limit, censored_limit, sigma_build2build = 0.3, intensities = np.geomspace(0.05, 10, 50), n_bootstrap=200, random_seed=None, fragility_rotation = False, rotation_percentile = 0.10, fragility_method ='lognormal'))
        Perform a modified cloud analysis (i.e., accounting for collapse and non-collapse cases) to assess fragility functions for a set of engineering demand parameters (EDPs) and intensity measure levels (IMLs).

    do_multiple_stripe_analysis(imls, edps, damage_thresholds, sigma_build2build=0.3, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_rotation=False, rotation_percentile=0.10)
        Perform maximum likelihood estimation (MLE) for fragility curve fitting following a multiple stripe analysis.

    calculate_sigma_loss(loss)
        Calculate the uncertainty in the loss estimates based on the method proposed in Silva (2019).

    get_vulnerability_function(poes, consequence_model, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), uncertainty=True)
        Calculate the vulnerability function given the probabilities of exceedance and a consequence model.

    calculate_average_annual_damage_probability(fragility_array, hazard_array, return_period=1, max_return_period=5000)
        Calculate the Average Annual Damage State Probability (AADP) based on fragility and hazard curves.

    calculate_average_annual_loss(vulnerability_array, hazard_array, return_period=1, max_return_period=5000)
        Calculate the Average Annual Loss (AAL) based on vulnerability and hazard curves.

    """

    def __init__(self):
        pass

    def calculate_lognormal_fragility(self,
                                      theta,
                                      sigma_record2record,
                                      sigma_build2build = 0.30,
                                      intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Computes the probability of exceeding a damage state using a lognormal cumulative distribution function (CDF).

        Parameters
        ----------
        theta : float
            The median seismic intensity corresponding to an EDP-based damage threshold.

        sigma_record2record : float
            The logarithmic standard deviation representing record-to-record variability.

        sigma_build2build : float, optional
            The logarithmic standard deviation representing building-to-building (or model) variability.
            Default value is 0.30.

        intensities : array-like, optional
            The set of intensity measure (IM) levels for which exceedance probabilities will be computed.
            Default is a geometric sequence from 0.05 to 10.0 with 50 points.

        Returns
        -------
        poes : numpy.ndarray
            An array of exceedance probabilities corresponding to each intensity measure in `intensities`.

        References
        -----
        1) Baker JW. Efficient Analytical Fragility Function Fitting Using Dynamic Structural Analysis.
        Earthquake Spectra. 2015;31(1):579-599. doi:10.1193/021113EQS025M

        2) Singhal A, Kiremidjian AS. Method for probabilistic evaluation of seismic structural damage.
        Journal of Structural Engineering 1996; 122: 1459–1467. DOI:10.1061/(ASCE)0733-9445(1996)122:12(1459)

        3) Lallemant, D., Kiremidjian, A., and Burton, H. (2015), Statistical procedures for developing
        earthquake damage fragility curves. Earthquake Engng Struct. Dyn., 44, 1373–1389. doi: 10.1002/eqe.2522.

        4) Bird JF, Bommer JJ, Bray JD, Sancio R, Spence RJS. Comparing loss estimation with observed damage in a zone
        of ground failure: a study of the 1999 Kocaeli Earthquake in Turkey. Bulletin of Earthquake Engineering 2004; 2:
        329–360. DOI: 10.1007/s10518-004-3804-0

        """

        # Calculate the total uncertainty
        beta_total = np.sqrt(sigma_record2record**2+sigma_build2build**2)

        # Calculate probabilities of exceedance for a range of intensity measure levels
        return lognorm.cdf(intensities, s=beta_total, loc=0, scale=theta)

    def calculate_rotated_fragility(self,
                                    percentile,
                                    theta,
                                    sigma_record2record,
                                    sigma_build2build = 0.30,
                                    intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Calculates a rotated fragility function based on a lognormal cumulative distribution function (CDF),
        adjusting the median intensity to align with a specified target percentile.

        This function modifies the median intensity based on the desired target percentile and total uncertainty
        (considering both record-to-record variability and modeling variability). The resulting rotated fragility
        curve represents the damage exceedance probabilities for a range of intensity measure levels, as defined
        by the lognormal distribution.

        ----------
        Parameters
        ----------
        percentile : float
            The target percentile for fragility function rotation. This value corresponds to the desired
            percentile (e.g., 0.2 corresponds to the 20th percentile of the fragility curve). The curve is adjusted
            such that this percentile aligns with the calculated fragility function.

        theta : float
            The median seismic intensity corresponding to the edp-based damage threshold.

        sigma_record2record : float
            The uncertainty associated with record-to-record variability in the seismic records used to derive the fragility.

        sigma_build2build : float, optional, default=0.30
            The uncertainty associated with modeling variability between different buildings or building types.

        intensities : array-like, optional, default=np.round(np.geomspace(0.05, 10.0, 50), 3)
            A list or array of intensity measure levels at which to evaluate the fragility function, typically representing
            seismic intensity levels (e.g., spectral acceleration). The default is a geometric space ranging from 0.05 to 10.0.

        -------
        Returns
        -------
        theta_prime : float
            The new median intensity after the rotation based on the specified percentile.

        beta_total : float
            The total standard deviation of the lognormal distribution, calculated from both record-to-record and
            building-to-building (modelling) uncertainties.

        poes : array-like
            The probabilities of exceedance (fragility values) corresponding to the input intensity measure levels.
            This is the lognormal CDF evaluated at the given intensities with the rotated median and combined uncertainty.

        ----------
        References
        ----------
        1) Porter, K. (2017), "When Addressing Epistemic Uncertainty in a Lognormal Fragility Function,
        How Should One Adjust the Median?", Proceedings of the 16th World Conference on Earthquake Engineering
        (16WCEE), Santiago, Chile.

        """

        # Calculate the combined logarithmic standard deviation (total uncertainty)
        beta_total = np.sqrt(sigma_record2record**2 + sigma_build2build**2)

        # Adjust the median intensity based on the target percentile
        theta_prime = theta * np.exp(-stats.norm.ppf(percentile) * (beta_total - sigma_record2record))

        # Calculate and return the rotated lognormal CDF (probabilities of exceedance) for the given intensities
        return theta_prime, beta_total, stats.lognorm(s=beta_total, scale=theta_prime).cdf(intensities)


    def calculate_glm_fragility(self,
                                imls,
                                edps,
                                damage_thresholds,
                                intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                fragility_method = 'logit'):

        """
        Computes non-parametric fragility functions using Generalized Linear Models (GLM) with
        either a Logit or Probit link function.

        Parameters:
        -----------
        imls : array-like
            Intensity Measure Levels (IMLs) corresponding to each observation.

        edps : array-like
            Engineering Demand Parameters (EDPs) representing structural response values.

        damage_thresholds : array-like
            List of thresholds defining different damage states.

        intensities : array-like, optional
            Intensity measure values at which probabilities of exceedance (PoEs) are evaluated.
            Defaults to np.round(np.geomspace(0.05, 10.0, 50), 3).

        fragility_method : str, optional
            Specifies the GLM model to be used for fragility function fitting.
            Options:
            - 'logit' (default): Uses a logistic regression model.
            - 'probit': Uses a probit regression model.

        Returns:
        --------
        poes : ndarray
            A 2D array where each column represents the probability of exceeding a
            specific damage state at each intensity level.

        References:
        ------
        1) Charvet, I., Ioannou, I., Rossetto, T., Suppasri, A., and Imamura, F.: Empirical fragility
        assessment of buildings affected by the 2011 Great East Japan tsunami using improved statistical models,
        Nat. Hazards, 73, 951–973, 2014. 

        2) Lahcene, E., Ioannou, I., Suppasri, A., Pakoksung, K., Paulik, R., Syamsidik, S., Bouchette, F.,
        and Imamura, F.: Characteristics of building fragility curves for seismic and non-seismic tsunamis:
        case studies of the 2018 Sunda Strait, 2018 Sulawesi–Palu, and 2004 Indian Ocean tsunamis,
        Nat. Hazards Earth Syst. Sci., 21, 2313–2344, https://doi.org/10.5194/nhess-21-2313-2021, 2021.

        3) Lallemant, D., Kiremidjian, A., and Burton, H. (2015), Statistical procedures for developing
        earthquake damage fragility curves. Earthquake Engng Struct. Dyn., 44, 1373–1389. doi: 10.1002/eqe.2522.

        4) Jalayer, F., Ebrahamian, H., Trevlopoulos, K., and Bradley, B. (2023). Empirical tsunami fragility modelling
        for hierarchical damage levels. Natural Hazards and Earth System Sciences, 23(2), 909–931.
        https://doi.org/10.5194/nhess-23-909-2023

        """

        # Create probabilities of exceedance array
        poes = np.zeros((len(intensities),len(damage_thresholds)))

        for ds, current_threshold in enumerate(damage_thresholds):

            # Count exceedances
            exceedances = [1 if edp>damage_thresholds[ds] else 0 for edp in edps]

            # Assemble dictionary containing log of IMs and binary damage state assignments
            data = {'IM': np.log(imls),
                    'Damage': exceedances}

            # Create DataFrame
            df = pd.DataFrame(data)

            # Add a constant for the intercept term
            X = sm.add_constant(df['IM'])
            y = df['Damage']

            if fragility_method.lower() == 'probit':

                # Fit the Probit GLM model
                probit_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
                probit_results = probit_model.fit()

                # Generate a range of IM values for plotting
                log_IM_range = np.log(intensities)
                X_range = sm.add_constant(log_IM_range)

                # Predict probabilities using the Probit GLM model
                poes[:,ds] = probit_results.predict(X_range)

            elif fragility_method.lower() == 'logit':

                # Fit the Logit GLM model
                logit_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
                logit_results = logit_model.fit()

                # Generate a range of IM values for plotting
                log_IM_range = np.log(intensities)
                X_range = sm.add_constant(log_IM_range)

                # Predict probabilities using the Probit GLM model
                poes[:,ds] = logit_results.predict(X_range)

        return poes

    def calculate_ordinal_fragility(self,
                                    imls,
                                    edps,
                                    damage_thresholds,
                                    intensities=np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Fits an ordinal (cumulative) probit model to estimate fragility curves for different damage states.

        This function estimates the probability of exceeding various damage states using an ordinal
        regression model based on observed Engineering Demand Parameters (EDPs) and corresponding
        Intensity Measure Levels (IMLs).

        Parameters
        ----------
        imls : array-like
            Intensity measure levels corresponding to the observed EDPs.

        edps : array-like
            Engineering Demand Parameters (EDPs) representing structural responses.

        damage_thresholds : array-like
            Damage state thresholds for classifying exceedance levels.

        intensities : array-like, optional
            Intensity measure levels for which fragility curves are evaluated (default: np.geomspace(0.05, 10.0, 50)).

        Returns
        -------
        poes : numpy.ndarray
            A 2D array of exceedance probabilities (CDF values) for each intensity level.
            Shape: (len(intensities), len(damage_thresholds) + 1), where the last column
            represents the probability of exceeding the highest damage state.

        References
        -----
        1) Lallemant, D., Kiremidjian, A., and Burton, H. (2015), Statistical procedures for developing
        earthquake damage fragility curves. Earthquake Engng Struct. Dyn., 44, 1373–1389. doi: 10.1002/eqe.2522.

        2) Nguyen, M. and Lallemant, D. (2022), Order Matters: The Benefits of Ordinal Fragility Curves for Damage and Loss Estimation. Risk Analysis, 42: 1136-1148. https://doi.org/10.1111/risa.13815

        """

        # Create probabilities of exceedance array
        poes = np.zeros((len(intensities), len(damage_thresholds) + 1))  # +1 to include the highest damage state

        # Initialize damage state assignments
        damage_states = np.zeros(len(edps), dtype=int)

        # Loop over each EDP and determine the highest exceeded damage state
        for i, edp in enumerate(edps):
            exceeded = np.where(edp > damage_thresholds)[0]  # Indices where EDP exceeds thresholds
            damage_states[i] = exceeded[-1] + 1 if exceeded.size > 0 else 0  # Assign highest exceeded state (0-based)

        # Assemble DataFrame containing log(IM) and damage state assignment
        df = pd.DataFrame({'IM': np.log(imls), 'Damage State': damage_states})

        # Fit the Cumulative Probit Model
        X_ordinal = df[['IM']]
        y_ordinal = df['Damage State']

        # Create and fit the OrderedModel
        ordinal_model = OrderedModel(y_ordinal, X_ordinal, distr='probit')
        ordinal_results = ordinal_model.fit(method='bfgs', disp=False)  # Silent optimization

        # Generate log-transformed IM values for prediction
        log_IM_range = np.log(intensities)
        X_range_ordinal = pd.DataFrame({'IM': log_IM_range})

        # Predict probabilities for each damage state (PMF)
        pmf_values = ordinal_results.predict(X_range_ordinal)  # Shape: (len(intensities), num_damage_states)

        # Convert PMF to CDF (probabilities of exceedance) by cumulative sum across damage states
        poes = 1 - np.cumsum(pmf_values, axis=1)  # Cumulative sum along damage state axis

        return poes.values

    def do_modified_cloud_analysis(self,
                                   imls,
                                   edps,
                                   damage_thresholds,
                                   lower_limit,
                                   censored_limit,
                                   sigma_build2build = 0.3,
                                   intensities = np.geomspace(0.05, 10, 50),
                                   n_bootstrap=200,
                                   random_seed=None,
                                   fragility_rotation = False,
                                   rotation_percentile = 0.10,
                                   fragility_method ='lognormal'):

        """
        Perform a Modified Cloud Analysis (MCA) to derive seismic fragility functions.
        This method extends classical cloud analysis by incorporating logistic regression
        to account for structural collapse cases and using bootstrapping to ensure
        statistical stability. It supports lognormal, ordinal, and GLM-based fragility fitting.

        Parameters
        ----------
        imls : array_like
            Intensity Measure Levels (e.g., Sa, AvgSA) from the cloud of data.
        edps : array_like
            Engineering Demand Parameters (e.g., maximum interstory drift) from the cloud.
        damage_thresholds : list of float
            The demand-based thresholds defining the onset of different damage states.
        lower_limit : float
            The EDP value below which data is ignored for regression (demand is
            considered negligible for damage).
        censored_limit : float
            The "Collapse" threshold. EDP values above this are treated as collapse
            instances in the logistic regression.
        sigma_build2build : float, optional
            Additional modeling uncertainty (building-to-building variability).
            Default is 0.3.
        intensities : np.array, optional
            The seismic intensity range over which to evaluate the fragility functions.
            Default is a geometric space from 0.05 to 10.
        n_bootstrap : int, optional
            Number of bootstrap samples to draw for statistical stability. Default is 200.
        random_seed : int, optional
            Seed for reproducibility of the bootstrap sampling. Default is None.
        fragility_rotation : float, optional
            Parameter for rotating fragility functions around a specific percentile
            to adjust for target reliability. Default is 0.1.
        fragility_method : {'lognormal', 'ordinal', 'probit', 'logit'}, optional
            The methodology used to fit the fragility functions. Default is 'lognormal'.

        Returns
        -------
        cloud_dict : dict
            A comprehensive results dictionary containing:
            - 'cloud inputs': The original and filtered IM and EDP data.
            - 'fragility': Fitted parameters (medians, betas) and exceedance probabilities (PoEs).
            - 'regression': Mean coefficients (b0, b1, sigma) from the log-log regression.
            - 'bootstraps': Raw iteration data for slope, intercept, and uncertainty.
            - 'raw_data_split': Data separated into collapse and non-collapse categories.

        Notes
        -----
        The 'lognormal' method specifically implements a dual-regression approach:
        1.  **Linear Regression**: Performed in log-log space on non-collapse data
            (log(EDP) = log(a) + b * log(IM)).
        2.  **Logistic Regression**: Used to predict the probability of collapse P(C|IM).
        3.  **Total Fragility**: Calculated as P(DS|IM) = P(DS|NC,IM) * P(NC|IM) + P(C|IM).
        """

        def cond_fragility(x, a, b):
            """
            Helper function that fits the conditioned fragility functions (i.e.,
            considering collapse and non-collapse cases)
            """
            return (1-np.exp(-a*x))**b

        def prepare_mca_data(imls, edps, collapse_limit, bootstrap=True):
            """
            Helper function that standardizes the cloud input parameters and splits IMs
            and EDPs into collapse and non-collapse cases. Then implements bootstrapping
            to ensure stability in Logistic Regression
            """
            # Ensure inputs are arrays
            imls    = np.array(imls)
            edps    = np.array(edps)
            n_total = len(imls)

            # Identify indicies where collapse occurs in the original dataset
            mask_coll_ori = edps > collapse_limit
            im_coll_ori   = imls[mask_coll_ori]
            edp_coll_ori  = edps[mask_coll_ori]
            npt_ori       = len(im_coll_ori)

            # Bootstrap sampling
            if bootstrap:
                idx_boot   = np.random.randint(0, n_total, size=n_total)
                sample_im  = imls[idx_boot]
                sample_edp = edps[idx_boot]
            else:
                sample_im  = imls.copy()
                sample_edp = edps.copy()

            # Standardization: stability check
            npts_sample_coll = np.sum(sample_edp > collapse_limit)
            target_min       = math.ceil(0.5*npt_ori)
            if npts_sample_coll < target_min:
                add_count = target_min - npts_sample_coll
                # Append original collapse points to stabilize
                sample_im  = np.concatenate([sample_im, im_coll_ori[:, add_count]])
                sample_edp = np.concatenate([sample_edp, edp_coll_ori[:, add_count]])

            # Final split
            is_coll = sample_edp > collapse_limit # Indices of collapse instances

            im_nc  = sample_im[~is_coll]  # Non-collapse IMs
            edp_nc = sample_edp[~is_coll] # Non-collapse EDPs
            im_c   = sample_im[is_coll]   # Collapse IMs

            return im_nc, edp_nc, im_c

        # Compute exceedance probabilities using the specified fragility method
        if fragility_method in ['probit', 'logit']:

            # Get the probabilities of exceedance
            poes = self.calculate_glm_fragility(imls, edps, damage_thresholds, fragility_method=fragility_method)

            # Compute equivalent lognormal fragility parameters from the GLM model
            thetas               = [np.interp(0.50, poes[:,ds], intensities)  for ds in range(len(damage_thresholds))]                                                                       # Dummy Median intensities
            sigmas_record2record = [np.abs(0.50*(np.log(np.interp(0.84,poes[:,ds], intensities))-np.log(np.interp(0.16,poes[:,ds], intensities)))) for ds in range(len(damage_thresholds))]  # Dummy Record-to-record variability
            sigmas_build2build   = np.full(len(damage_thresholds), sigma_build2build)                                                                                                        # Modelling uncertainty
            betas_total          = [np.sqrt(sigma_record2record**2+sigma_build2build**2) for sigma_record2record, sigma_build2build in zip (sigmas_record2record, sigmas_build2build)]       # Dummy Total Dispersion

            # Create the dictionary
            cloud_dict = {
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {'imls'             : imls,               # Store the intensity measure levels (cloud)
                                 'edps'             : edps,               # Store the engineering demand parameters (cloud)
                                 'lower_limit'      : None,               # Store the lower limit for censored regression
                                 'upper_limit'      : None,               # Store the upper limit for censored regression
                                 'damage_thresholds': damage_thresholds}, # Store the demand-based damage state thresholds

                # Add a nested dictionary for fragility functions parameters
                'fragility': {'fragility_method'   : fragility_method.lower(), # Store the fragility fitting methodology
                              'intensities'        : intensities,              # Store the intensities used for sampling fragility functions
                              'poes'               : poes,                     # Store the probabilities of damage state exceedance
                              'medians'            : thetas,                   # Store the median seismic intensities
                              'sigma_record2record': sigmas_record2record,     # Store the record-to-record variability
                              'sigma_build2build'  : sigmas_build2build,       # Store the modelling uncertainty
                              'betas_total'        : betas_total},             # Store the total variability accounting for record-to-record and modelling uncertainties

                # Add a nested dictionary for regression coefficients
                'regression': {'b1'      : None,   # Store 'b1' coefficient
                               'b0'      : None,   # Store 'b0' coefficient
                               'sigma'   : None,   # Store 'sigma' value
                               'fitted_x': None,   # Store the fitted x-values
                               'fitted_y': None}   # Store the fitted y-values
                }


        elif fragility_method.lower() == 'ordinal':

            # Compute exceedance probabilities using the specified fragility method
            poes = self.calculate_ordinal_fragility(imls, edps, damage_thresholds)

            # Compute equivalent lognormal fragility parameters from the GLM model
            thetas               = [np.interp(0.50, poes[:,ds], intensities)  for ds in range(len(damage_thresholds))]                                                                       # Dummy Median intensities
            sigmas_record2record = [np.abs(0.50*(np.log(np.interp(0.84,poes[:,ds], intensities))-np.log(np.interp(0.16,poes[:,ds], intensities)))) for ds in range(len(damage_thresholds))]  # Dummy Record-to-record variability
            sigmas_build2build   = np.full(len(damage_thresholds), sigma_build2build)                                                                                                        # Modelling uncertainty
            betas_total          = [np.sqrt(sigma_record2record**2+sigma_build2build**2) for sigma_record2record, sigma_build2build in zip (sigmas_record2record, sigmas_build2build)]       # Dummy Total Dispersion

            # Create the dictionary
            cloud_dict = {
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {'imls'             : imls,               # Store the intensity measure levels (cloud)
                                 'edps'             : edps,               # Store the engineering demand parameters (cloud)
                                 'lower_limit'      : None,               # Store the lower limit for censored regression
                                 'upper_limit'      : None,               # Store the upper limit for censored regression
                                 'damage_thresholds': damage_thresholds}, # Store the demand-based damage state thresholds

                # Add a nested dictionary for fragility functions parameters
                'fragility': {'fragility_method'   : fragility_method.lower(), # Store the fragility fitting methodology
                              'intensities'        : intensities,              # Store the intensities used for sampling fragility functions
                              'poes'               : poes,                     # Store the probabilities of damage state exceedance
                              'medians'            : thetas,                   # Store the median seismic intensities
                              'sigma_record2record': sigmas_record2record,     # Store the record-to-record variability
                              'sigma_build2build'  : sigmas_build2build,       # Store the modelling uncertainty
                              'betas_total'        : betas_total},             # Store the total variability accounting for record-to-record and modelling uncertainties

                # Add a nested dictionary for regression coefficients
                'regression': {'b1'      : None,   # Store 'b1' coefficient
                               'b0'      : None,   # Store 'b0' coefficient
                               'sigma'   : None,   # Store 'sigma' value
                               'fitted_x': None,   # Store the fitted x-values
                               'fitted_y': None}   # Store the fitted y-values
                }

        elif fragility_method.lower() == 'lognormal':

            # Initialise seed for reproducibility
            if random_seed is not None:
                np.random.seed(random_seed)

            # Ensure inputs are in the right format
            imls, edps = np.asarray(imls), np.asarray(edps)

            # Create storage for probabilities of exceedance and regression parameters
            n_ds            = len(damage_thresholds)
            n_im            = len(intensities)
            poes_s          = np.zeros((n_bootstrap, n_im, n_ds +1)) # We add +1 to the damage states to store the "collapse" fragility in the last index
            a_s, b_s, sig_s = np.zeros(n_bootstrap), np.zeros(n_bootstrap), np.zeros(n_bootstrap)
            al0_s, al1_s    = np.zeros(n_bootstrap), np.zeros(n_bootstrap)

            # Bootstrapping loop
            for i in range(n_bootstrap):

                # Prepare bootstrap samples
                im_nc_b, edp_nc_b, im_c_b = prepare_mca_data(imls,
                                                                  edps,
                                                                  censored_limit,
                                                                  bootstrap=True)

                # Do classical cloud regression considering non-collapse cases only
                # Only keeping the EDPs above the lower limit (below the lower limit, the EDPs do not contribute to damage)
                mask_lower    = edp_nc_b >= lower_limit
                ln_im, ln_edp = np.log(im_nc_b[mask_lower]), np.log(edp_nc_b[mask_lower]) # Transform IMs and EDPs to log-log scale
                b = np.sum((ln_im - ln_im.mean()) * (ln_edp - ln_edp.mean())) / np.sum((ln_im - ln_im.mean())**2) # Calculate the b-parameter
                a = np.exp(ln_edp.mean()-b*ln_im.mean())                                                          # Calculate the a-parameter
                res = ln_edp - np.log(a*im_nc_b[mask_lower]**b)                                                   # Apply the regression to get the mean
                sig = np.linalg.norm(res)/np.sqrt(len(res)-2)                                                     # Get the standard error

                # Store the cloud analysis coefficients associated with the current bootstrap iteration
                a_s[i], b_s[i], sig_s[i] = a, b, sig

                # Do logistic regression to account for the collapse cases
                y_logit   = np.concatenate([np.zeros(len(im_nc_b)), np.ones(len(im_c_b))])
                x_logit   = sm.add_constant(np.log(np.concatenate([im_nc_b, im_c_b])))
                logit_mod = sm.GLM(y_logit, x_logit, family = sm.families.Binomial()).fit(disp=0)

                # Store the logistic regression coefficients associated with the current bootstrap iteration
                al0_s[i], al1_s[i] = logit_mod.params

                # Calculate the probabilities of exceedance
                p_collapse = logit_mod.predict(sm.add_constant(np.log(intensities))) # The probability of collapse
                mu_ln      = np.log(a* intensities**b)                               # The cloud regression
                sig_total = np.sqrt(sig**2 + sigma_build2build)                      # Total uncertainty (inflated with the building-to-building variability)

                # Loop over damage states
                for ds in range(n_ds):
                    # Calculate the non-collapse fragility functions
                    poe_nc = 1- norm.cdf(np.log(damage_thresholds[ds]), loc = mu_ln, scale = sig_total)
                    # Calculate the conditional fragility functions P(NC, IM|C)
                    poes_s[i,:, ds] = poe_nc * (1-p_collapse) + p_collapse

                # Store the collapse fragility
                poes_s[i,:,-1] = p_collapse

            # Calculate storage to calculate mean statistics using all the results of each bootstrap sample
            # and to store the equivalent lognormal CDF parameters
            poes_mean          = poes_s.mean(axis = 0)
            poes_fitted        = np.zeros_like(poes_mean)
            params_a, params_b = np.zeros(n_ds+1), np.zeros(n_ds+1)
            medians            = np.zeros(n_ds+1)
            betas_total        = np.zeros(n_ds+1)

            # Loop over damage states and collapse
            for ds in range(n_ds+1):
                # Fit functional form
                try:
                    popt, _ = curve_fit(cond_fragility,
                                        intensities,
                                        poes_mean[:, ds],
                                        bounds=((0,0), (np.inf, np.inf)))
                    params_a[ds], params_b[ds] = popt

                except Exception as e:
                    raise RuntimeError('ERROR! Curve fitting failed for damage state {ds}: {e}')

                # Calculate the fitted probabilities
                poes_fitted[:, ds] = cond_fragility(intensities,
                                                    params_a[ds],
                                                    params_b[ds])

                # Interpolate for lognormal equivalents: IMs at 16%, 50% and 84%
                f_interp    = interp1d(poes_fitted[:,ds], intensities, bounds_error=False, fill_value = 'extrapolate')
                medians[ds] = f_interp(0.5)
                im16        = f_interp(0.16)
                im84        = f_interp(0.84)

                # Calculate the uncertainty
                if im16 > 0 and im84 > im16:
                    betas_total[ds] = np.log(im84/im16)/2
                else:
                    betas_total[ds] = np.nan

            # Recalculate the lognormal fragility functions
            for ds in range(n_ds+1):

                if fragility_rotation:
                    fragility_method = f'lognormal - rotated around the {rotation_percentile}th percentile'
                    medians[ds],betas_total[ds],poes_fitted[:,ds] = self.calculate_rotated_fragility(rotation_percentile,
                                                                                                     medians[ds],
                                                                                                     betas_total[ds],
                                                                                                     sigma_build2build = 0.0)
                else:
                    poes_fitted[:,ds] = self.calculate_lognormal_fragility(medians[ds],
                                                                           betas_total[ds],
                                                                           sigma_build2build = 0.0)

            # Final cleanup: Make sure fragility functions are not crossing due to fit
            # Work backwards from Collapse to DS1 to ensure PoE(DS_i) is always
            # >= PoE(Collapse) and PoE(DS_i) >= PoE(DS_i+1)
            for i in range(n_ds-1,-1,-1):
                poes_fitted[:,i] = np.maximum(poes_fitted[:,i], poes_fitted[:,i+1])

            # Store everything in dedicated dictionary
            is_collapse = edps >= censored_limit
            is_nc_plot  = (~is_collapse) & (edps >= lower_limit)

            # Create the dictionary
            cloud_dict = {
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {'imls'             : imls,               # Store the intensity measure levels (cloud)
                                 'edps'             : edps,               # Store the engineering demand parameters (cloud)
                                 'lower_limit'      : lower_limit,        # Store the lower limit for censored regression
                                 'upper_limit'      : censored_limit,     # Store the upper limit for censored regression
                                 'damage_thresholds': damage_thresholds}, # Store the cloud analysis regression method

                # Add a nested dictionary for fragility functions parameters
                'fragility': {'fragility_method'   : fragility_method.lower(), # Store the fragility fitting methodology
                              'intensities'        : intensities,              # Store the intensities used for sampling fragility functions
                              'poes'               : poes_fitted,              # Store the probabilities of damage state exceedance
                              'medians'            : medians,                  # Store the median seismic intensities
                              'sigma_record2record': sig_s.mean(),             # Store the record-to-record variability
                              'sigma_build2build'  : sigma_build2build,        # Store the modelling uncertainty
                              'betas_total'        : betas_total},             # Store the total variability accounting for record-to-record and modelling uncertainties

                # Add a nested dictionary for regression coefficients
                'regression': {'b1'      : b_s.mean(),                                          # Store 'b1' coefficient
                               'b0'      : np.log(a_s.mean()),                                  # Store 'b0' coefficient
                               'sigma'   : sig_s.mean(),                                        # Store 'sigma' value
                               'fitted_x': np.log(intensities),                                 # Store the fitted x-values
                               'fitted_y': np.log(a_s.mean())+b_s.mean()*np.log(intensities)},  # Store the fitted y-values

                # Add a nested dictionary for bootstrap iteration results
                'bootstraps': {'b1'      : b_s,             # Array of all bootstrap slopes
                               'a'       :  a_s,            # Array of all bootstrap a-coefficients
                               'sigma_rr': sig_s,           # Array of all record-to-record variabilities
                               'alpha0'  : al0_s,           # Array of all Logistic intercept params
                               'alpha1'  : al1_s,           # Array of all Logistic slope params
                               'poes_all': poes_s},         # Full 3D array (n_boot, n_im, n_ds+1)

                'raw_data': {'im_nc' : imls[is_nc_plot],
                            'edp_nc': edps[is_nc_plot],
                            'im_c'  : imls[is_collapse]}
                }

            return cloud_dict

    def do_multiple_stripe_analysis(self,
                                    imls,
                                    edps,
                                    damage_thresholds,
                                    sigma_build2build=0.3,
                                    intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                    fragility_rotation=False,
                                    rotation_percentile=0.10):
        """
        Perform maximum likelihood estimation (MLE) for fragility curve fitting following a multiple stripe analysis.
        This method calculates the fragility function by fitting to the provided intensity measure levels (IMLs)
        and engineering demand parameters (EDPs) "stripes", with the option to rotate the fragility curve around
        a target percentile.

        The method is useful for deriving fragility functions by determining the probability
        of exceedance for various damage states based on the provided data.

        Parameters:
        -----------
        imls : list or array
            A list or array of intensity measure levels (IMLs) representing the seismic intensity levels used for
            sampling the fragility functions.

        edps : list or array
            A list or array of engineering demand parameters (EDPs), which describe the structural response to
            seismic events. Examples include maximum interstorey drifts, maximum peak floor acceleration, or top
            displacements.

        damage_thresholds : list
            A list of EDP-based damage thresholds that correspond to different levels of structural damage, such
            as slight, moderate, extensive, and complete. These thresholds help categorize the severity of damage
            based on EDP values.

        sigma_build2build : float, optional, default=0.3
            The building-to-building variability or modeling uncertainty. It accounts for differences in performance
            between buildings with similar characteristics due to random variations or model uncertainties.

        intensities : array, optional, default=np.geomspace(0.05, 10.0, 50)
            An array of intensity measure levels over which the fragility function will be sampled. By default,
            this is a logarithmic space ranging from 0.05 to 10.0, with 50 sample points.

        fragility_rotation : bool, optional, default=False
            A boolean flag that determines whether or not to rotate the fragility curve about a given percentile.
            If `True`, the fragility curve will be adjusted based on the specified `rotation_percentile`.

        rotation_percentile : float, optional, default=0.10
            The target percentile (between 0 and 1) around which the fragility function will be rotated. A value of
            0.10 corresponds to rotating the curve to the 10th percentile.

        Returns:
        --------
        msa_dict : dict
            A dictionary containing the results of the multiple stripe analysis, including:
            - 'medians': The estimated medians of the fragility function.
            - 'dispersions': The estimated dispersions (variability) associated with the fragility function.
            - 'poes': The probabilities of exceedance (damage probabilities) for different damage states.

        Notes:
        ------
        This method fits the fragility curve using MLE, which minimizes the difference between observed and predicted
        exceedance probabilities. The option for fragility curve rotation allows for adjusting the curve to better
        match the expected percentile of damage occurrence, offering greater flexibility in representing the fragility
        of the structure.
        """

        # Convert to numpy arrays to ensure indexing works
        imls = np.array(imls)
        edps = np.array(edps)

        # Extract unique IM levels for each stripe (one value per column)
        # Assuming each column in imls has the same IM value
        stripe_imls = np.mean(imls, axis=0)
        num_stripes = len(stripe_imls)
        num_gmrs_per_stripe = np.array([len(edps[:, i]) for i in range(num_stripes)])

        def likelihood(params, x, n, z, sigma_b2b):
            """
            Negative Log-Likelihood for Binomial Distribution.
            x: stripe IMs, n: total GMs per stripe, z: exceedances per stripe
            """
            theta = params[0]
            beta_r2r = params[1]

            beta_tot = np.sqrt(beta_r2r**2 + sigma_b2b**2)

            # Probability of exceedance based on lognormal CDF
            p = stats.norm.cdf(np.log(x / theta) / beta_tot)
            p = np.clip(p, 1e-10, 1 - 1e-10) # Numerical stability

            # log-likelihood of binomial pmf: log(nCz * p^z * (1-p)^(n-z))
            log_f = stats.binom.logpmf(z, n, p)
            return -np.sum(log_f)

        thetas = []
        sigmas_record2record = []
        betas_total = []

        # Iterate through each Damage State threshold
        for threshold in damage_thresholds:
            # Count exceedances per stripe (column-wise)
            num_exc = np.array([np.sum(edps[:, i] >= threshold) for i in range(num_stripes)])

            # Initial guess: theta = median of stripes, beta = 0.4
            initial_guess = [np.median(stripe_imls), 0.4]
            bounds = optimize.Bounds([0.001 * np.min(stripe_imls), 0.05],
                                     [10.0 * np.max(stripe_imls), 1.5])

            sol = optimize.minimize(likelihood, initial_guess,
                                    args=(stripe_imls, num_gmrs_per_stripe, num_exc, sigma_build2build),
                                    bounds=bounds, method='L-BFGS-B')

            t_val = sol.x[0]
            s_val = sol.x[1]
            b_val = np.sqrt(s_val**2 + sigma_build2build**2)

            thetas.append(t_val)
            sigmas_record2record.append(s_val)
            betas_total.append(b_val)

        # Calculate Fragility Curves (POEs)
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        for i in range(len(damage_thresholds)):
            if fragility_rotation:
                poes[:, i] = self.calculate_rotated_fragility(thetas[i],
                                                             rotation_percentile,
                                                             sigmas_record2record[i],
                                                             sigma_build2build=sigma_build2build)
            else:
                poes[:, i] = self.calculate_lognormal_fragility(thetas[i],
                                                               sigmas_record2record[i],
                                                               sigma_build2build=sigma_build2build)

        # Output dictionary following requested nested structure
        msa_dict = {
            'msa inputs': {
                'imls': imls,
                'edps': edps,
                'damage_thresholds': damage_thresholds,
                'sigma_build2build': sigma_build2build,
                'is_rotated': fragility_rotation
            },
            'fragility': {
                'fragility_method': 'mle',
                'intensities': intensities,
                'poes': poes,
                'medians': thetas,
                'sigma_record2record': sigmas_record2record,
                'sigma_build2build': sigma_build2build,
                'betas_total': betas_total
            }
        }

        return msa_dict

    def do_incremental_dynamic_analysis(self,
                                        ansys_dict,
                                        im_matrix,
                                        damage_thresholds,
                                        edp_key,
                                        sigma_build2build=0.3,
                                        intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                        edp_range = np.linspace(0.00, 0.05, 101),
                                        fragility_rotation=False,
                                        rotation_percentile=0.10):
        """
        Perform fragility function fitting and statistical processing on Incremental Dynamic Analysis (IDA) results.

        This method processes raw IDA data by interpolating individual record response curves to a continuous
        Engineering Demand Parameter (EDP) range. It accounts for "flatlining" (global dynamic instability)
        using Maximum Likelihood Estimation (MLE) for censored data to estimate the fragility parameters
        (median and dispersion) for multiple damage states. It also supports fragility curve rotation
        around a target percentile to account for modeling uncertainties.

        Parameters
        ----------
        ansys_dict : dict
            A dictionary containing the structural response data. Must include:
            - 'max_peak_drift_list' or 'max_peak_accel_list': A list where each entry is a dictionary mapping
              Scale Factors (SF) to the resulting peak drift or acceleration values.
            - 'sf_matrix': A 2D numpy array (n_records x max_runs) containing the
              scale factors used for each analysis run.

        im_matrix : numpy.ndarray
            A 2D array of intensity measure levels corresponding to the ground motion records and
            number of runs carried out in IDA

        damage_thresholds : list of float
            A list of EDP-based damage thresholds (e.g., interstorey drift ratios) defining
            different limit states (e.g., Slight, Moderate, Extensive, Collapse).

        edp_key : str, optional, default='max_peak_drift_list' (other option is "max_peak_accel_list")
            The key in `ansys_dict` used to retrieve the engineering demand parameter data.

        sigma_build2build : float, optional, default=0.3
            The modeling uncertainty or building-to-building variability. This is combined
            with the record-to-record variability to calculate total fragility dispersion.

        intensities : numpy.ndarray, optional, default=np.geomspace(0.05, 10.0, 50)
            The array of intensity measure levels over which the final fragility functions (POEs)
            will be sampled.

        edp_range: numpy.ndarray, optional, default = np.linspace(0.00, 0.05, 101) (i.e., 0% to 5% drift)
            The array of engineering demand parameters over which the IDA curves will be evaluated

        fragility_rotation : bool, optional, default=False
            Flag to determine if the fragility curves should be rotated around a specific
            percentile to adjust for modeling bias or target reliability levels.

        rotation_percentile : float, optional, default=0.10
            The target percentile (0.0 to 1.0) around which the fragility curve rotation is
            anchored if `fragility_rotation` is True.

        Returns
        -------
        ida_dict : dict
            A nested dictionary containing the analysis results:
            - 'ida_inputs': Metadata including target EDPs and raw interpolated curves.
            - 'fragility': Medians, dispersions (record-to-record and total), and Probabilities
              of Exceedance (POEs).
            - 'stats': Statistical IDA curves including the 16th, 50th (median), and 84th
              intensity percentiles across the EDP range.

        Notes
        -----
        The method uses a log-likelihood minimization approach to handle records that do not
        reach a specific damage threshold within the analyzed range (right-censored data),
        ensuring the fragility curves remain statistically robust even near collapse.
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
                idx = np.argsort(rec_edps)
                sorted_edps = np.array(rec_edps)[idx]
                sorted_ims = np.array(rec_ims)[idx]

                raw_curves.append({'im': sorted_ims, 'edp': sorted_edps})

                # Interpolate IM = f(EDP) to ensure horizontal flatlining
                f_im_cap = interp1d(sorted_edps, sorted_ims,
                                    bounds_error=False, fill_value="extrapolate")
                im_at_edp_matrix.append(f_im_cap(edp_range))
            else:
                im_at_edp_matrix.append(np.full_like(edp_range, np.nan))

        im_at_edp_matrix = np.array(im_at_edp_matrix)

        # Calculate Statistical IDA Percentiles (16/50/84)
        median_ida_im = np.nanmedian(im_at_edp_matrix, axis=0)
        p16_ida_im = np.nanpercentile(im_at_edp_matrix, 16, axis=0)
        p84_ida_im = np.nanpercentile(im_at_edp_matrix, 84, axis=0)

        # Fragility Fitting (MLE for Censored Data)
        im_max = np.nanmax(im_matrix)
        thetas = []
        sigmas_rec2rec = []

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
                    if t <= 0 or b <= 0: return 1e10
                    collapsed = capacities[~np.isnan(capacities)]
                    term1 = np.sum(np.log(stats.norm.pdf((np.log(collapsed)-np.log(t))/b)/(collapsed*b)))
                    term2 = (n_records - num_collapsed) * np.log(1 - stats.norm.cdf((np.log(im_max)-np.log(t))/b))
                    return -(term1 + term2)

                sol = optimize.minimize(log_likelihood, [im_max, 0.4], method='Nelder-Mead')
                theta, beta_rec = sol.x[0], sol.x[1]

            thetas.append(theta)
            sigmas_rec2rec.append(beta_rec)

        # Generate Probabilities of Exceedance with Rotation Option
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        betas_total = []

        for i, threshold in enumerate(damage_thresholds):
            theta = thetas[i]
            beta_rec = sigmas_rec2rec[i]

            if fragility_rotation:
                # Combined uncertainty isn't a simple SRSS in rotation,
                # but we report total for consistency in dict
                betas_total.append(np.sqrt(beta_rec**2 + sigma_build2build**2))
                poes[:, i] = self.calculate_rotated_fragility(theta,
                                                             rotation_percentile,
                                                             beta_rec,
                                                             sigma_build2build,
                                                             intensities)
            else:
                beta_total = np.sqrt(beta_rec**2 + sigma_build2build**2)
                betas_total.append(beta_total)
                poes[:, i] = self.calculate_lognormal_fragility(theta, beta_total)

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
                'sigma_build2build': sigma_build2build,
                'betas_total': betas_total,
                'rotation_active': fragility_rotation,
                'rotation_percentile': rotation_percentile if fragility_rotation else None
            },

            'stats': {
                'fitted_edps': edp_range,
                'median_im': median_ida_im,
                'p16_im': p16_ida_im,
                'p84_im': p84_ida_im
            }
        }

        return ida_dict

    def calculate_sigma_loss(self,
                             loss):
        """
        Calculate the uncertainty in the loss estimates based on the method proposed in Silva (2019),
        which incorporates the sigma (standard deviation) for loss ratios within seismic vulnerability functions.

        This method computes the sigma loss ratio for expected loss ratios and also estimates the parameters
        of a beta distribution (coefficients a and b), which describe the uncertainty and variability in
        the loss estimates. The formula used is derived from seismic vulnerability research.

        Parameters:
        -----------
        loss : list or array
            A list or array of expected loss ratios. The expected loss ratio represents the proportion of
            the building's value that is expected to be lost due to an earthquake event, ranging from 0 to 1.

        Returns:
        --------
        sigma_loss_ratio : list or array
            The calculated uncertainty (sigma) associated with the mean loss ratio for each input loss value.
            The sigma loss ratio represents the variability of the loss estimates and is computed based on the
            loss ratios provided.

        a_beta_dist : list or array
            The coefficient 'a' of the beta distribution for each loss ratio. This parameter represents the shape
            of the beta distribution and is used to model the uncertainty in the loss estimates.

        b_beta_dist : list or array
            The coefficient 'b' of the beta distribution for each loss ratio. This parameter also represents the
            shape of the beta distribution, complementing the coefficient 'a' to fully describe the distribution's
            behavior.

        References:
        ----------
        1) Silva, V. (2019) "Uncertainty and correlation in seismic vulnerability functions of building classes."
        Earthquake Spectra. DOI: 10.1193/013018eqs031m.

        """
        sigma_loss_ratio = np.where(loss == 0, 0,
                                    np.where(loss == 1, 1,
                                             np.sqrt(loss * (-0.7 - 2 * loss + np.sqrt(6.8 * loss + 0.5)))))
        a_beta_dist = np.zeros(loss.shape)
        b_beta_dist = np.zeros(loss.shape)

        return sigma_loss_ratio, a_beta_dist, b_beta_dist

    def get_vulnerability_function(self,
                                   poes,
                                   consequence_model,
                                   intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                   uncertainty=True):
        """
        Calculate the vulnerability function given the probabilities of exceedance and a consequence model,
        and optionally compute the uncertainty (coefficient of variation) in the expected loss.

        This function computes the expected loss ratios for a range of intensity measure levels (IMLs)
        based on the probabilities of exceedance and the corresponding consequence model. Additionally,
        it calculates the coefficient of variation (COV) of the loss ratio if the uncertainty flag is set to True.

        Parameters:
        -----------
        poes : array
            An array of probabilities of exceedance associated with the damage states considered.
            The shape is (number of intensities, number of damage states).

        consequence_model : list
            A list of damage-to-loss ratios corresponding to each damage state. It has a length equal
            to the number of damage states.

        intensities : array, optional
            An array of intensity measure levels. The default is a geometric sequence ranging from
            0.05 to 10.0 with 50 points.

        uncertainty : bool, optional
            A flag to indicate whether to calculate (or not) the coefficient of variation associated
            with Loss|IM. The default is True.

        Returns:
        --------
        df : pandas DataFrame
            A DataFrame containing the intensity measure levels (IML), expected loss ratios, and
            optionally, the coefficient of variation (COV) for each IML. The COV is calculated only
            if the uncertainty flag is True.

        """

        # Consistency checks
        if len(consequence_model) != np.size(poes, 1):
            raise Exception('Mismatch between the fragility consequence models!')
        if len(intensities) != np.size(poes, 0):
            raise Exception('Mismatch between the number of IMLs and fragility models!')

        # Initialize loss array
        loss = np.zeros([len(intensities),])

        # Calculate expected loss ratios
        for i in range(len(intensities)):
            for j in range(0, np.size(poes, 1)):
                if j == (np.size(poes, 1) - 1):
                    loss[i,] = loss[i,] + poes[i, j] * consequence_model[j]
                else:
                    loss[i,] = loss[i,] + (poes[i, j] - poes[i, j + 1]) * consequence_model[j]

        # If uncertainty is true, calculate the coefficient of variation
        if uncertainty:
            cov = np.zeros(loss.shape)

            for m in range(loss.shape[0]):
                mean_loss_ratio = loss[m]

                if mean_loss_ratio < 1e-4:
                    loss[m] = 1e-8
                    cov[m] = 1e-8
                elif np.abs(1 - mean_loss_ratio) < 1e-4:
                    loss[m] = 0.99999
                    cov[m] = 1e-8
                else:
                    # Use the calculate_sigma_loss function for loss-related uncertainty
                    sigma_loss_ratio, a_beta_dist, b_beta_dist = self.calculate_sigma_loss(loss[m])

                    # Coefficient of variation
                    cov[m] = np.min([sigma_loss_ratio / mean_loss_ratio, 0.90 * np.sqrt(mean_loss_ratio * (1 - mean_loss_ratio)) / mean_loss_ratio])

            # Store to DataFrame with COV
            df = pd.DataFrame({'IML': intensities,
                               'Loss': loss,
                               'COV': cov})

        else:
            # Store to DataFrame without COV
            df = pd.DataFrame({'IML': intensities,
                               'Loss': loss})

        return df

    def calculate_average_annual_damage_probability(self,
                                                    fragility_array,
                                                    hazard_array,
                                                    return_period=1,
                                                    max_return_period=5000):
        """
        Calculate the Average Annual Damage State Probability (AADP) based on fragility and hazard curves.

        This function estimates the average annual probability of damage states occurring over a given return period,
        using the fragility curve (which relates intensity measure levels to damage state probabilities) and the hazard
        curve (which relates intensity measure levels to annual rates of exceedance).

        The calculation integrates the product of the fragility function and the hazard curve over the specified range
        of intensity measure levels, accounting for the return period and a maximum return period threshold.

        Parameters:
        -----------
        fragility_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the
            corresponding probabilities of exceedance for each intensity level.

        hazard_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the
            annual rates of exceedance (i.e., the probability of exceedance per year) for each intensity level.

        return_period : float, optional, default=1
            The return period used to scale the hazard rate. This is the time span (in years) over which the
            average annual damage probability is calculated. A typical value is 1 year, but longer periods can be used
            for multi-year assessments.

        max_return_period : float, optional, default=5000
            The maximum return period threshold used to filter out very low hazard rates. The hazard curve is truncated
            to include only intensity levels with exceedance rates above this threshold.

        Returns:
        --------
        average_annual_damage_probability : float
            The average annual damage state probability, calculated by integrating the product of the fragility
            function and the hazard curve over the given intensity measure levels.

        """

        # Filter hazard array based on the maximum return period
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]

        # Compute mean intensity levels and rate of occurrences
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        rate_occ = np.diff(hazard_array[:, 1]) / -return_period

        # Define fragility curve for interpolation
        curve_imls = np.hstack(([0], fragility_array[:, 0], [20]))
        curve_ordinates = np.hstack(([0], fragility_array[:, 1], [1]))

        # Interpolate fragility curve values at the mean intensity levels
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)

        # Compute the average annual damage probability
        return np.dot(interpolated_values, rate_occ)

    def calculate_average_annual_loss(self,
                                      vulnerability_array,
                                      hazard_array,
                                      return_period=1,
                                      max_return_period=5000):
        """
        Calculate the Average Annual Loss (AAL) based on vulnerability and hazard curves.

        This function computes the average annual loss by integrating the product of the vulnerability function
        (which relates intensity measure levels to loss ratios) and the hazard curve (which relates intensity measure
        levels to annual rates of exceedance). The result represents the expected average loss over a given return period.

        Parameters:
        -----------
        vulnerability_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the
            corresponding loss ratios (representing expected loss relative to the building value or some other metric).

        hazard_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the
            annual rates of exceedance (i.e., the probability of exceedance per year) for each intensity level.

        return_period : float, optional, default=1
            The return period used to scale the hazard rate. This is the time span (in years) over which the
            average annual loss is calculated. Typically, this value is 1 year, but longer periods can be used
            for multi-year assessments.

        max_return_period : float, optional, default=5000
            The maximum return period threshold used to filter out very low hazard rates. The hazard curve is truncated
            to include only intensity levels with exceedance rates above this threshold.

        Returns:
        --------
        average_annual_loss : float
            The average annual loss, calculated by integrating the product of the vulnerability function and the
            hazard curve over the given intensity measure levels. This value represents the expected loss per year.

        """

        # Filter hazard array based on the maximum return period
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]

        # Compute mean intensity levels and rate of occurrences
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        rate_occ = np.diff(hazard_array[:, 1]) / -return_period

        # Define vulnerability curve for interpolation
        curve_imls = np.hstack(([0], vulnerability_array[:, 0], [20]))
        curve_ordinates = np.hstack(([0], vulnerability_array[:, 1], [1]))

        # Interpolate vulnerability curve values at the mean intensity levels
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)

        # Compute the average annual loss
        return np.dot(interpolated_values, rate_occ)

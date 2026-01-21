import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

class plotter:
    """
    A class for creating and customizing various types of plots for structural analysis results.

    This class provides methods to visualize data from structural analyses, including cloud analysis,
    fragility analysis, demand profiles, vulnerability analysis, and animations of seismic responses.
    It also includes utility methods for setting consistent plot styles and saving plots.

    Attributes
    ----------
    font_sizes : dict
        Dictionary containing font sizes for titles, labels, ticks, and legends.
    line_widths : dict
        Dictionary containing line widths for thick, medium, and thin lines.
    marker_sizes : dict
        Dictionary containing marker sizes for large, medium, and small markers.
    colors : dict
        Dictionary containing color schemes for fragility, damage states, and GEM colors.
    resolution : int
        Resolution for saving plots (default: 500 DPI).
    font_name : str
        Font name for plot text (default: 'Arial').

    Methods
    -------
    _set_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True)
        Sets consistent plot style for all plots.
    _save_plot(output_directory, plot_label)
        Saves the plot to the specified directory.
    duplicate_for_drift(peak_drift_list, control_nodes)
        Creates data for box plots of peak storey drifts.
    plot_cloud_analysis(cloud_dict, output_directory=None, plot_label='cloud_analysis_plot', xlabel='Peak Ground Acceleration, PGA [g]', ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]')
        Plots cloud analysis results.
    plot_fragility_analysis(cloud_dict, output_directory=None, plot_label='fragility_plot', xlabel='Peak Ground Acceleration, PGA [g]')
        Plots fragility analysis results.
    plot_demand_profiles(peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='demand_profiles')
        Plots demand profiles for peak drifts and accelerations.
    plot_ansys_results(cloud_dict, peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='ansys_results', cloud_xlabel='PGA', cloud_ylabel='MPSD')
        Plots a 2x2 grid of analysis results, including cloud, fragility, and demand profiles.
    plot_vulnerability_analysis(intensities, loss, cov, xlabel, ylabel, output_directory=None, plot_label='vulnerability_plot')
        Plots vulnerability analysis results, including Beta distributions and loss curves.
    plot_slf_model(out, cache, xlabel, output_directory=None, plot_label='slf')
        Plots Storey Loss Function (SLF) model results.
    animate_model_run(control_nodes, acc, dts, nrha_disps, nrha_accels, drift_thresholds, output_directory=None, plot_label='animation')
        Animates the seismic demands for a single nonlinear time-history analysis (NRHA) run.

    Notes
    -----
    - The class uses Matplotlib and Seaborn for plotting.
    - The `_set_plot_style` method ensures consistent styling across all plots.
    - The `_save_plot` method handles saving plots with high resolution.
    - The `plot_cloud_analysis`, `plot_fragility_analysis`, and `plot_demand_profiles` methods are used for visualizing structural analysis results.
    - The `plot_vulnerability_analysis` method visualizes loss distributions and loss curves.
    - The `plot_slf_model` method visualizes Storey Loss Function (SLF) results.
    - The `animate_model_run` method creates animations of seismic responses.

    """

    def __init__(self):
        # Define default styles
        self.font_sizes = {
            'title': 16,
            'labels': 14,
            'ticks': 12,
            'legend': 10
        }
        self.line_widths = {
            'thick': 3,
            'medium': 2,
            'thin': 1
        }
        self.marker_sizes = {
            'large': 100,
            'medium': 60,
            'small': 10
        }
        self.colors = {
            'fragility': ['green', 'yellow', 'orange', 'red'],
            'damage_states': ['blue', 'green', 'yellow', 'orange', 'red'],
            'gem': ["#0A4F4E", "#0A4F5E", "#54D7EB", "#54D6EB", "#399283", "#399264", "#399296"]
        }
        self.resolution = 500
        self.font_name = 'Arial'

    def _set_plot_style(self, ax, title=None, xlabel=None, ylabel=None, grid=True):
        """Set consistent plot style for all plots."""
        if title:
            ax.set_title(title, fontsize=self.font_sizes['title'], fontname=self.font_name)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.font_sizes['labels'], fontname=self.font_name)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.font_sizes['labels'], fontname=self.font_name)
        ax.tick_params(axis='both', labelsize=self.font_sizes['ticks'])
        if grid:
            ax.grid(visible=True, which='major')
            ax.grid(visible=True, which='minor')

    def _save_plot(self, output_directory, plot_label):
        """Save the plot if output_directory is provided."""
        if output_directory:
            plt.savefig(f'{output_directory}/{plot_label}.png', dpi=self.resolution, format='png')
        plt.show()

    def duplicate_for_drift(self,
                            peak_drift_list,
                            control_nodes):
        """Creates data to process box plots for peak storey drifts."""
        x = []; y = []
        for i in range(len(control_nodes)-1):
            y.extend((float(control_nodes[i]),float(control_nodes[i+1])))
            x.extend((peak_drift_list[i],peak_drift_list[i]))
        y.append(float(control_nodes[i+1]))
        x.append(0.0)

        return x, y

    def plot_cloud_analysis(self,
                            cloud_dict,
                            output_directory=None,
                            plot_label='cloud_analysis_plot',
                            xlabel='Peak Ground Acceleration, PGA [g]',
                            ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]'):

        """
        Generate a cloud analysis plot with scatter points and regression line,
        visualizing the relationship between Peak Ground Acceleration (PGA)
        and Maximum Peak Storey Drift.

        This method plots cloud data, damage thresholds, a fitted regression line,
        and upper and lower censoring limits. The data is presented in logarithmic
        scale for both axes.

        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the cloud analysis. The dictionary
            should have the following keys (direct output from do_cloud_analysis method)

        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved
            in the current working directory.

        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is
            'cloud_analysis_plot'.

        xlabel : str, optional
            The label for the x-axis. Default is 'Peak Ground Acceleration, PGA [g]'.

        ylabel : str, optional
            The label for the y-axis. Default is 'Maximum Peak Storey Drift, $\theta_{max}$ [%]'.

        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.

        """

        fig, ax = plt.subplots(figsize=(6, 6))
        self._set_plot_style(ax, xlabel=xlabel, ylabel=ylabel)

        ax.scatter(cloud_dict['cloud inputs']['imls'], cloud_dict['cloud inputs']['edps'], color=self.colors['gem'][2], s=self.marker_sizes['medium'], alpha=0.5, label='Cloud Data', zorder=0)
        for i in range(len(cloud_dict['cloud inputs']['damage_thresholds'])):
            ax.scatter(cloud_dict['fragility']['medians'][i], cloud_dict['cloud inputs']['damage_thresholds'][i], color=self.colors['fragility'][i], s=self.marker_sizes['large'], alpha=1.0, zorder=2)

        ax.plot(cloud_dict['regression']['fitted_x'], cloud_dict['regression']['fitted_y'], linestyle='solid', color=self.colors['gem'][1], lw=self.line_widths['thick'], label='Cloud Regression', zorder=1)
        ax.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['upper_limit'], cloud_dict['cloud inputs']['upper_limit']], '--', color=self.colors['gem'][-1], label='Upper Censoring Limit')
        ax.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['lower_limit'], cloud_dict['cloud inputs']['lower_limit']], '-.', color=self.colors['gem'][-1], label='Lower Censoring Limit')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])])
        ax.set_ylim([min(cloud_dict['cloud inputs']['edps']), max(cloud_dict['cloud inputs']['edps'])])
        ax.legend(fontsize=self.font_sizes['legend'])

        self._save_plot(output_directory, plot_label)

    def plot_fragility_analysis(self,
                                cloud_dict,
                                output_directory=None,
                                plot_label='fragility_plot',
                                xlabel='Peak Ground Acceleration, PGA [g]'):

        """
        Generate a fragility analysis plot showing the probability of exceedance (PoE)
        for various damage states as a function of Peak Ground Acceleration (PGA).

        This method plots fragility curves for multiple damage states based on the
        fragility data in the input dictionary. Each curve represents the probability
        of exceedance for a specific damage state, and the plot is presented in a
        linear scale for both axes.

        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the fragility analysis. The dictionary
            should have the following keys:
                - 'fragility': A dictionary containing:
                    - 'intensities': List or array of intensity values (e.g., PGA levels).
                    - 'poes': 2D array of probabilities of exceedance for each damage state.
                - 'medians': List of medians for each damage state.

        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved
            in the current working directory.

        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is
            'fragility_plot'.

        xlabel : str, optional
            The label for the x-axis. Default is 'Peak Ground Acceleration, PGA [g]'.

        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.

        """

        fig, ax = plt.subplots(figsize=(6, 6))
        self._set_plot_style(ax, xlabel=xlabel, ylabel='Probability of Exceedance')

        for i in range(len(cloud_dict['fragility']['medians'])):
            ax.plot(cloud_dict['fragility']['intensities'], cloud_dict['fragility']['poes'][:, i], linestyle='solid', color=self.colors['fragility'][i], lw=self.line_widths['thick'], label=f'DS{i+1}')

        ax.set_xlim([0, 5])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=self.font_sizes['legend'])

        self._save_plot(output_directory, plot_label)


    def plot_demand_profiles(self,
                             peak_drift_list,
                             peak_accel_list,
                             control_nodes,
                             output_directory=None,
                             plot_label='demand_profiles'):

        """
        Generate demand profile plots for peak storey drifts and peak floor accelerations.

        This method creates two side-by-side plots:
        - A plot of peak storey drift (%), displaying how the drift varies with floor number.
        - A plot of peak floor acceleration (g), displaying how the acceleration varies with floor number.

        The data is presented as lines representing each control node's response at different floors.

        Parameters:
        ----------
        peak_drift_list : list of np.ndarray
            A list of arrays where each array contains peak drift values for each floor, with the first column being the drift values and the second column being the floor numbers.

        peak_accel_list : list of np.ndarray
            A list of arrays where each array contains peak acceleration values for each floor, with the first column being the acceleration values and the second column being the floor numbers.

        control_nodes : list
            A list of floor numbers or nodes that represent the control points in the structure.

        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.

        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'demand_profiles'.

        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self._set_plot_style(ax1, xlabel=r'Peak Storey Drift, $\theta_{max}$ [%]', ylabel='Floor No.')
        self._set_plot_style(ax2, xlabel=r'Peak Floor Acceleration, $a_{max}$ [g]', ylabel='Floor No.')

        nst = len(control_nodes) - 1
        for i in range(len(peak_drift_list)):
            x, y = self.duplicate_for_drift(peak_drift_list[i][:, 0], control_nodes)
            ax1.plot([float(i) * 100 for i in x], y, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][1], alpha=0.7)
            ax2.plot([float(x) / 9.81 for x in peak_accel_list[i][:, 0]], control_nodes, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][0], alpha=0.7)

        ax1.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax2.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax1.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax2.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax1.set_xlim([0, 5.0])
        ax2.set_xlim([0, 5.0])

        self._save_plot(output_directory, plot_label)


    def plot_ansys_results(self,
                           cloud_dict,
                           peak_drift_list,
                           peak_accel_list,
                           control_nodes,
                           output_directory=None,
                           plot_label='ansys_results',
                           cloud_xlabel='PGA',
                           cloud_ylabel='MPSD'):
        """
        Generate a 2x2 grid of plots to visualize analysis results, including cloud analysis,
        fragility analysis, and demand profiles for both peak drifts and peak accelerations.

        This function generates four plots in a 2x2 grid layout:
        1. **Cloud Analysis**: Scatter plot of cloud data, fitted regression line, and censoring limits.
        2. **Fragility Analysis**: Plot of probability of exceedance (PoE) for different damage states.
        3. **Demand Profiles for Drifts**: Plot of peak storey drift (%) versus floor number.
        4. **Demand Profiles for Accelerations**: Plot of peak floor acceleration (g) versus floor number.

        Each plot is customized with appropriate labels, legends, and color schemes for clarity.

        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the cloud and fragility analyses. The dictionary should contain:
                - 'imls': Intensity Measure Levels for cloud analysis.
                - 'edps': Engineering Demand Parameters for cloud analysis.
                - 'cloud inputs': Dictionary with damage thresholds, upper and lower limits.
                - 'fragility': Dictionary with fragility intensities and probabilities of exceedance.
                - 'regression': Fitted x and y values for the cloud regression line.
                - 'medians': List of median values for each damage state.

        peak_drift_list : list of np.ndarray
            A list of arrays where each array contains peak drift values for each floor. The first column should be the drift values and the second column the floor numbers.

        peak_accel_list : list of np.ndarray
            A list of arrays where each array contains peak acceleration values for each floor. The first column should be the acceleration values and the second column the floor numbers.

        control_nodes : list
            A list of control node (floor) numbers for the structure.

        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.

        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'ansys_results'.

        cloud_xlabel : str, optional
            The label for the x-axis of the cloud analysis plot. Default is 'PGA'.

        cloud_ylabel : str, optional
            The label for the y-axis of the cloud analysis plot. Default is 'MPSD'.

        Returns:
        --------
        None
            This function saves the 2x2 grid of plots to a file in the specified output directory.

        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        plt.rcParams['axes.axisbelow'] = True

        # Cloud Analysis
        self._set_plot_style(ax1, xlabel=cloud_xlabel, ylabel=cloud_ylabel)
        ax1.scatter(cloud_dict['cloud inputs']['imls'], cloud_dict['cloud inputs']['edps'], color=self.colors['gem'][2], s=self.marker_sizes['medium'], alpha=0.5, label='Cloud Data', zorder=0)
        for i in range(len(cloud_dict['cloud inputs']['damage_thresholds'])):
            ax1.scatter(cloud_dict['fragility']['medians'][i], cloud_dict['cloud inputs']['damage_thresholds'][i], color=self.colors['fragility'][i], s=self.marker_sizes['large'], alpha=1.0, zorder=2)
        ax1.plot(cloud_dict['regression']['fitted_x'], cloud_dict['regression']['fitted_y'], linestyle='solid', color=self.colors['gem'][1], lw=self.line_widths['thick'], label='Cloud Regression', zorder=1)
        ax1.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['upper_limit'], cloud_dict['cloud inputs']['upper_limit']], '--', color=self.colors['gem'][-1], label='Upper Censoring Limit')
        ax1.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['lower_limit'], cloud_dict['cloud inputs']['lower_limit']], '-.', color=self.colors['gem'][-1], label='Lower Censoring Limit')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=self.font_sizes['legend'])

        # Fragility Analysis
        self._set_plot_style(ax2, xlabel=cloud_xlabel, ylabel='Probability of Exceedance')
        for i in range(len(cloud_dict['fragility']['medians'])):
            ax2.plot(cloud_dict['fragility']['intensities'], cloud_dict['fragility']['poes'][:, i], linestyle='solid', color=self.colors['fragility'][i], lw=self.line_widths['thick'], label=f'DS{i+1}')
        ax2.set_xlim([0, 5])
        ax2.set_ylim([0, 1])
        ax2.legend(fontsize=self.font_sizes['legend'])

        # Demand Profiles: Drifts
        self._set_plot_style(ax3, xlabel=r'Peak Storey Drift, $\theta_{max}$ [%]', ylabel='Floor No.')
        nst = len(control_nodes) - 1
        for i in range(len(peak_drift_list)):
            x, y = self.duplicate_for_drift(peak_drift_list[i][:, 0], control_nodes)
            ax3.plot([float(i) * 100 for i in x], y, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][1], alpha=0.7)
        ax3.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax3.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax3.set_xlim([0, 5.0])

        # Demand Profiles: Accelerations
        self._set_plot_style(ax4, xlabel=r'Peak Floor Acceleration, $a_{max}$ [g]', ylabel='Floor No.')
        for i in range(len(peak_accel_list)):
            ax4.plot([float(x) for x in peak_accel_list[i][:, 0]], control_nodes, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][0], alpha=0.3)
        ax4.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax4.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax4.set_xlim([0, 5.0])

        plt.tight_layout()
        self._save_plot(output_directory, plot_label)


    def plot_slf_model(self,
                       out,
                       cache,
                       xlabel,
                       output_directory=None,
                       plot_label='slf'):

        """
        Generate a plot to visualize the Storey Loss Function (SLF) model output.

        This function visualizes the storey loss for different realizations of a model
        by plotting the following:
        1. Scatter plot of total storey loss for each realization.
        2. Shaded region representing the 16th to 84th percentiles of the empirical data.
        3. Plot of the median of the empirical data for simulations.
        4. Fitted Storey Loss Function (SLF) curve.

        The plot includes:
        - A scatter plot of the total loss per storey for each realization.
        - A shaded area representing the empirical 16th to 84th percentiles.
        - The median storey loss curve based on simulations.
        - The fitted SLF curve.

        Parameters:
        ----------
        out : dict
            A dictionary containing the results of the model. It should include keys for:
                - 'edp_range': A range of Engineering Demand Parameters (EDP) used in the analysis.
                - 'slf': The fitted Storey Loss Function curve.

        cache : dict
            A dictionary containing cached data, including:
                - 'total_loss_storey': A list of total storey losses for each realization.
                - 'empirical_16th', 'empirical_84th': Empirical data representing the 16th and 84th percentiles.
                - 'empirical_median': Empirical median values of the storey loss for the simulations.

        xlabel : str
            The label for the x-axis, typically representing the Engineering Demand Parameter (EDP) range.

        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.

        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'slf'.

        Returns:
        --------
        None
            This function saves the generated plot for each key in the `cache` dictionary to the specified directory.
        """
        keys_list = list(cache.keys())
        for i, current_key in enumerate(keys_list):
            rlz = len(cache[current_key]['total_loss_storey'])
            total_loss_storey_array = np.array([cache[current_key]['total_loss_storey'][i] for i in range(rlz)])

            fig, ax = plt.subplots(figsize=(8, 6))
            self._set_plot_style(ax, xlabel=xlabel, ylabel='Storey Loss')

            for i in range(rlz):
                ax.scatter(out[current_key]['edp_range'], total_loss_storey_array[i, :], color=self.colors['gem'][3], s=self.marker_sizes['small'], alpha=0.5)

            ax.fill_between(out[current_key]['edp_range'], cache[current_key]['empirical_16th'], cache[current_key]['empirical_84th'], color='gray', alpha=0.3, label=r'16$^{\text{th}}$-84$^{\text{th}}$ Percentile')
            ax.plot(out[current_key]['edp_range'], cache[current_key]['empirical_median'], lw=self.line_widths['medium'], color='blue', label='Simulations - Median')
            ax.plot(out[current_key]['edp_range'], out[current_key]['slf'], color='black', lw=self.line_widths['medium'], label='SLF - Fitted')

            ax.legend(fontsize=self.font_sizes['legend'])
            self._save_plot(output_directory, f"{plot_label}_{current_key}")

    def plot_modes(self, node_list, mode_shape_vectors, T, export_path=None):
        """
        Plots the undeformed structure (3D, left) and 2D mode shape profiles (right)

        - 3D plot X and Y limits are set to encompass the structure coordinates and a minimum range of [-2,2]
        - 3D plot Z-limit is fixed to start at 0.0
        - Mode shape vectors are normalized by the X-displacement of the top node (max(Z))

        Parameters
        ----------
        node_list : list
            List of node tags
        mode_shape_vectors : list of numpy.ndarray
            Mode shape vectors (one per mode)
        T : list
            List of natural periods corresponding to the modes
        export_path : str, optional
            If provided, saves the figure to this path (e.g., 'modes.png') instead of displaying it
        """

        # Extract number of modes from length of periods list
        num_modes = len(T)

        # Data retrieval and structuring
        node_coords_list = [ops.nodeCoord(tag) for tag in node_list]
        node_coords_undeformed = np.array(node_coords_list)
        element_list = ops.getEleTags()

        # Identify Base Nodes (first node) and Top Nodes (max Z coordinate)
        base_node_tag = node_list[0] if node_list else -1

        X, Y, Z = node_coords_undeformed.T
        z_max = np.max(Z)
        top_node_indices = np.where(Z == z_max)[0]

        # Z-levels for 2D plots (must be unique and ordered for interpolation)
        unique_z_levels = np.unique(Z)
        z_min = np.min(unique_z_levels)
        z_max = np.max(unique_z_levels)

        # --- CALCULATE 3D AXES LIMITS (Enforcing X/Y range [-2, 2] and Z min 0.0) ---
        x_min_data, x_max_data = np.min(X), np.max(X)
        y_min_data, y_max_data = np.min(Y), np.max(Y)
        z_min_data = np.min(Z)

        epsilon = 1e-6 # Small buffer for axes with zero extent

        # X Limits: Must span at least [-2, 2] and cover the data range plus a buffer
        x_min_3d = min(x_min_data, -2.0)
        x_max_3d = max(x_max_data, 2.0)
        x_range = x_max_3d - x_min_3d
        x_lim_3d = (x_min_3d - 0.05 * x_range, x_max_3d + 0.05 * x_range)
        if np.isclose(x_range, 0.0): # Handle case where all X coords are the same
            x_lim_3d = (x_min_3d - epsilon, x_max_3d + epsilon)

        # Y Limits: Must span at least [-2, 2] and cover the data range plus a buffer
        y_min_3d = min(y_min_data, -2.0)
        y_max_3d = max(y_max_data, 2.0)
        y_range = y_max_3d - y_min_3d
        y_lim_3d = (y_min_3d - 0.05 * y_range, y_max_3d + 0.05 * y_range)
        if np.isclose(y_range, 0.0): # Handle case where all Y coords are the same
            y_lim_3d = (y_min_3d - epsilon, y_max_3d + epsilon)

        # Z Limits: Force minimum to 0.0
        z_lim_3d = (max(0.0, z_min_data), z_max * 1.1)

        # Create Figure and GridSpec Layout
        fig = plt.figure(figsize=(18, 10), facecolor='white')
        gs = gridspec.GridSpec(num_modes, 3, figure=fig, width_ratios=[2, 0.1, 1], wspace=0.1)

        # Set figure-wide title for 2D plots
        title_x_pos = 0.835 # Position centered over the right column
        fig.suptitle('2D Mode Shapes: Deformed Profile (X-Z View)',
                     fontsize=16,
                     weight='bold',
                     color='black',
                     y=0.95,
                     x=title_x_pos)

        # Plot the 3D Axes
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax3d.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # 3D Aesthetics
        ax3d.set_xlabel('X-Direction [m]', fontsize=14, color='black', labelpad=10)
        ax3d.set_ylabel('Y-Direction [m]', fontsize=14, color='black', labelpad=10)
        ax3d.set_zlabel('Z-Direction [m]', fontsize=14, color='black', labelpad=10)
        ax3d.grid(True, linestyle=':', alpha=0.6, color='gray')
        ax3d.view_init(elev=20, azim=-60)

        # Set the corrected limits
        ax3d.set_xlim(x_lim_3d); ax3d.set_ylim(y_lim_3d); ax3d.set_zlim(z_lim_3d)
        ax3d.set_title('3D Undeformed Structure', fontsize=16, weight='bold', color='black', pad=15)

        # Plot Undeformed Nodes (Black markers)
        for i, node_tag in enumerate(node_list):
            x, y, z = node_coords_undeformed[i]
            marker_style = 's' if node_tag == base_node_tag else 'o'
            marker_size = 200 if node_tag == base_node_tag else 150
            ax3d.scatter(x, y, z, marker=marker_style, s=marker_size, color='black', zorder=2)

        # Plot Undeformed Elements (Solid Blue Line)
        for ele_tag in element_list:
            ele_nodes_tags = ops.eleNodes(ele_tag)
            if len(ele_nodes_tags) == 2:
                idx_i = node_list.index(ele_nodes_tags[0])
                idx_j = node_list.index(ele_nodes_tags[1])

                x_u = [node_coords_undeformed[idx_i, 0], node_coords_undeformed[idx_j, 0]]
                y_u = [node_coords_undeformed[idx_i, 1], node_coords_undeformed[idx_j, 1]]
                z_u = [node_coords_undeformed[idx_i, 2], node_coords_undeformed[idx_j, 2]]

                ax3d.plot(x_u, y_u, z_u, color='blue', linewidth=1.5, linestyle='-', alpha=0.7, zorder=1)

        # Normalization and 2D plot setup
        normalized_mode_vectors = []
        for mode_vec in mode_shape_vectors:
            top_node_disp_x = mode_vec[top_node_indices, 0]
            max_top_disp = np.max(np.abs(top_node_disp_x))

            if max_top_disp != 0:
                normalized_vec = mode_vec / max_top_disp
            else:
                normalized_vec = mode_vec
            normalized_mode_vectors.append(normalized_vec)

        deformed_color = 'blue'
        max_disp_for_plotting = 1.0
        x_lim_2d = (-max_disp_for_plotting * 1.5, max_disp_for_plotting * 1.5)
        z_lim_2d = (z_min - 0.5, z_max + 0.5)

        # Iterate through modes for 2D profile plot

        for mode_idx, mode_vector in enumerate(normalized_mode_vectors):
            mode_num = mode_idx + 1
            period = T[mode_idx]

            ax2d = fig.add_subplot(gs[mode_idx, 2])

            # Extract 2D Plot Data (X-displacement vs Z-height)
            node_displacements_x = []
            for z_level in unique_z_levels:
                z_indices = np.where(Z == z_level)[0]
                node_displacements_x.append(np.mean(mode_vector[z_indices, 0]))

            # Interpolation for deformed mode shape function
            N_z = len(unique_z_levels)
            if N_z < 3:
                interpolation_kind = 'linear'
            elif N_z == 3:
                interpolation_kind = 'quadratic'
            else:
                interpolation_kind = 'cubic'

            # 1. Undeformed Reference Line (Solid Gray Line)
            ax2d.plot([0] * N_z, unique_z_levels, color='gray', linewidth=3.0, linestyle='-', alpha=0.7, zorder=1)

            # 2. Plot Undeformed Nodes (Black Square/Circle at X=0)
            for i, node_tag in enumerate(node_list):
                z_u = node_coords_undeformed[i, 2]
                if z_u not in unique_z_levels: continue

                marker_style = 's' if node_tag == base_node_tag else 'o'
                marker_size = 80 if node_tag == base_node_tag else 50

                ax2d.scatter(0, z_u, marker=marker_style, s=marker_size, color='black', edgecolor='black', linewidth=0.5, zorder=2)

            # 3. Smooth Deformed Profile (Fixed Blue Line)
            f_interp = interp1d(unique_z_levels, node_displacements_x, kind=interpolation_kind)
            Z_smooth = np.linspace(z_min, z_max, 100)
            X_smooth = f_interp(Z_smooth)

            ax2d.plot(X_smooth, Z_smooth, color=deformed_color, linewidth=3.0, linestyle='-', zorder=4)

            # Plot Deformed Nodes (Black square/circle at displaced position)
            for i, node_tag in enumerate(node_list):
                z_u = node_coords_undeformed[i, 2]
                if z_u not in unique_z_levels: continue

                z_idx = np.where(unique_z_levels == z_u)[0][0]
                x_disp_at_z = node_displacements_x[z_idx]

                marker_style = 's' if node_tag == base_node_tag else 'o'
                marker_size = 80 if node_tag == base_node_tag else 50

                ax2d.scatter(x_disp_at_z, z_u, marker=marker_style, s=marker_size, color='black', edgecolor='black', linewidth=0.5, zorder=5)


            # 2D Plot Aesthetics and Labels
            title_text = f'Mode {mode_num}, $T_{{{mode_num}}} = {period:.3f}$ s'
            ax2d.set_title(title_text, fontsize=12, color='black', pad=5)

            ax2d.set_ylim(z_lim_2d)
            ax2d.grid(True, linestyle=':', alpha=0.5)
            ax2d.set_xlim(x_lim_2d)
            ax2d.set_ylabel('Z-Height [m]', fontsize=10)

            # X-Label placement
            if mode_idx < num_modes - 1:
                ax2d.tick_params(labelbottom=False)
                ax2d.set_xlabel(' ', fontsize=10)
            else:
                ax2d.set_xlabel('X-Displacement (Normalized)', fontsize=10)

            # Consistent Y-axis tick and label placement
            if mode_idx > 0:
                 ax2d.sharey(fig.axes[2])
                 ax2d.tick_params(labelleft=False)

        # Align labels again after tight_layout to finalize position
        fig.align_labels()
        plt.subplots_adjust(top=0.9)

        # Export figure
        if export_path:
            print(f"Saving figure to {export_path}")
            plt.savefig(export_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()


    def animate_spo(self, spo_top_disp, spo_rxn, spo_disps, spo_midr, nodeList, elementList, push_dir, save_path):
        """Generates and saves the SPO animation using FuncAnimation."""
        deform_factor = 1 # Scaling factor for visualization
        # spo_midr is now passed in as an argument, so its length determines the number of frames.
        num_frames = len(spo_top_disp)

        # ------------------ Data Processing ------------------
        # Get undeformed coordinates once
        NodeCoordListX_und = [ops.nodeCoord(tag, 1) for tag in nodeList]
        NodeCoordListY_und = [ops.nodeCoord(tag, 2) for tag in nodeList]
        NodeCoordListZ_und = [ops.nodeCoord(tag, 3) for tag in nodeList]

        # Determine the plotting coordinates based on push_dir for the 2D view
        if push_dir == 1:
            plot_coords_und = (NodeCoordListX_und, NodeCoordListZ_und)
            x_label_model = 'X-Direction [m]'
            y_label_model = 'Z-Direction [m]'
        elif push_dir == 2:
            plot_coords_und = (NodeCoordListY_und, NodeCoordListZ_und)
            x_label_model = 'Y-Direction [m]'
            y_label_model = 'Z-Direction [m]'
        elif push_dir == 3:
            plot_coords_und = (NodeCoordListZ_und, NodeCoordListX_und)
            x_label_model = 'Z-Direction [m]'
            y_label_model = 'X-Direction [m]'
        else:
            plot_coords_und = (NodeCoordListX_und, NodeCoordListZ_und)
            x_label_model = 'X-Direction [m]'
            y_label_model = 'Z-Direction [m]'

        # Max coordinate for consistent plot limits
        max_abs_coord_x = np.max(np.abs(plot_coords_und[0]))
        max_abs_coord_y = np.max(np.abs(plot_coords_und[1]))
        model_x_lim = (-max_abs_coord_x * 3.0, max_abs_coord_x * 3.0)
        model_y_lim = (0, max_abs_coord_y * 1.5)

        # Max Interstorey Drift History (passed as spo_midr)
        max_drift_history = np.maximum.accumulate(spo_midr)

        # ------------------ Initialize the Figure and Subplots ------------------
        fig = plt.figure(figsize=(16, 8))

        # Layout: (1, 2, 1) is big left plot; (2, 2, 2) is top right; (2, 2, 4) is bottom right
        ax_model = fig.add_subplot(1, 2, 1)
        ax_curve = fig.add_subplot(2, 2, 2)
        ax_drift = fig.add_subplot(2, 2, 4)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Store the number of static (undeformed) artists for easy cleanup in update()
        num_static_lines = len(elementList)
        num_static_collections = 1 # For the single undeformed nodes scatter plot

        # ------------------ Set up static plot elements ------------------
        # 2D Model Plot (Undeformed - static gray background)
        ax_model.scatter(plot_coords_und[0], plot_coords_und[1],
                          marker='o', s=50, color='gray', alpha=0.5, label='Undeformed Nodes')
        for eleTag in elementList:
            [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
            i = nodeList.index(NodeItag)
            j = nodeList.index(NodeJtag)
            x_und = [plot_coords_und[0][i], plot_coords_und[0][j]]
            y_und = [plot_coords_und[1][i], plot_coords_und[1][j]]
            ax_model.plot(x_und, y_und, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        ax_model.set_xlabel(x_label_model)
        ax_model.set_ylabel(y_label_model)
        ax_model.set_title('Deformed Model Shape')
        ax_model.set_xlim(model_x_lim)
        ax_model.set_ylim(model_y_lim)
        ax_model.grid(True)

        # Pushover Curve (Base Shear vs Top Disp)
        ax_curve.set_xlabel('Top Displacement [m]')
        ax_curve.set_ylabel('Base Shear [kN]')
        ax_curve.set_title('Pushover Curve (Base Shear vs Top Disp)')
        ax_curve.plot(spo_top_disp, spo_rxn, 'gray', linewidth=2, alpha=0.5, label='Static Curve')
        curve_anim, = ax_curve.plot([], [], 'blue', linewidth=2, label='Current Step')
        ax_curve.legend()
        ax_curve.set_xlim(np.min(spo_top_disp)*1.1 if np.min(spo_top_disp) < 0 else 0, np.max(spo_top_disp)*1.1)
        ax_curve.set_ylim(np.min(spo_rxn)*1.1, np.max(spo_rxn)*1.1)
        ax_curve.grid(True)

        # Max Drift vs Base Shear
        ax_drift.set_xlabel('Max Interstorey Drift Ratio [%]')
        ax_drift.set_ylabel('Base Shear [kN]')
        ax_drift.set_title('Base Shear vs Max Interstorey Drift Ratio')
        drift_anim, = ax_drift.plot([], [], 'green', linewidth=2, label='Current Max Drift')
        ax_drift.legend()
        # Use spo_midr limits
        ax_drift.set_xlim(0, np.max(max_drift_history) * 1.2)
        ax_drift.set_ylim(np.min(spo_rxn)*1.1, np.max(spo_rxn)*1.1)
        ax_drift.grid(True)


        # ------------------ The update function for FuncAnimation ------------------
        def update(frame):
            nonlocal num_static_lines, num_static_collections

            # --- 2D Model Plot Cleanup ---
            # Remove dynamically drawn lines (deformed elements) from the LAST frame
            while len(ax_model.lines) > num_static_lines:
                ax_model.lines[-1].remove()

            # Remove dynamically drawn collections (deformed nodes) from the LAST frame
            while len(ax_model.collections) > num_static_collections:
                ax_model.collections[-1].remove()

            # 2D Model Plot Redraw (Deformed Shape)
            # Get displacement data for the current frame
            current_disps_floor = spo_disps[frame]
            # Include ground floor (index 0) displacement = 0
            full_node_disps = np.insert(current_disps_floor, 0, 0, axis=0)

            # Calculate deformed coordinates based on push_dir
            if push_dir == 1: # X-Z plane
                X_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (X_def, Z_def)
            elif push_dir == 2: # Y-Z plane
                Y_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Y_def, Z_def)
            elif push_dir == 3: # Z-X plane
                Z_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                X_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Z_def, X_def)
            else:
                 plot_coords_def = plot_coords_und

            # Plot Deformed Shape (Blue)
            ax_model.scatter(plot_coords_def[0], plot_coords_def[1],
                              marker='o', s=50, color='blue', label='Deformed Nodes')
            for eleTag in elementList:
                [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                i = nodeList.index(NodeItag)
                j = nodeList.index(NodeJtag)
                x_def = [plot_coords_def[0][i], plot_coords_def[0][j]]
                y_def = [plot_coords_def[1][i], plot_coords_def[1][j]]
                ax_model.plot(x_def, y_def, color='blue', linewidth=1.5)

            ax_model.set_title(f'Frame: {frame}/{num_frames-1} (Scale: {deform_factor}x)')

            # Pushover Curve Update
            curve_anim.set_data(spo_top_disp[:frame+1], spo_rxn[:frame+1])

            # Max Drift vs Base Shear Update (Using spo_midr)
            drift_anim.set_data(spo_midr[:frame+1], spo_rxn[:frame+1])

            # Return the artists that were modified
            return curve_anim, drift_anim

        # Create the animation object
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

        # Save the animation
        print(f"\nSaving animation to: {save_path}")

        if save_path.lower().endswith('.gif'):
            ani.save(save_path, writer='pillow', dpi=150)
        elif save_path.lower().endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', dpi=200)
        else:
            print("WARNING: Animation path extension not recognized (.gif or .mp4 recommended). Saving as default.")
            ani.save(save_path, dpi=150)

        plt.close(fig)

    # Helper function to create and save the animation for cyclic pushover analysis
    def animate_cpo(self, cpo_dict, nodeList, elementList, push_dir, save_path):
        """
        Generates and saves the CPO animation using FuncAnimation, showing:
        1. Deformed model shape.
        2. Base shear vs. top displacement (hysteretic curve, spanning negative/positive).
        3. Base shear vs. maximum interstorey drift (newly updated to show hysteresis).

        Parameters
        ----------
        cpo_dict: dict
            The analysis results dictionary returned by do_cpo_analysis.
        nodeList: list
            List of node tags in the model.
        elementList: list
            List of element tags in the model.
        push_dir: int
            Direction of the pushover analysis (1=X, 2=Y, 3=Z).
        save_path: str
            File path to save the animation (e.g., 'cpo_animation.gif' or 'cpo_animation.mp4').
        """

        # ------------------ Data Extraction and Processing ------------------
        cpo_top_disp = cpo_dict['cpo_top_disp']
        cpo_rxn = cpo_dict['cpo_rxn']
        cpo_disps = cpo_dict['cpo_disps']
        cpo_drifts = cpo_dict['cpo_idr']

        deform_factor = 1.0 # Scaling factor for visualization
        num_frames = len(cpo_top_disp)

        # Calculate the maximum interstorey drift at each step:
        # Find the drift (with sign) of the floor that experiences the maximum absolute drift at this step.
        max_drift_indices = np.argmax(np.abs(cpo_drifts), axis=1)
        governing_drift_history = cpo_drifts[np.arange(num_frames), max_drift_indices]

        # Max absolute drift for setting limits
        max_drift_limit = np.max(np.abs(governing_drift_history))

        # Get undeformed coordinates once
        NodeCoordListX_und = [ops.nodeCoord(tag, 1) for tag in nodeList]
        NodeCoordListY_und = [ops.nodeCoord(tag, 2) for tag in nodeList]
        NodeCoordListZ_und = [ops.nodeCoord(tag, 3) for tag in nodeList]

        # Determine the plotting coordinates based on push_dir for the 2D view
        if push_dir == 1:
            plot_coords_und = (NodeCoordListX_und, NodeCoordListZ_und)
            x_label_model = 'X-Direction [m]'
            y_label_model = 'Z-Direction [m]'
        elif push_dir == 2:
            plot_coords_und = (NodeCoordListY_und, NodeCoordListZ_und)
            x_label_model = 'Y-Direction [m]'
            y_label_model = 'Z-Direction [m]'
        elif push_dir == 3:
            plot_coords_und = (NodeCoordListZ_und, NodeCoordListX_und)
            x_label_model = 'Z-Direction [m]'
            y_label_model = 'X-Direction [m]'
        else:
            plot_coords_und = (NodeCoordListX_und, NodeCoordListZ_und)
            x_label_model = 'X-Direction [m]'
            y_label_model = 'Z-Direction [m]'

        # Max coordinate for consistent plot limits
        max_abs_coord_x = np.max(np.abs(plot_coords_und[0]))
        max_abs_coord_y = np.max(np.abs(plot_coords_und[1]))
        model_x_lim = (-max_abs_coord_x * 3.0, max_abs_coord_x * 3.0)
        model_y_lim = (0, max_abs_coord_y * 1.5)

        # ------------------ Initialize the Figure and Subplots ------------------
        fig = plt.figure(figsize=(16, 8))

        # Layout: (1, 2, 1) is big left plot; (2, 2, 2) is top right; (2, 2, 4) is bottom right
        ax_model = fig.add_subplot(1, 2, 1)
        ax_curve = fig.add_subplot(2, 2, 2)
        ax_drift = fig.add_subplot(2, 2, 4)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Store the number of static (undeformed) artists for easy cleanup in update()
        num_static_lines = len(elementList)
        num_static_collections = 1 # For the single undeformed nodes scatter plot

        # ------------------ Set up static plot elements (Undeformed Shape) ------------------
        ax_model.scatter(plot_coords_und[0], plot_coords_und[1],
                         marker='o', s=50, color='gray', alpha=0.5, label='Undeformed Nodes')
        for eleTag in elementList:
            try:
                [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                i = nodeList.index(NodeItag)
                j = nodeList.index(NodeJtag)
            except:
                continue

            x_und = [plot_coords_und[0][i], plot_coords_und[0][j]]
            y_und = [plot_coords_und[1][i], plot_coords_und[1][j]]
            ax_model.plot(x_und, y_und, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        ax_model.set_xlabel(x_label_model)
        ax_model.set_ylabel(y_label_model)
        ax_model.set_title('Deformed Model Shape (Cyclic Pushover)')
        ax_model.set_xlim(model_x_lim)
        ax_model.set_ylim(model_y_lim)
        ax_model.grid(True)

        # Hysteretic Curve (Base Shear vs Top Disp)
        ax_curve.set_xlabel('Top Displacement [m]')
        ax_curve.set_ylabel('Base Shear [kN]')
        ax_curve.set_title('Hysteretic Curve (Base Shear vs Top Disp)')
        ax_curve.plot(cpo_top_disp, cpo_rxn, 'gray', linewidth=1, alpha=0.5, label='History')
        curve_anim, = ax_curve.plot([], [], 'blue', linewidth=2, label='Current Step')
        ax_curve.legend(loc='lower right')

        # Set limits for cyclic analysis (must cover negative space)
        max_x_curve = np.max(np.abs(cpo_top_disp)) * 1.1
        max_y_curve = np.max(np.abs(cpo_rxn)) * 1.1
        ax_curve.set_xlim(-max_x_curve, max_x_curve)
        ax_curve.set_ylim(-max_y_curve, max_y_curve)
        ax_curve.grid(True)

        # Governing Drift Hysteresis (Base Shear vs MIDR) - UPDATED
        ax_drift.set_xlabel('Maximum Interstorey Drift [-]')
        ax_drift.set_ylabel('Base Shear [kN]')
        ax_drift.set_title('Hysteretic Curve (Base Shear vs MIDR)')
        # Plot full history in gray
        ax_drift.plot(governing_drift_history, cpo_rxn, 'gray', linewidth=1, alpha=0.5, label='History')
        # Plot current step in green
        drift_anim, = ax_drift.plot([], [], 'green', linewidth=2, label='Current Step')
        ax_drift.legend(loc='lower right')

        # Set limits for cyclic analysis (must cover negative space for drift) - UPDATED
        ax_drift.set_xlim(-max_drift_limit * 1.1, max_drift_limit * 1.1)
        ax_drift.set_ylim(-max_y_curve, max_y_curve)
        ax_drift.grid(True)


        # ------------------ The update function for FuncAnimation ------------------
        def update(frame):
            nonlocal num_static_lines, num_static_collections

            # --- 2D Model Plot Cleanup ---
            while len(ax_model.lines) > num_static_lines:
                ax_model.lines[-1].remove()

            while len(ax_model.collections) > num_static_collections:
                ax_model.collections[-1].remove()

            # --- 2D Model Plot Redraw (Deformed Shape) ---
            current_disps_floor = cpo_disps[frame]
            # Include ground floor (index 0) displacement = 0
            full_node_disps = np.insert(current_disps_floor, 0, 0, axis=0)

            # Calculate deformed coordinates based on push_dir
            if push_dir == 1: # X-Z plane
                X_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (X_def, Z_def)
            elif push_dir == 2: # Y-Z plane
                Y_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Y_def, Z_def)
            elif push_dir == 3: # Z-X plane
                Z_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                X_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Z_def, X_def)
            else:
                plot_coords_def = plot_coords_und

            # Plot Deformed Shape (Blue)
            ax_model.scatter(plot_coords_def[0], plot_coords_def[1],
                             marker='o', s=50, color='blue', label='Deformed Nodes')
            for eleTag in elementList:
                try:
                    [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                    i = nodeList.index(NodeItag)
                    j = nodeList.index(NodeJtag)
                except:
                    continue

                x_def = [plot_coords_def[0][i], plot_coords_def[0][j]]
                y_def = [plot_coords_def[1][i], plot_coords_def[1][j]]
                ax_model.plot(x_def, y_def, color='blue', linewidth=1.5)

            ax_model.set_title(f'Frame: {frame}/{num_frames-1} (Scale: {deform_factor}x)')

            # --- Hysteretic Curve Update (Top Disp) ---
            curve_anim.set_data(cpo_top_disp[:frame+1], cpo_rxn[:frame+1])

            # --- Governing Drift Hysteresis Update ---
            drift_anim.set_data(governing_drift_history[:frame+1], cpo_rxn[:frame+1])

            # Return the artists that were modified
            return curve_anim, drift_anim

        # Create the animation object
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

        # Save the animation
        print(f"\nSaving animation to: {save_path}")

        if save_path.lower().endswith('.gif'):
            ani.save(save_path, writer='pillow', dpi=300)
        elif save_path.lower().endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', dpi=300)
        else:
            print("WARNING: Animation path extension not recognized (.gif or .mp4 recommended). Saving as default.")
            ani.save(save_path, dpi=300)

        plt.close(fig)

    def animate_nrha(self,
                     control_nodes,
                     acc,
                     dts,
                     nrha_disps,
                     nrha_accels,
                     drift_thresholds=None,
                     export_path=None):
        """
        Animate the seismic response for a nonlinear time-history analysis (NRHA).
        Automatically infers storey heights and applies cumulative color and annotation updates.
        """

        # --- Compute storey heights from Z coordinates ---
        node_z_coords = np.array([ops.nodeCoord(n, 3) for n in control_nodes])
        sorted_idx = np.argsort(node_z_coords)
        control_nodes = np.array(control_nodes)[sorted_idx]
        node_z_coords = node_z_coords[sorted_idx]
        storey_heights = np.diff(node_z_coords)

        if np.any(storey_heights <= 1e-6):
            print("⚠️ Warning: Zero or near-zero storey height detected")

        # --- Initialize Figure and Axes ---
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.6])

        ax1 = fig.add_subplot(gs[0])  # Displacement
        ax2 = fig.add_subplot(gs[1])  # Acceleration
        ax3 = fig.add_subplot(gs[2])  # Ground motion

        # --- Plot undeformed shape (gray background) ---
        ax1.plot(np.zeros_like(control_nodes), node_z_coords, color='gray', lw=1.0, alpha=0.6)
        ax2.plot(np.zeros_like(control_nodes), node_z_coords, color='gray', lw=1.0, alpha=0.6)

        # --- Plot ground motion background ---
        ax3.plot(dts, acc, color='lightgray', lw=1.0, alpha=0.6)

        # --- Initialize Lines ---
        line_disp, = ax1.plot([], [], 'o-', color="blue", lw=2.0, markersize=6)
        line_acc, = ax2.plot([], [], 'o-', color="red", lw=2.0, markersize=6)
        line_acc_time, = ax3.plot([], [], color="green", lw=1.8)

        # --- Grid and formatting ---
        for ax in (ax1, ax2, ax3):
            ax.grid(True, ls='--', alpha=0.4)

        ax1.set_title("Floor Displacements (m)")
        ax1.set_xlabel("Displacement (m)")
        ax1.set_ylabel("Elevation (m)")
        ax1.set_ylim(node_z_coords[0] - 0.1 * storey_heights.mean(),
                     node_z_coords[-1] + 0.1 * storey_heights.mean())

        ax2.set_title("Floor Accelerations (g)")
        ax2.set_xlabel("Acceleration (g)")
        ax2.set_ylabel("Elevation (m)")
        ax2.set_ylim(node_z_coords[0] - 0.1 * storey_heights.mean(),
                     node_z_coords[-1] + 0.1 * storey_heights.mean())
        ax2.set_xlim(-5.0, 5.0)

        ax3.set_title("Input Ground Motion")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (g)")
        ax3.set_xlim(0, dts[-1])
        ax3.set_ylim(np.floor(acc.min()), np.ceil(acc.max()))

        # --- Damage state colors ---
        damage_state_colors = ['#1E88E5', '#43A047', '#FDD835', '#FB8C00', '#E53935']

        # --- Trackers for cumulative state ---
        max_damage_state = 0
        max_drift_val = 0.0
        max_accel_val = 0.0

        # --- Persistent Annotations ---
        drift_annot = ax1.text(0.02, 0.9, "", transform=ax1.transAxes,
                               fontsize=9, color="black", bbox=dict(facecolor='white', alpha=0.5))
        accel_annot = ax2.text(0.02, 0.9, "", transform=ax2.transAxes,
                               fontsize=9, color="black", bbox=dict(facecolor='white', alpha=0.5))

        # --- Update function ---
        def update(frame):
            nonlocal max_damage_state, max_drift_val, max_accel_val

            disp_values = nrha_disps[frame, :]
            accel_values = nrha_accels[frame, :] / 9.81  # convert to g

            line_disp.set_data(disp_values, node_z_coords)
            line_acc.set_data(accel_values, node_z_coords)
            line_acc_time.set_data(dts[:frame], acc[:frame])

            # --- Interstory drift ---
            interstory_drifts = np.abs(np.diff(disp_values)) / storey_heights
            current_drift = np.max(interstory_drifts)
            current_accel = np.max(np.abs(accel_values))

            # --- Update cumulative maxima ---
            max_drift_val = max(max_drift_val, current_drift)
            max_accel_val = max(max_accel_val, current_accel)

            # --- Update annotations continuously ---
            drift_annot.set_text(f"Max drift: {max_drift_val:.4f}")
            accel_annot.set_text(f"Max accel: {max_accel_val:.3f} g")

            # --- Cumulative damage state logic ---
            if drift_thresholds is not None and len(drift_thresholds) > 0:
                # Determine the current frame's damage state for all floors
                frame_states = np.sum(interstory_drifts[:, None] > np.array(drift_thresholds), axis=1)
                # Take the highest among all floors
                frame_max_state = frame_states.max() if len(frame_states) > 0 else 0
                # Update cumulative maximum damage state
                if frame_max_state > max_damage_state:
                    max_damage_state = frame_max_state
                color = damage_state_colors[min(max_damage_state, len(damage_state_colors)-1)]
            else:
                color = "blue"

            # Always apply the cumulative damage color
            line_disp.set_color(color)
            line_acc.set_color(color)
            line_acc_time.set_color(color)

            return line_disp, line_acc, line_acc_time, drift_annot, accel_annot

        # Create animation
        ani = FuncAnimation(fig, update, frames=len(dts), interval=10, blit=False, repeat=False)
        plt.tight_layout()

        # Save or show
        if export_path:
            print(f"\nSaving animation to: {export_path}")
            try:
                if export_path.lower().endswith('.gif'):
                    ani.save(export_path, writer='pillow', dpi=self.resolution)
                elif export_path.lower().endswith('.mp4'):
                    try:
                        ani.save(export_path, writer='ffmpeg', dpi=self.resolution)
                    except Exception:
                        print("⚠️ FFmpeg not found — falling back to Pillow GIF encoding.")
                        ani.save(export_path.replace('.mp4', '.gif'), writer='pillow', dpi=self.resolution)
                else:
                    print("WARNING: Unknown extension (.gif or .mp4 recommended). Saving as GIF.")
                    ani.save(export_path + ".gif", writer='pillow', dpi=self.resolution)
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Animation save failed: {e}")
                plt.show()
        else:
            plt.show()

        return ani

    ###############################################################################################################
    #                                                                                                             #
    #                                         PLOT NLTHA OUTPUT                                                   #
    #                                                                                                             #
    ###############################################################################################################

    def plot_ida_analysis(self,
                          ida_dict,
                          imt_label,
                          edp_label,
                          title=None,
                          pFlag = True,
                          export_path = None):

        """
        Visualizes the Incremental Dynamic Analysis (IDA) suite and statistical summary.

        This method generates a comprehensive IDA plot featuring individual ground motion
        record curves as a background "cloud" and overlays the statistical response
        percentiles. It is designed to provide an immediate visual assessment of
        structural performance across a range of intensities.

        Parameters
        ----------
        ida_dict : dict
            The processed results dictionary returned by `do_incremental_dynamic_analysis`.
            Must contain the following nested keys:
            - ['ida_inputs']['raw_curves']: List of (IM, EDP) pairs for each record.
            - ['ida_inputs']['damage_thresholds']: EDP values for limit states.
            - ['stats']: Dictionary containing 'fitted_edps', 'median_im', 'p16_im', and 'p84_im'.
            - ['ida_inputs']['imt_key']: The label of the intensity measure used.

        title : str, optional, default=None
            A custom title for the figure. If not provided, a default title
            incorporating the Intensity Measure (IM) label is used.

        Returns
        -------
        None
            Displays the matplotlib figure.

        Notes
        -----
        - The statistical lines represent the interpolated "flatlining" behavior, ensuring
          the curves correctly represent global dynamic instability (collapse) where
          the intensity increases but the structural capacity is exhausted.
        - The individual gray curves are plotted with low alpha transparency to allow
          the summary percentiles and vertical thresholds to remain the focal point.
        """

        # Initialise the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        inputs = ida_dict['ida_inputs']
        stats  = ida_dict['stats']

        # Plot Individual Raw IDA Curves (Background Cloud)
        for i, curve in enumerate(inputs['raw_curves']):
            ax.plot(curve['edp'], curve['im'], '-o', color='gray', alpha=0.15,
                    lw=self.line_widths['thin'],
                    markersize=3,
                    label='Individual Record' if i == 0 else "")

        # Plot Statistical Percentile Lines with requested colors
        # 16th Percentile - Green (using class colors if applicable or specific request)
        ax.plot(stats['fitted_edps'], stats['p16_im'],
                color='green', ls='--', lw=self.line_widths['medium'],
                label='$16^{th}$ Percentile')

        # Median (50th) - Blue Solid (using GEM color or standard blue)
        ax.plot(stats['fitted_edps'], stats['median_im'],
                color='blue', lw=self.line_widths['thick'],
                label='$50^{th}$ Percentile (Median)')

        # 84th Percentile - Blue Dashed
        ax.plot(stats['fitted_edps'], stats['p84_im'],
                color='red', ls='--', lw=self.line_widths['medium'],
                label='$84^{th}$ Percentile')

        # Add Vertical Lines for Damage Thresholds
        # Uses the 'fragility' color palette from the class
        ds_colors = self.colors['fragility']

        for i, thresh in enumerate(inputs['damage_thresholds']):
            color = ds_colors[i % len(ds_colors)]
            ax.axvline(thresh,
                       color=color,
                       ls=':',
                       alpha=0.8,
                       lw=self.line_widths['medium'],
                       label=f'$DS_{{{i+1}}}$ Threshold: {thresh:.3f}')


        # Apply consistent Class Styling
        default_title = f"IDA: {imt_label} vs {edp_label}"
        self._set_plot_style(ax,
                             title=title if title else default_title,
                             xlabel= edp_label,
                             ylabel= imt_label)

        # Final layout adjustments
        ax.set_xlim([0, np.max(stats['fitted_edps'])])
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=self.font_sizes['legend'])
        plt.tight_layout()

        # Save or Show
        if pFlag:
            if export_path:
                # Save to disk
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution, bbox_inches='tight')
                plt.show()
            # Show if no path OR if you want to see it after saving
            if not export_path:
                # Display but do not save to disk
                plt.show()
            else:
                # Close the plot to free memory after saving if not showing
                plt.close()

    ###############################################################################################################
    #                                                                                                             #
    #                                         PLOT FRAGILITY OUTPUT                                               #
    #                                                                                                             #
    ###############################################################################################################

    def plot_fragility_from_ca(self,
                               cloud_dict,
                               imt_label,
                               title,
                               pFlag = True,
                               export_path=None):

        """
        Generate a fragility analysis plot showing the probability of exceedance (PoE)
        for various damage states as a function of Peak Ground Acceleration (PGA).

        This method plots fragility curves for multiple damage states based on the
        fragility data in the input dictionary. Each curve represents the probability
        of exceedance for a specific damage state, and the plot is presented in a
        linear scale for both axes.

        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the fragility analysis. The dictionary
            should have the following keys:
                - 'fragility': A dictionary containing:
                    - 'intensities': List or array of intensity values (e.g., PGA levels).
                    - 'poes': 2D array of probabilities of exceedance for each damage state.
                - 'medians': List of medians for each damage state.

        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved
            in the current working directory.

        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is
            'fragility_plot'.

        xlabel : str, optional
            The label for the x-axis. Default is 'Peak Ground Acceleration, PGA [g]'.

        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.

        """

        # Setup data
        frag_data   = cloud_dict['fragility']
        intensities = frag_data['intensities']
        poes        = frag_data['poes']
        medians     = frag_data['medians']

        # Initialise plot
        fig, ax = plt.subplots(figsize=(12, 6))
        self._set_plot_style(ax,
                            xlabel=imt_label,
                            ylabel='Probability of Exceedance')

        for i in range(poes.shape[1]):
            color = self.colors['fragility'][i % len(self.colors['fragility'])]
            ax.plot(intensities, poes[:, i], linestyle='solid', color=color, lw=self.line_widths['thick'], label=f'$DS_{{{i+1}}}$ ($\mu_{{{i+1}}}={medians[i]:.2f}$g)')

        ax.set_xlim([0, 5])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=self.font_sizes['legend'], loc='lower right', frameon=True)
        plt.tight_layout()

        # Save or Show
        if pFlag:
            if export_path:
                # Save to disk
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution, bbox_inches='tight')
            # Show if no path OR if you want to see it after saving
            if not export_path:
                # Display but do not save to disk
                plt.show()
            else:
                # Close the plot to free memory after saving if not showing
                plt.close()

    def plot_fragility_from_ida(self,
                                ida_dict,
                                imt_label,
                                title = None,
                                pFlag = True,
                                export_path = None):

        """
        Generate a fragility analysis plot showing the probability of exceedance (PoE)
        for multiple damage states derived from IDA results.
        This method visualizes the lognormal fragility functions fitted during the
        Incremental Dynamic Analysis. Each curve represents the probability that a
        specific engineering demand parameter (EDP) threshold (e.g., drift limit)
        is exceeded given a specific intensity measure (IM) level.

        Parameters
        ----------
        ida_dict : dict
            A nested dictionary containing the processed IDA results.
            Required structure:
            - 'fragility': A dictionary containing:
                - 'intensities': 1D array of IM levels used for sampling.
                - 'poes': 2D array [n_intensities x n_thresholds] of probabilities.
                - 'medians': List of the estimated median capacities for each state.
            - 'ida_inputs': A dictionary containing:
                - 'imt_key': String label of the intensity measure (e.g., 'Sa(T1)').

        output_directory : str, optional
            The filesystem path where the plot will be saved. If None, the plot
            is displayed but not saved to disk.

        plot_label : str, optional, default='ida_fragility_plot'
            The filename (without extension) for the exported PNG file.

        xlabel : str, optional
            Custom label for the x-axis. If None, the label is automatically
            constructed from the 'imt_key' found in the ida_dict.

        Returns
        -------
        None
            The method renders the plot using Matplotlib and handles saving
            via the internal `_save_plot` utility.
        """

        # Setup Data
        frag_data   = ida_dict['fragility']
        inputs      = ida_dict['ida_inputs']
        intensities = frag_data['intensities']
        poes        = frag_data['poes']        # 2D array [n_intensities x n_thresholds]
        medians     = frag_data['medians']

        # Initialize Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        default_title = f"Fragility Functions for {imt_label}"
        self._set_plot_style(ax,
                             title=title if title else default_title,
                             xlabel=imt_label,
                             ylabel='Probability of Exceedance (PoE)')

        # Plot Each Damage State Curve
        # We cycle through colors defined in self.colors['fragility']
        for i in range(poes.shape[1]):
            color = self.colors['fragility'][i % len(self.colors['fragility'])]
            # Plot the fragility curve
            ax.plot(intensities, poes[:, i], linestyle='solid', color=color, lw=self.line_widths['thick'], label=f'$DS_{{{i+1}}}$ ($\mu_{{{i+1}}}={medians[i]:.2f}$g)')
        ax.set_xlim([0, 5.0])
        ax.set_ylim([0, 1.0]) # Small buffer above 1.0
        ax.legend(fontsize=self.font_sizes['legend'], loc='lower right', frameon=True)
        plt.tight_layout()

        # Save or Show
        if pFlag:
            if export_path:
                # Save to disk
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution, bbox_inches='tight')
                plt.show()
            # Show if no path OR if you want to see it after saving
            if not export_path:
                # Display but do not save to disk
                plt.show()
            else:
                # Close the plot to free memory after saving if not showing
                plt.close()

    ###############################################################################################################
    #                                                                                                             #
    #                                           PLOT VULNERABILITY OUTPUT                                         #
    #                                                                                                             #
    ###############################################################################################################

    def plot_vulnerability_function(self,
                                    intensities,
                                    loss,
                                    cov,
                                    imt_label,
                                    ylabel,
                                    title=None,
                                    pFlag=True,
                                    export_path=None):
            """
            Generate a vulnerability analysis plot featuring Beta distributions and a mean loss curve.

            This method visualizes the uncertainty in loss ratios across different seismic intensities.
            It simulates Beta distributions based on mean loss and CoV, rendering them as
            truncated violin plots (strictly bounded 0-1) to represent the physical limits
            of structural damage.

            Parameters
            ----------
            intensities : list of float
                Intensity Measure (IM) levels (e.g., PGA, Sa) analyzed.
            loss : list of float
                Mean loss ratios (0.0 to 1.0) corresponding to each intensity.
            cov : list of float
                Coefficient of Variation for loss at each intensity.
            imt_label : str
                Label for the X-axis (e.g., 'PGA [g]').
            ylabel : str
                Label for the primary Y-axis loss curve (e.g., 'Mean Damage Ratio').
            title : str, optional
                Custom plot title.
            pFlag : bool, optional, default=True
                If True, the plot is processed (saved/shown).
            export_path : str, optional
                Full path including filename to save the plot. Creates directories if missing.

            Returns
            -------
            None
            """
            # Simulating Beta distributions for each intensity measure
            simulated_data = []
            intensity_labels = []

            for j, mean_loss in enumerate(loss):
                # Beta distribution requires mean in (0, 1)
                mu = np.clip(mean_loss, 0.0001, 0.9999)
                variance = (cov[j] * mu) ** 2

                # Constraint: Variance must be < mu * (1 - mu)
                limit = mu * (1 - mu)
                if variance >= limit:
                    variance = limit * 0.99  # Cap variance to allow distribution fitting

                alpha = mu * (mu * (1 - mu) / variance - 1)
                beta_param = (1 - mu) * (mu * (1 - mu) / variance - 1)

                # Generate samples and clip to ensure physical bounds
                data = np.random.beta(alpha, beta_param, 10000)
                simulated_data.append(data)
                intensity_labels.extend([intensities[j]] * len(data))

            # We name the column 'Loss_Val' for consistent reference in Seaborn
            df_sns = pd.DataFrame({'Intensity Measure': intensity_labels,
                                   'Loss_Val': np.concatenate(simulated_data)})

            # Initialise Plot
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Set plot style
            default_title = f"Vulnerability Function: {imt_label}"
            self._set_plot_style(ax1,
                                 title=title if title else default_title,
                                 xlabel=imt_label,
                                 ylabel="Loss Distribution")

            # Violin plot for Beta distributions
            # We use 'Loss_Val' to match the DataFrame column
            violin = sns.violinplot(x='Intensity Measure', y='Loss_Val', data=df_sns,
                                    density_norm='width', bw_method=0.2,
                                    cut=0,
                                    inner=None,
                                    ax=ax1,
                                    zorder=1,
                                    color='skyblue')

            # Overlay a strip plot for sample density
            sns.stripplot(x='Intensity Measure', y='Loss_Val', data=df_sns, color='black',
                          size=1,
                          alpha=0.2,
                          ax=ax1,
                          zorder=2)

            # Style the primary axis (Distributions)
            ax1.set_ylim(0, 1.0)
            ax1.yaxis.label.set_color('blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Secondary Axis for the Mean Loss Curve
            ax2 = ax1.twinx()
            ax2.plot(range(len(intensities)), loss, marker='s', ls='-', color='red',
                     lw=self.line_widths['medium'], label="Mean Loss Ratio", zorder=5)

            # Style the secondary axis (Loss Curve)
            ax2.set_ylabel(ylabel, color='red', rotation=270, labelpad=20,
                           fontsize=self.font_sizes['labels'], fontname=self.font_name)
            ax2.tick_params(axis='y', labelcolor='red', labelsize=self.font_sizes['ticks'])
            ax2.set_ylim(0, 1.0)

            # Sync X-axis ticks with intensity values
            ax1.set_xticks(range(len(intensities)))
            ax1.set_xticklabels([f"{x:.3f}" for x in intensities], rotation=45, ha='right', fontsize=8)

            # Combined Legend
            beta_patch = mpatches.Patch(color='skyblue', label="Beta Distribution (Uncertainty)")
            ax1.legend(handles=[beta_patch], loc='upper left', fontsize=self.font_sizes['legend'])
            ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.93), fontsize=self.font_sizes['legend'])

            plt.tight_layout()

            # Save or show
            if pFlag:
                if export_path:
                    directory = os.path.dirname(export_path)
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    plt.savefig(export_path, dpi=self.resolution, bbox_inches='tight')
                    print(f"Vulnerability plot saved to: {export_path}")
                    plt.show()
                else:
                    plt.show()
            else:
                plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from openquake.vmtk.units import units
from openquake.vmtk.plotter import plotter

class modeller():
    """
    A class to model and analyze multi-degree-of-freedom (MDOF) oscillators using OpenSees.

    This class provides functionality to create, analyze, and visualize structural models
    for dynamic and static analyses, including gravity analysis, modal analysis, static
    pushover analysis, cyclic pushover analysis, and nonlinear time-history analysis.

    Attributes
    ----------
    number_storeys : int
        The number of storeys in the building model.
    storey_heights : list
        List of storey heights in meters.
    floor_masses : list
        List of floor masses in tonnes.
    storey_disps : np.array
        Array of storey displacements (size = number of storeys, CapPoints).
    storey_forces : np.array
        Array of storey forces (size = number of storeys, CapPoints).
    degradation : bool
        Boolean to enable or disable hysteresis degradation.

    Methods
    -------
    __init__(number_storeys, storey_heights, floor_masses, storey_disps, storey_forces, degradation)
        Initializes the modeller object and validates input parameters.
    create_Pinching4_material(mat1Tag, mat2Tag, storey_forces, storey_disps, degradation)
        Creates a Pinching4 material model for the MDOF oscillator.
    compile_model()
        Compiles and sets up the MDOF oscillator model in OpenSees.
    plot_model(display_info=True)
        Plots the 3D visualization of the OpenSees model.
    do_gravity_analysis(nG=100, ansys_soe='UmfPack', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=500, algorithm_type='Newton', integrator='LoadControl', analysis='Static')
        Performs gravity analysis on the MDOF system.
    do_modal_analysis(num_modes=3, solver='-genBandArpack', doRayleigh=False, pFlag=False)
        Performs modal analysis to determine natural frequencies and mode shapes.
    do_spo_analysis(ref_disp, disp_scale_factor, push_dir, phi, pFlag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-5, init_iter=1000, algorithm_type='KrylovNewton', save_animation_path)
        Performs static pushover analysis (SPO) on the MDOF system.
    do_cpo_analysis(ref_disp, mu_levels, push_dir, dispIncr, pFlag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-5, init_iter=1000, algorithm_type='KrylovNewton', safe_animation_path)
        Performs cyclic pushover analysis (CPO) on the MDOF system.
    do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, pFlag=True, xi=0.05, ansys_soe='BandGeneral', constraints_handler='Plain', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, algorithm_type='Newton', save_animation_path, drift_thresholds)
        Performs nonlinear time-history analysis (NRHA) on the MDOF system.
    do_incremental_dynamic_analysis(fnames, dt_gm, t_max, dt_ansys,target_drift=0.05, initial_sf = 0.1, hunt_step =2.0,max_fill_gap=0.2, max_runs =15, capping_drift = 0.10, xi=0.05, pFlag=False))
        Performs nonlinear time-history analysis (NRHA) on the MDOF system.

    """
    def __init__(self, number_storeys, storey_heights, floor_masses, storey_disps, storey_forces, degradation):
        """
        Initializes the modeller object and validates the input parameters.

        Parameters
        ----------
        number_storeys : int
            The number of storeys in the building model.
        storey_heights : list
            List of storey heights in meters (e.g., [2.5, 3.0]).
        floor_masses : list
            List of floor masses in tonnes (e.g., [1000, 1200]).
        storey_disps : np.array
            Array of storey displacements (size = number of storeys, CapPoints).
        storey_forces : np.array
            Array of storey forces (size = number of storeys, CapPoints).
        degradation : bool
            Boolean to enable or disable hysteresis degradation.

        Raises
        ------
        ValueError
            If the number of entries in `storey_heights` or `floor_masses` does not match `number_storeys`.
        """

        ### Run tests on input parameters
        if len(storey_heights)!=number_storeys or len(floor_masses)!=number_storeys:
            raise ValueError('Number of entries exceed the number of storeys!')

        self.number_storeys = number_storeys
        self.storey_heights  = storey_heights
        self.floor_masses   = floor_masses
        self.storey_disps   = storey_disps
        self.storey_forces  = storey_forces
        self.degradation    = degradation


    def create_Pinching4_material(self, mat1Tag, mat2Tag, storey_forces, storey_disps, degradation):
        """
        Creates a Pinching4 material model for the multi-degree-of-freedom material object in stick model analysis.

        The Pinching4 material model is used to simulate hysteretic behavior in structures under dynamic loading,
        including degradation if enabled. The method assigns the material properties to the building storeys based
        on the given parameters.

        Parameters
        ----------
        mat1Tag : int
            Material tag for the first material in the Pinching4 model.
        mat2Tag : int
            Material tag for the second material in the Pinching4 model.
        storey_forces : np.array
            Array of storey forces at each storey in the model.
        storey_disps : np.array
            Array of storey displacements corresponding to the forces.
        degradation : bool
            Boolean flag to enable or disable hysteresis degradation in the Pinching4 material model.

        Returns
        -------
        None
            This method does not return any value but modifies the internal material definitions for the model.

        References:
        -----------
        1) Vamvatsikos D (2011) Software—earthquake, steel dynamics and probability, viewed January 2021.
        http://users.ntua.gr/divamva/software.html

        2) Martins, L., Silva, V., Crowley, H. et al. Vulnerability modellers toolkit, an open-source platform
        for vulnerability analysis. Bull Earthquake Eng 19, 5691–5709 (2021). https://doi.org/10.1007/s10518-021-01187-w

        3) Minjie Zhu, Frank McKenna, Michael H. Scott, OpenSeesPy: Python library for the OpenSees finite element framework,
        SoftwareX, Volume 7, 2018, Pages 6-11, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2017.10.009.
        (https://www.sciencedirect.com/science/article/pii/S2352711017300584)

        Notes
        -----
        The `mat1Tag` and `mat2Tag` represent different materials used in the Pinching4 hysteretic model,
        where the degradation flag controls the material's degradation behavior during the simulation.
        """

        force=np.zeros([5,1])
        disp =np.zeros([5,1])

        # Bilinear
        if len(storey_forces)==2:
              #bilinear curve
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]

              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]

              disp[2]=disp[1]+(disp[4]-disp[1])/3
              disp[3]=disp[1]+2*((disp[4]-disp[1])/3)

              force[2]=np.interp(disp[2],storey_disps,storey_forces)
              force[3]=np.interp(disp[3],storey_disps,storey_forces)

        # Trilinear
        elif len(storey_forces)==3:

              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]

              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]

              force[2]=storey_forces[1]
              disp[2] =storey_disps[1]

              disp[3]=np.mean([disp[2],disp[-1]])
              force[3]=np.interp(disp[3],storey_disps,storey_forces)

        # Quadrilinear
        elif len(storey_forces)==4:
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]

              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]

              force[2]=storey_forces[1]
              disp[2]=storey_disps[1]

              force[3]=storey_forces[2]
              disp[3]=storey_disps[2]

        if degradation==True:
            matargs=[force[1,0],disp[1,0],force[2,0],disp[2,0],force[3,0],disp[3,0],force[4,0],disp[4,0],
                                 -1*force[1,0],-1*disp[1,0],-1*force[2,0],-1*disp[2,0],-1*force[3,0],-1*disp[3,0],-1*force[4,0],-1*disp[4,0],
                                 0.5,0.25,0.05,
                                 0.5,0.25,0.05,
                                 0,0.1,0,0,0.2,
                                 0,0.1,0,0,0.2,
                                 0,0.4,0,0.4,0.9,
                                 10,'energy']
        else:
            matargs=[force[1,0],disp[1,0],force[2,0],disp[2,0],force[3,0],disp[3,0],force[4,0],disp[4,0],
                                 -1*force[1,0],-1*disp[1,0],-1*force[2,0],-1*disp[2,0],-1*force[3,0],-1*disp[3,0],-1*force[4,0],-1*disp[4,0],
                                 0.5,0.25,0.05,
                                 0.5,0.25,0.05,
                                 0,0,0,0,0,
                                 0,0,0,0,0,
                                 0,0,0,0,0,
                                 10,'energy']

        ops.uniaxialMaterial('Pinching4', mat1Tag,*matargs)
        ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -1*disp[-1,0], '-max', disp[-1,0])

    def compile_model(self):
        """
        Compiles and sets up the multi-degree-of-freedom (MDOF) oscillator model in OpenSees.

        This method constructs the model by defining nodes, assigning masses, imposing boundary conditions,
        and creating elements with associated material models for each storey in the building structure.
        It also defines rigid elastic materials for restrained degrees of freedom and nonlinear materials
        for unrestrained degrees of freedom. The method finally assembles the model for dynamic analysis.

        The process involves:
        1. Initializing the OpenSees model.
        2. Creating base and floor nodes.
        3. Assigning masses and degrees of freedom.
        4. Applying boundary conditions for the nodes.
        5. Creating zero-length elements for each storey with their respective material properties.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method uses OpenSees' `ops.node`, `ops.mass`, and `ops.element` to define nodes, masses,
          and zero-length elements for the MDOF oscillator.
        - Boundary conditions are applied with the base node being fully fixed, while the upper storeys
          have horizontal degrees of freedom released.
        - The material model used for each storey is a Pinching4 hysteretic model, created by the
          `create_Pinching4_material` method.
        """


        ### Set model builder
        ops.wipe() # wipe existing model
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        ### Define base node (tag = 0)
        ops.node(0, *[0.0, 0.0, 0.0])

        ### Define floor nodes (tag = 1+)
        current_height = 0.0

        # Use range based on the length of heights to ensure we never go out of bounds
        for i in range(len(self.storey_heights)):
            nodeTag = i + 1 # Nodes will be 1, 2, 3...

            current_height += self.storey_heights[i]
            current_mass = self.floor_masses[i]

            coords = [0.0, 0.0, current_height]
            # Assign mass to X and Y translations
            masses = [current_mass, current_mass, 1e-9, 1e-9, 1e-9, 1e-9]

            ops.node(nodeTag, *coords)
            ops.mass(nodeTag, *masses)

        # Update number_storeys to match the actual number of nodes created
        self.number_storeys = len(self.storey_heights)

        ### Get list of model nodes
        nodeList = ops.getNodeTags()
        ### Impose boundary conditions
        for i in nodeList:
            # fix the base node against all DOFs
            if i==0:
                ops.fix(i,1,1,1,1,1,1)
            # release the horizontal DOFs (1,2) and fix remaining
            else:
                ops.fix(i,0,0,1,1,1,1)

        ### Get number of zerolength elements required
        nodeList = ops.getNodeTags()

        for i in range(self.number_storeys):

            ### define the material tag associated with each storey
            mat1Tag = int(f'1{i}00') # hysteretic material tag
            mat2Tag = int(f'1{i}01') # min-max material tag

            ### get the backbone curve definition
            current_storey_disps = self.storey_disps[i,:].tolist() # deformation capacity (i.e., storey displacement in m)
            current_storey_forces = self.storey_forces[i,:].tolist() # strength capacity (i.e., storey base shear in kN)

            ### Create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e16)

            ### Create the nonlinear material for the unrestrained dofs
            self.create_Pinching4_material(mat1Tag, mat2Tag, current_storey_forces, current_storey_disps, self.degradation)

            ### Define element connectivity
            eleTag = int(f'200{i}')
            eleNodes = [i, i+1]

            ### Create the element
            ops.element('zeroLength', eleTag, eleNodes[0], eleNodes[1], '-mat', mat2Tag, mat2Tag, rigM, rigM, rigM, rigM, '-dir', 1, 2, 3, 4, 5, 6, '-doRayleigh', 1)


    def plot_model(self, display_info=True, export_path=None):
        """
        Plots a 2-D visualisation of the stick-and-mass model as two subplots.

        Left panel  — node elevation diagram with zigzag spring symbols
        (zero-length Pinching4 elements), individual storey height brackets,
        and a separate double-headed total-height dimension line.
        Right panel — force-deformation backbone curves for every storey,
        both axes starting from zero.
        Legend boxes are placed at the same vertical level below each panel.

        Parameters
        ----------
        display_info : bool, optional
            Annotate each node with its tag, elevation and mass.  Default True.
        export_path : str, optional
            If provided the figure is saved here instead of displayed.

        Returns
        -------
        None
        """
        import matplotlib.lines as mlines
        from matplotlib.lines import Line2D

        # ── Collect data from OpenSees domain ────────────────────────────────
        nodeList    = ops.getNodeTags()
        NodeCoordListZ, NodeMassList = [], []
        for tag in nodeList:
            NodeCoordListZ.append(ops.nodeCoord(tag, 3))
            NodeMassList.append(ops.nodeMass(tag, 1))

        n_st    = self.number_storeys
        total_h = max(NodeCoordListZ)

        # ── Colours ───────────────────────────────────────────────────────────
        COL_BASE   = '#B71C1C'
        COL_NODE   = '#1565C0'
        COL_ANN    = '#37474F'
        COL_GRID   = '#EBEBEB'
        COL_SPRING = '#546E7A'
        BG         = 'white'
        s_colors   = [plt.cm.tab10(i % 10) for i in range(n_st)]

        # ── Spring drawing helper ─────────────────────────────────────────────
        def _draw_spring(ax, x, z_bot, z_top, color, n_teeth=6, width=0.06):
            pad   = (z_top - z_bot) * 0.15
            n_pts = n_teeth * 2 + 1
            zs    = np.linspace(z_bot + pad, z_top - pad, n_pts)
            xs    = np.empty(n_pts)
            xs[0] = x; xs[-1] = x
            for k in range(1, n_pts - 1):
                xs[k] = x + width if k % 2 == 1 else x - width
            ax.plot([x, x], [z_bot, z_bot + pad], color=color, lw=1.5, zorder=3)
            ax.plot([x, x], [z_top - pad, z_top], color=color, lw=1.5, zorder=3)
            ax.plot(xs, zs, color=color, lw=1.5, zorder=3,
                    solid_capstyle='round', solid_joinstyle='round')

        # ── Custom legend handler that draws a zigzag spring icon ─────────────
        class _SpringHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                w, h   = handlebox.width, handlebox.height
                n      = 4; n_pts = n * 2 + 1
                xs_l   = np.linspace(x0 + 2, x0 + w - 2, n_pts)
                ys_l   = np.empty(n_pts)
                cy     = y0 + h / 2; amp = h * 0.38
                ys_l[0] = cy; ys_l[-1] = cy
                for k in range(1, n_pts - 1):
                    ys_l[k] = cy + amp if k % 2 == 1 else cy - amp
                line = mlines.Line2D(xs_l, ys_l, color=COL_SPRING, lw=1.5,
                                     solid_capstyle='round',
                                     solid_joinstyle='round')
                handlebox.add_artist(line)
                return line

        # ── Figure ────────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 9),
            gridspec_kw={'width_ratios': [1, 2.2]})
        fig.patch.set_facecolor(BG)
        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)

        # ═════════════════════════════════════════════════════════════════════
        # LEFT — node elevation diagram
        # ═════════════════════════════════════════════════════════════════════
        col_x = 0.0

        for i in range(n_st):
            _draw_spring(ax1, col_x,
                         NodeCoordListZ[i], NodeCoordListZ[i + 1],
                         COL_SPRING)

        for i, (z, m) in enumerate(zip(NodeCoordListZ, NodeMassList)):
            mk = 's' if i == 0 else 'o'
            co = COL_BASE if i == 0 else COL_NODE
            sz = 260 if i == 0 else 200
            ax1.scatter(col_x, z, s=sz, marker=mk, color=co,
                        edgecolors='white', linewidths=1.5, zorder=5)
            if display_info:
                ax1.plot([col_x + 0.02, col_x + 0.09], [z, z],
                         lw=0.8, color='#B0BEC5', zorder=1)
                ax1.text(col_x + 0.11, z,
                         f'Node {i}   z = {z:.2f} m   m = {m:.3f} t',
                         fontsize=9, color=COL_ANN,
                         va='center', ha='left', fontfamily='monospace')

        # ── Storey brackets (left of spring) ─────────────────────────────────
        bx_st = -0.12   # vertical bar of individual storey brackets
        for i in range(n_st):
            z_bot = NodeCoordListZ[i]; z_top = NodeCoordListZ[i + 1]
            z_mid = (z_bot + z_top) / 2.0; sh = z_top - z_bot
            ax1.plot([bx_st - 0.03, bx_st], [z_bot, z_bot], lw=0.7, color='#90A4AE')
            ax1.plot([bx_st - 0.03, bx_st], [z_top, z_top], lw=0.7, color='#90A4AE')
            ax1.plot([bx_st - 0.03, bx_st - 0.03], [z_bot, z_top],
                     lw=0.7, color='#90A4AE')
            ax1.text(bx_st - 0.05, z_mid, f'{sh:.2f} m',
                     fontsize=7.5, color='#90A4AE', ha='right', va='center')

        # ── Element ID labels (right of spring) ───────────────────────────────
        ele_list = ops.getEleTags()
        for i in range(n_st):
            z_mid = (NodeCoordListZ[i] + NodeCoordListZ[i + 1]) / 2.0
            ele_id = ele_list[i] if i < len(ele_list) else i
            ax1.text(col_x + 0.09, z_mid,
                     f'Ele. {ele_id}',
                     fontsize=8, color=COL_SPRING,
                     ha='left', va='center', style='italic')

        # axis styling
        for sp in ['top', 'right', 'bottom']:
            ax1.spines[sp].set_visible(False)
        ax1.spines['left'].set_color('#90A4AE')
        ax1.spines['left'].set_linewidth(0.8)
        ax1.set_xticks([])
        ax1.set_xlim(-0.40, 1.45)
        ax1.set_ylim(0, total_h + 0.5)
        ax1.set_ylabel('Height,  z  [m]', fontsize=12, fontweight='bold',
                       color=COL_ANN, labelpad=8)
        ax1.tick_params(axis='y', labelsize=10, colors=COL_ANN)
        ax1.set_title('Node Positions', fontsize=13, fontweight='bold',
                      color='#1A237E', pad=12)

        # ═════════════════════════════════════════════════════════════════════
        # RIGHT — storey force-deformation backbones
        # ═════════════════════════════════════════════════════════════════════
        ax2.grid(True, color=COL_GRID, linewidth=0.7, zorder=0)
        ax2.set_axisbelow(True)

        for i in range(n_st):
            d  = np.concatenate(([0.0], self.storey_disps[i, :]))
            f  = np.concatenate(([0.0], self.storey_forces[i, :]))
            sc = s_colors[i]
            ax2.plot(d, f, color=sc, lw=2.2, zorder=3,
                     label=f'Storey {i + 1}', solid_capstyle='round')
            ax2.scatter(d[1:], f[1:], color=sc, s=40, zorder=4,
                        edgecolors='white', linewidths=0.8)

        for sp in ['top', 'right']:
            ax2.spines[sp].set_visible(False)
        ax2.spines['left'].set_color(COL_ANN)
        ax2.spines['left'].set_linewidth(1.0)
        ax2.spines['bottom'].set_color(COL_ANN)
        ax2.spines['bottom'].set_linewidth(1.0)

        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel('Storey Drift Capacity,  \u03b4\u1d62  [m]',
                       fontsize=12, fontweight='bold', color=COL_ANN, labelpad=10)
        ax2.set_ylabel('Storey Shear Force,  V\u1d62  [kN]',
                       fontsize=12, fontweight='bold', color=COL_ANN, labelpad=10)
        ax2.tick_params(labelsize=10, colors=COL_ANN)
        ax2.set_title('Storey Force\u2013Deformation Relationships',
                      fontsize=13, fontweight='bold', color='#1A237E', pad=12)

        # ── Legends — same vertical level below each panel ────────────────────
        spring_handle = Line2D([], [], color=COL_SPRING, lw=1.5,
                               label='Zero-length spring')
        handles1 = [
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=COL_BASE, markersize=9,
                   label='Fixed base node'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COL_NODE, markersize=9,
                   label='Floor node'),
            spring_handle,
        ]
        handles2 = [
            Line2D([0], [0], color=s_colors[i], lw=2.2, label=f'Storey {i + 1}')
            for i in range(n_st)
        ]

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14)
        fig.canvas.draw()

        p1    = ax1.get_position()
        p2    = ax2.get_position()
        leg_y = 0.01

        fig.legend(handles=handles1,
                   handler_map={spring_handle: _SpringHandler()},
                   fontsize=8.5, ncol=3, loc='lower center',
                   bbox_to_anchor=(p1.x0 + p1.width / 2, leg_y),
                   bbox_transform=fig.transFigure,
                   framealpha=0.95, edgecolor='#CFD8DC',
                   borderpad=0.7, handletextpad=0.4)
        fig.legend(handles=handles2,
                   fontsize=9, ncol=min(n_st, 5), loc='lower center',
                   bbox_to_anchor=(p2.x0 + p2.width / 2, leg_y),
                   bbox_transform=fig.transFigure,
                   framealpha=0.95, edgecolor='#CFD8DC',
                   borderpad=0.7, handletextpad=0.5)

        # ── Super-title ───────────────────────────────────────────────────────
        label = 'SDOF Oscillator' if n_st == 1 else f'{n_st}-Storey MDOF'
        fig.suptitle(f'OpenSees {label}  \u2014  Stick-and-Mass Model',
                     fontsize=14, fontweight='bold', color='#1A237E', y=1.01)

        if export_path:
            plt.savefig(export_path, dpi=150,
                        bbox_inches='tight', facecolor=BG)
        else:
            plt.show()
        plt.close()

##########################################################################
#                             ANALYSIS MODULES                           #
##########################################################################
    def do_gravity_analysis(self, nG=100,
                            ansys_soe='UmfPack',
                            constraints_handler='Transformation',
                            numberer='RCM',
                            test_type='NormDispIncr',
                            init_tol = 1.0e-6,
                            init_iter = 500,
                            algorithm_type='Newton' ,
                            integrator='LoadControl',
                            analysis='Static'):
        """
        Perform a gravity analysis on a multi-degree-of-freedom (MDOF) system in OpenSees.

        This method sets up and runs a gravity analysis using specified parameters for various analysis objects
        in OpenSees. The gravity analysis solves for the static equilibrium of the system under self-weight loads
        (e.g., gravity loads).

        Parameters
        ----------
        nG: int, optional
            Number of gravity analysis steps to perform. Default is 100.

        ansys_soe: string, optional
            The system of equations type to be used in the analysis. This defines how the system of equations
            will be solved. Default is 'UmfPack' (sparse direct solver).

        constraints_handler: string, optional
            The constraints handler determines how the constraint equations are enforced in the analysis.
            It controls the enforcement of specified values for degrees-of-freedom (DOFs) or relationships
            between them. Default is 'Transformation' (transforming the constrained DOFs into active ones).

        numberer: string, optional
            The degree-of-freedom numberer defines how DOFs are numbered. This is important for system
            efficiency in solving. Default is 'RCM' (Reverse Cuthill-McKee, a reordering algorithm).

        test_type: string, optional
            Defines the test type used to check the convergence of the solution. It is used in constructing
            the LinearSOE and LinearSolver objects. Default is 'NormDispIncr' (norm of displacement increment).

        init_tol: float, optional
            The tolerance criterion for checking convergence. A smaller value means stricter convergence.
            Default is 1.0e-6.

        init_iter: int, optional
            The maximum number of iterations to check for convergence. Default is 500.

        algorithm_type: string, optional
            Defines the solution algorithm used in the analysis. Common options are 'Newton' (Newton-Raphson)
            for solving the system of equations. Default is 'Newton'.

        integrator: string, optional
            Defines the integrator for the analysis. The integrator dictates how the analysis steps are taken
            in time or load. Default is 'LoadControl' (control load increments).

        analysis: string, optional
            Defines the type of analysis to be performed. 'Static' is typically used for gravity analysis,
            but other options (e.g., 'Transient') can be used depending on the type of analysis. Default is 'Static'.

        Returns
        -------
        None.

        Notes
        -----
        - This method sets up the analysis using OpenSees by defining the system of equations, constraints
          handler, numberer, convergence test, solution algorithm, integrator, and analysis type.
        - The gravity analysis solves for the static equilibrium under self-weight or gravity loads and is
          typically used to determine the initial equilibrium state of a structure before dynamic loading.
        - The analysis can be modified by changing the parameters to adjust solver settings, tolerance,
          and other relevant options.
        - After the analysis is completed, the analysis objects are wiped to ensure a clean state for further analyses.
        """

        ### Define the analysis objects and run gravity analysis
        ops.system(ansys_soe) # creates the system of equations, a sparse solver with partial pivoting
        ops.constraints(constraints_handler) # creates the constraint handler, the transformation method
        ops.numberer(numberer) # creates the DOF numberer, the reverse Cuthill-McKee algorithm
        ops.test(test_type, init_tol, init_iter, 3) # creates the convergence test
        ops.algorithm(algorithm_type) # creates the solution algorithm, a Newton-Raphson algorithm
        ops.integrator(integrator, (1/nG)) # creates the integration scheme
        ops.analysis(analysis) # creates the analysis object
        ops.analyze(nG) # perform the gravity load analysis
        ops.loadConst('-time', 0.0)

        ### Wipe the analysis objects
        ops.wipeAnalysis()

    def do_modal_analysis(self,
                          num_modes=3,
                          solver = '-genBandArpack',
                          doRayleigh=False,
                          pFlag=False,
                          plot_modes=True,
                          export_path = None):
        """
        Perform modal analysis on a multi-degree-of-freedom (MDOF) system to determine its natural frequencies
        and mode shapes.

        This method calculates the natural frequencies and corresponding mode shapes of the system. The natural
        frequencies are determined by solving the eigenvalue problem, and the mode shapes are normalized
        for the system's degrees of freedom. The results can be used to assess the dynamic characteristics
        of the system.

        Parameters
        ----------
        num_modes: int, optional
            The number of modes to consider in the analysis. Default is 3. This parameter determines how many
            modes will be computed in the modal analysis.

        solver: string, optional
            The type of solver to use for the eigenvalue problem. Default is '-genBandArpack', which uses a
            generalized banded Arnoldi method for large sparse eigenvalue problems.

        doRayleigh: bool, optional
            Flag to enable or disable Rayleigh damping in the modal analysis. This parameter is not used directly
            in this method but can be set in the OpenSees model. Default is False.

        pFlag: bool, optional
            Flag to control whether to print the modal analysis report. If True, the fundamental period and
            mode shape will be printed to the console. Default is False.

        plot_modes: bool, optional
            Flag to control whether to plot the modes. If True, the mode shapes are plotted against the
            undeformed shape. Default is True

        export_path: str, optional
            If a string path is provided (e.g., 'modal_results.png'), the plot will be saved to this location.
            If None, the plot will be only displayed and not saved. Default is None.

        Returns
        -------
        T: array
            The periods of vibration for the system, calculated as 2π/ω, where ω are the natural frequencies
            obtained from the eigenvalue problem.

        mode_shape: list
            A list of the normalized mode shapes for the system, with each element representing the displacement
            in the x-direction for the corresponding mode. The mode shapes are normalized by the last node's
            displacement.
        """

        # Get frequency and period
        self.omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0*np.pi/self.omega

        # Extract mode shape vectors
        node_list = ops.getNodeTags()

        # Fallback: determine the largest node tag index for eigenvector extraction
        if not hasattr(self, 'number_storyes'):
            self.number_storeys = len(node_list)
        mode_shape_vectors = []
        for mode_num in range(1, num_modes+1):
            # Extract X, Y, Z displacements for all nodes in the current mode
            ux_all = np.array([ops.nodeEigenvector(tag, mode_num, 1) for tag in node_list])
            uy_all = np.array([ops.nodeEigenvector(tag, mode_num, 2) for tag in node_list])
            uz_all = np.array([ops.nodeEigenvector(tag, mode_num, 3) for tag in node_list])

            # Combine into a single (N_nodes x 3) vector for plotting
            mode_vector = np.column_stack((ux_all, uy_all, uz_all))

            # Normalization
            max_disp = np.max(np.abs(mode_vector))
            if max_disp!=0:
                mode_vector/=max_disp
            mode_shape_vectors.append(mode_vector)

        # Optional printing
        if pFlag:
            ops.modalProperties('-print')
            print(r'Fundamental Period: T = {.3f} s'.format(T[0]))

        # Optional plotting
        if plot_modes:
            # Initialise the plotter class
            pl=plotter()
            pl.plot_modes(node_list, mode_shape_vectors, T, export_path =export_path)

        # Internal cleanup of analysis objects
        ops.wipeAnalysis()

        return T, mode_shape_vectors

    def do_spo_analysis(self,
                        ref_disp,
                        disp_scale_factor,
                        push_dir,
                        phi,
                        pFlag=True,
                        num_steps=200,
                        ansys_soe='BandGeneral',
                        constraints_handler='Transformation',
                        numberer='RCM',
                        test_type='EnergyIncr',
                        init_tol=1.0e-5,
                        init_iter=1000,
                        algorithm_type='KrylovNewton',
                        save_animation_path=None):
        """
        Perform static pushover analysis (SPO) on a multi-degree-of-freedom (MDOF) system.

        This method simulates a static pushover analysis where a lateral load pattern is incrementally applied
        to the structure. The displacement at the control node is increased step by step, and the corresponding
        base shear, floor displacements, and forces in non-linear elements are recorded. The analysis helps in
        evaluating the structural response to lateral loads, such as earthquake forces.

        Parameters
        ----------
        ref_disp: float
            The reference displacement at which the analysis starts, corresponding to the yield or other
            significant displacement (e.g., 1mm).

        disp_scale_factor: float
            The scale factor applied to the reference displacement to determine the final displacement.
            The analysis will be run to this scaled displacement.

        push_dir: int
            The direction in which the pushover load is applied:
                1 = X direction
                2 = Y direction
                3 = Z direction

        phi: list of floats
            The lateral load pattern shape. This is typically a mode shape or a predefined load distribution.
            For example, it can be the first-mode shape from the calibrateModel function.

        pFlag: bool, optional
            Flag to print (or not) the pushover analysis steps. If True, detailed feedback on each step will be printed. Default is True.

        num_steps: int, optional
            The number of steps to increment the pushover load. Default is 200.

        ansys_soe: string, optional
            The type of system of equations solver to use. Default is 'BandGeneral'.

        constraints_handler: string, optional
            The constraints handler object to determine how constraint equations are enforced. Default is 'Transformation'.

        numberer: string, optional
            The degree-of-freedom (DOF) numberer object to determine the mapping between equation numbers and degrees-of-freedom. Default is 'RCM'.

        test_type: string, optional
            The type of test to use for the linear system of equations. Default is 'EnergyIncr'.

        init_tol: float, optional
            The tolerance criterion to check for convergence. Default is 1.0e-5.

        init_iter: int, optional
            The maximum number of iterations to perform when checking for convergence. Default is 1000.

        algorithm_type: string, optional
            The type of algorithm used to solve the system. Default is 'KrylovNewton'.

        save_animation_path: string, optional,
            If provided, saves the figure to this path (e.g., 'spo.gif')

        Returns
        -------
        spo_dict: dict
            A dictionary containing the SPO results with the following keys:
            'spo_disps': array - Displacements at each floor level (TimeSteps x Floors).
            'spo_rxn': array - Base shear recorded at the base (TimeSteps).
            'spo_disps_spring': array - Displacements in the storey zero-length elements (TimeSteps x Springs).
            'spo_forces_spring': array - Shear forces in the storey zero-length elements (TimeSteps x Springs).
            'spo_idr': array - Interstorey drift ratio history for each storey (TimeSteps x Storeys).
            'spo_midr': array - Maximum interstorey drift ratio history (max IDR across all stories at each step, TimeSteps).
        """

        # --- Setup OpenSees Model for Analysis ---
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]] # Base node for reaction calculation

        # Apply the lateral load pattern
        for i in np.arange(len(pattern_nodes)):
            load_val = 1.0 if len(pattern_nodes)==1 else phi[i]*self.floor_masses[i]
            if push_dir == 1:
                ops.load(pattern_nodes[i], load_val, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, load_val, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, load_val, 0.0, 0.0, 0.0)

        # Set analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Set integrator
        target_disp = float(ref_disp)*float(disp_scale_factor)
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, push_dir, delta_disp)
        ops.analysis('Static')

        elementList = ops.getEleTags()

        if pFlag is True:
            print(f"\n------ Static Pushover Analysis of Node # {control_node} to {target_disp} ---------")

        ok = 0
        step = 1
        loadf = 1.0

        # Initialize result arrays with current state (usually 0.0)
        spo_rxn = np.array([0.])
        spo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)]) # Used for animation and Pushover Curve
        spo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])
        spo_disps_spring = np.array([[ops.eleResponse(ele, 'deformation')[0] for ele in elementList]])
        spo_forces_spring = np.array([[ops.eleResponse(ele, 'force')[0] for ele in elementList]])

        # Main Analysis Loop
        while step <= num_steps and ok == 0 and loadf > 0:

            ok = ops.analyze(1)

            # Adaptive Convergence Scheme
            if ok != 0:
                if pFlag: print('FAILED: Trying relaxing convergence...')
                ops.test(test_type, init_tol*0.01, init_iter)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                if pFlag: print('FAILED: Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                if pFlag: print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                if pFlag: print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                if pFlag: print('FAILED: Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                if ok != 0: # Final check before breaking
                    break

            loadf = ops.getTime()

            if pFlag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print(f'Currently pushed node {control_node} to {curr_disp:.4f} with load factor {loadf:.4f}')

            step += 1

            # Record Results
            spo_top_disp = np.append(spo_top_disp, ops.nodeResponse(control_node, push_dir, 1))

            current_disps = np.array([ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes])
            spo_disps = np.append(spo_disps, np.array([current_disps]), axis=0)

            spo_disps_spring = np.append(spo_disps_spring, np.array([
                [ops.eleResponse(ele, 'deformation')[0] for ele in elementList]
            ]), axis=0)

            spo_forces_spring = np.append(spo_forces_spring, np.array([
                [ops.eleResponse(ele, 'force')[0] for ele in elementList]
            ]), axis=0)

            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            spo_rxn = np.append(spo_rxn, -temp)


        # Final Cleanup and Output
        if ok != 0:
            print('------ ANALYSIS FAILED --------')
        elif ok == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        if loadf < 0:
            print('Stopped because of load factor below zero')

        ops.wipeAnalysis()

        # Calculate Interstorey Drift Ratio (IDR) and Max IDR (MIDR)
        # Use a COPY of the original displacement history for IDR calculation
        idr_disps = spo_disps.copy()

        if not hasattr(self, 'storey_heights'):
            raise AttributeError("Cannot calculate IDR: 'storey_heights' property is required but not defined in the class.")

        # Prepend ground floor (zero displacement)
        ground_disps = np.zeros((idr_disps.shape[0], 1))
        full_idr_disps = np.hstack([ground_disps, idr_disps])

        # Compute interstorey displacements (ISD)
        spo_isd = np.diff(full_idr_disps, axis=1)

        # Convert storey_heights to a numpy array for division
        storey_heights = np.array(self.storey_heights)

        # Normalize by corresponding storey heights to get IDR (x100 requested)
        spo_idr = (spo_isd / storey_heights) * 100

        # Take the maximum interstorey drift ratio per step
        spo_midr = np.max(np.abs(spo_idr), axis=1)

        # Handle Animation (Call updated function with spo_midr)
        if save_animation_path:
            pl = plotter()
            pl.animate_spo(spo_top_disp, spo_rxn, spo_disps, spo_midr, nodeList, elementList, push_dir, save_animation_path)

        # Pack and Return results into a dictionary
        spo_dict = {'spo_disps': spo_disps,
                    'spo_rxn': spo_rxn,
                    'spo_disps_spring': spo_disps_spring,
                    'spo_forces_spring': spo_forces_spring,
                    'spo_idr': spo_idr,
                    'spo_midr': spo_midr}

        return spo_dict

    def do_cpo_analysis(self,
                        ref_disp,
                        mu_levels,
                        push_dir,
                        dispIncr,
                        phi,
                        pFlag=True,
                        ansys_soe='BandGeneral',
                        constraints_handler='Transformation',
                        numberer='RCM',
                        test_type='NormDispIncr',
                        init_tol=1.0e-5,
                        init_iter=1000,
                        algorithm_type='KrylovNewton',
                        save_animation_path=None):
        """
        Perform cyclic pushover (CPO) analysis on a Multi-Degree-of-Freedom (MDOF) system.

        Parameters
        ----------
        ref_disp: float
            Reference displacement (e.g., yield displacement) for scaling the cycles.
        mu_levels: list
            Target ductility factors (mu) for each cycle level.
        push_dir: int
            Direction of the pushover analysis (1=X, 2=Y, 3=Z).
        dispIncr: int
            The number of displacement increments for each loading cycle target.
        phi: list of floats
            The lateral load pattern shape vector (scaled by mass).
        pFlag: bool, optional, default=True
            If True, prints feedback during the analysis steps.
        save_animation_path: str, optional, default=None
            If provided, the path to save the animation (e.g., 'cpo.gif' or 'cpo.mp4').
        ansys_soe: string, optional, default='BandGeneral'
            System of equations solver.
        constraints_handler: string, optional, default='Transformation'
            Constraint handler method.
        numberer: string, optional, default='RCM'
            The numberer method.
        test_type: string, optional, default='NormDispIncr'
            Convergence test type.
        init_tol: float, optional, default=1e-5
            The initial tolerance for convergence.
        init_iter: int, optional, default=1000
            The maximum number of iterations for the solver.
        algorithm_type: string, optional, default='KrylovNewton'
            The type of algorithm used to solve the system of equations.
        save_animation_path: string, optional,
            If provided, saves the figure to this path (e.g., 'cpo.gif')

        Returns
        -------
        cpo_dict: dict
            A dictionary containing all the analysis results (displacements, base_shear, etc.).
        """

        if mu_levels is None:
            mu_levels = [1, 2, 4, 6, 8, 10]

        # Always start from a clean post-gravity state.
        self.compile_model()
        self.do_gravity_analysis()

        # Apply the load pattern
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain",1,1)

        # Get all tags needed for analysis and animation
        nodeList = ops.getNodeTags()
        elementList = ops.getEleTags()

        # Ensure model has nodes
        if not nodeList:
            print("ERROR: No nodes found in the OpenSees model.")
            return None

        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:] # All nodes above ground
        rxn_nodes = [nodeList[0]] # Ground node

        # Quality control
        assert len(phi) == len(pattern_nodes), "phi length must match pattern_nodes"
        assert len(self.floor_masses) == len(pattern_nodes), "floor_masses length mismatch"

        # Apply lateral load pattern scaled by mass
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                ops.load(pattern_nodes[i], phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0)

        # Set up the analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Create the list of target displacements (e.g., +1mu, -1mu, +2mu, -2mu, ...)
        cycleDispList = []
        for mu in mu_levels:
            cycleDispList.append(ref_disp * mu)   # push positive
            cycleDispList.append(-ref_disp * mu)  # pull negative
        dispNoMax = len(cycleDispList)

        if pFlag:
            print(f"\n------ Cyclic Pushover with ductility levels: {mu_levels} ------")

        # Recording data arrays
        cpo_rxn = [0.0]
        cpo_top_disp = [ops.nodeDisp(control_node, push_dir)]
        cpo_disps = [[ops.nodeDisp(node, push_dir) for node in pattern_nodes]]
        energy_steps = [0.0]

        for d in range(dispNoMax):
            numIncr = dispIncr
            current_disp = ops.nodeDisp(control_node, push_dir)
            target_disp = cycleDispList[d]
            dU = (target_disp - current_disp) / numIncr

            # Use DisplacementControl integrator
            ops.integrator('DisplacementControl', control_node, push_dir, dU)
            ops.analysis('Static')

            # Loop over displacement increments
            for l in range(numIncr):
                ok = ops.analyze(1)

                # Convergence Failure Handling (Extended Recovery)
                if ok != 0:
                    print(f'FAILED at cycle {d+1}/{dispNoMax}, increment {l}/{numIncr}: Starting complex recovery attempts...')

                # Try relaxing convergence tolerance
                if ok != 0:
                    print('FAILED: Trying relaxing convergence...')
                    ops.test(test_type, init_tol*0.01, init_iter)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)

                # Try relaxing convergence tolerance with more iterations
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iterations...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)

                # Try relaxing tolerance, more iterations, and Newton with 'initialThenCurrent'
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initialThenCurrent')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type) # Restore original algorithm

                # Try relaxing tolerance, more iterations, and Newton with 'initial'
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initial')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type) # Restore original algorithm

                # Attempt a Hail Mary (FixedNumIter)
                if ok != 0:
                    print('FAILED: Attempting a Hail Mary...')
                    ops.test('FixedNumIter', init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter) # Restore original test type

                # Final failure check
                if ok != 0:
                    print('Analysis Failed')
                    break

                # Data Recording (only if successful)
                if ok == 0:
                    curr_disp = ops.nodeDisp(control_node, push_dir)
                    cpo_top_disp.append(curr_disp)

                    current_floor_disps = [ops.nodeDisp(node, push_dir) for node in pattern_nodes]
                    cpo_disps.append(current_floor_disps)

                    ops.reactions()
                    temp = sum(ops.nodeReaction(n, push_dir) for n in rxn_nodes)
                    curr_rxn = -temp
                    cpo_rxn.append(curr_rxn)

                    if len(cpo_top_disp) >= 2:
                        dU_step = cpo_top_disp[-1] - cpo_top_disp[-2]
                        avg_F = 0.5 * (cpo_rxn[-1] + cpo_rxn[-2])
                        dE = abs(avg_F * dU_step)
                        energy_steps.append(energy_steps[-1] + dE)
                    else:
                        energy_steps.append(energy_steps[-1])

            if pFlag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print(f"Cycle target {d+1}/{dispNoMax}: Pushed node {control_node} to {curr_disp:.4f}")

        # Convert lists to numpy arrays
        cpo_rxn = np.array(cpo_rxn)
        cpo_top_disp = np.array(cpo_top_disp)
        cpo_disps = np.array(cpo_disps)
        pseudo_steps = np.arange(len(energy_steps))
        cpo_energy = np.column_stack((pseudo_steps, energy_steps))

        # Calculate Interstorey Drifts
        base_disps = np.zeros((cpo_disps.shape[0], 1))
        padded_disps = np.hstack((base_disps, cpo_disps))
        cpo_drifts = np.diff(padded_disps, axis=1)
        max_interstorey_drift = np.max(np.abs(cpo_drifts))

        ops.wipeAnalysis()

        # Final output dictionary (cpo_dict)
        cpo_dict = {'cpo_disps': cpo_disps,
                    'cpo_rxn': cpo_rxn,
                    'cpo_top_disp': cpo_top_disp,
                    'cpo_energy': cpo_energy,
                    'cpo_idr': cpo_drifts,
                    'cpo_midr': max_interstorey_drift}

        # Animation Call
        if save_animation_path:

            pl = plotter()
            pl.animate_cpo(cpo_dict, nodeList, elementList, push_dir, save_animation_path)

        return cpo_dict

    def do_nrha_analysis(self, fnames, dt_gm, sf, t_max, dt_ansys,
                         pFlag=True, xi=0.05, ansys_soe='BandGeneral',
                         constraints_handler='Plain', numberer='RCM',
                         test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50,
                         algorithm_type='Newton', save_animation_path=None, drift_thresholds=None):

        """
        Perform nonlinear time-history analysis on a Multi-Degree-of-Freedom (MDOF) system.

        This method performs a nonlinear time-history analysis where ground motion records are applied to the
        system to simulate real-world seismic conditions. The analysis uses step-by-step integration methods
        to solve the system's response under dynamic loading.

        Parameters
        ----------
        fnames: list
            List of file paths to the ground motion records for each direction (X, Y, Z). At least one ground motion
            record in the X direction is required.

        dt_gm: float
            Time-step of the ground motion records, which is typically the time between each data point in the records.

        sf: float
            Scale factor to apply to the ground motion records. Typically equal to the gravitational acceleration (9.81 m/s²).

        t_max: float
            The maximum time duration for the analysis. It is typically the total time span of the ground motion record.

        dt_ansys: float
            The time-step at which the analysis will be conducted. Typically smaller than the ground motion time-step to
            ensure accurate results.

        pFlag: bool, optional, default=True
            Flag to print progress updates during the analysis. If True, the function prints information about the analysis
            steps and progress.

        xi: float, optional, default=0.05
            The inherent damping ratio used in the analysis. The default is 5% damping (0.05).

        ansys_soe: string, optional, default='BandGeneral'
            Type of the system of equations solver to be used in the analysis (e.g., 'BandGeneral', 'FullGeneral', etc.).

        constraints_handler: string, optional, default='Plain'
            The method used to handle constraints in the analysis. This handles how boundary conditions or prescribed
            displacements are enforced.

        numberer: string, optional, default='RCM'
            The numberer object determines the equation numbering used in the analysis. Default is 'RCM' (Reverse Cuthill-McKee).

        test_type: string, optional, default='NormDispIncr'
            Type of convergence test used during the analysis to check whether the solution has converged. Default is 'NormDispIncr'.

        init_tol: float, optional, default=1.0e-6
            Initial tolerance for the convergence test, used to check if the solution is converging to a sufficiently accurate result.

        init_iter: int, optional, default=50
            Maximum number of iterations allowed during each time step for the analysis to converge.

        algorithm_type: string, optional, default='Newton'
            Type of algorithm used to solve the system of equations. Default is 'Newton', which uses the Newton-Raphson method.

        save_animation_path: string, optional,
            If provided, saves the figure to this path (e.g., 'modes.png') instead of displaying it

        drift_thresholds: list, optional,
            If provided, provides thresholds for animation to read and change color of stick model based on damage state exceedance

        Returns
        -------
        control_nodes: list
            List of the floor node tags in the MDOF system.

        conv_index: int
            Convergence status index: -1 indicates failure, 0 indicates success (converged).

        peak_drift: numpy.ndarray
            Array of peak storey drift values for each storey in the X and Y directions (radians).

        peak_accel: numpy.ndarray
            Array of peak floor acceleration values for each floor in the X and Y directions (g).

        max_peak_drift: float
            The maximum peak storey drift value (radians) across all floors.

        max_peak_drift_dir: string
            Direction of the maximum peak storey drift ('X' or 'Y').

        max_peak_drift_loc: int
            Location (storey) of the maximum peak storey drift.

        max_peak_accel: float
            The maximum peak floor acceleration value (g) across all floors.

        max_peak_accel_dir: string
            Direction of the maximum peak floor acceleration ('X' or 'Y').

        max_peak_accel_loc: int
            Location (floor) of the maximum peak floor acceleration.

        peak_disp: numpy.ndarray
            Array of peak displacement values (in meters) for each floor.
        """

        # Always rebuild from a clean state and re-run modal analysis to
        # populate self.omega, which is required for Rayleigh damping below.
        self.compile_model()
        self.do_gravity_analysis()
        self.do_modal_analysis(num_modes=min(self.number_storeys, 3), plot_modes=False)

        # Define control nodes
        control_nodes = ops.getNodeTags()
        n_nodes = len(control_nodes)

        # Define the timeseries and patterns first (same as original; no recorders)
        if len(fnames) > 0:
            nrha_tsTagX = 1
            nrha_pTagX = 1
            ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf)
            ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
        if len(fnames) > 1:
            nrha_tsTagY = 2
            nrha_pTagY = 2
            ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf)
            ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
        if len(fnames) > 2:
            nrha_tsTagZ = 3
            nrha_pTagZ = 3
            ops.timeSeries('Path', nrha_tsTagZ, '-dt', dt_gm, '-filePath', fnames[2], '-factor', sf)
            ops.pattern('UniformExcitation', nrha_pTagZ, 3, '-accel', nrha_tsTagZ)

        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        # Set up analysis parameters
        conv_index = 0   # -1 failure, 0 success
        control_time = 0.0
        ok = 0

        # Parse the data about the building (storey heights)
        if n_nodes < 2:
            top_nodes = []
            bottom_nodes = []
        else:
            top_nodes = control_nodes[1:]
            bottom_nodes = control_nodes[:-1]

        h = []
        for i in range(len(top_nodes)):
            topZ = ops.nodeCoord(top_nodes[i], 3)
            bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
            dist = topZ - bottomZ
            if dist == 0:
                print("WARNING: Zero length found in drift check, using very large distance 1e9 instead")
                h.append(1e9)
            else:
                h.append(dist)
        h = np.array(h) if len(h) > 0 else np.array([])

        # Create some arrays to record to (use dt_ansys for sizing)
        n_steps = int(np.ceil(t_max / dt_ansys)) + 1
        node_disps = np.zeros((n_steps, n_nodes))       # X displacements
        node_accels = np.zeros((n_steps, n_nodes))     # accelerations (will keep in g)
        peak_disp = np.zeros((n_nodes, 2))
        peak_drift = np.zeros((len(top_nodes), 2))
        peak_accel = np.zeros((len(top_nodes) + 1, 2))

        # Set damping (same logic)
        if self.number_storeys == 1:
            alphaM = 2 * self.omega[0] * xi
            ops.rayleigh(alphaM, 0, 0, 0)
        else:
            alphaM = 2 * self.omega[0] * self.omega[2] * xi / (self.omega[0] + self.omega[2])
            alphaK = 2 * xi / (self.omega[0] + self.omega[2])
            ops.rayleigh(alphaM, 0, alphaK, 0)

        # Progress print throttling (print every print_every steps)
        # Aim for ~50 prints across run
        print_every = max(1, int(np.ceil(n_steps / 50.0)))

        # Main time-stepping loop (preserve all adaptive attempts exactly)
        step = 0
        while conv_index == 0 and control_time <= t_max and ok == 0:
            ok = ops.analyze(1, dt_ansys)
            control_time = ops.getTime()

            # Throttled progress output
            if pFlag and (step % print_every == 0 or control_time >= t_max):
                print(f'Completed {control_time:.3f} of {t_max:.3f} seconds')

            # If analysis fails, run the exact adaptive recovery sequence (kept identical)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in half...')
                ok = ops.analyze(1, 0.5 * dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in quarter...')
                ok = ops.analyze(1, 0.25 * dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter * 10)
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)

            # If still failed -> exit
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Exiting analysis...')
                conv_index = -1
                break

            # Efficiently query node responses once per node and store locally
            # We'll query X and Y displacements and acceleration response (dof=1 accel) once per node,
            # store in the arrays, then compute drifts vectorized (avoid repeated calls).
            for i, node in enumerate(control_nodes):
                # nodeDisp(node, dof) is used; keep calls minimal
                disp_x = ops.nodeDisp(node, 1)
                # For peak_disp Y we still need nodeDisp(...,2) — fetch here too
                disp_y = ops.nodeDisp(node, 2)
                accel_resp = ops.nodeResponse(node, 1, 3)  # acceleration (relative) in units of length/time^2

                node_disps[step, i] = disp_x
                node_accels[step, i] = accel_resp / 9.81  # convert to g for consistency

                # update peak displacements (absolute)
                abs_dx = abs(disp_x)
                abs_dy = abs(disp_y)
                if abs_dx > peak_disp[i, 0]:
                    peak_disp[i, 0] = abs_dx
                if abs_dy > peak_disp[i, 1]:
                    peak_disp[i, 1] = abs_dy

            # Compute storey drifts for this time step via in-memory values (vectorized)
            if len(top_nodes) > 0:
                # top indices in control_nodes are 1..N-1, bottom 0..N-2
                disp_top = node_disps[step, 1:]   # length = n_nodes-1
                disp_bottom = node_disps[step, :-1]
                drift_vals = np.abs(disp_top - disp_bottom)
                # normalize by heights h
                # avoid division by zero because h already set to large value if zero
                normalized_drifts = drift_vals / h
                # update peak_drift for X (we used X displacements)
                for idx in range(len(normalized_drifts)):
                    if normalized_drifts[idx] > peak_drift[idx, 0]:
                        peak_drift[idx, 0] = normalized_drifts[idx]

                # For Y drifts, call nodeDisp Y values previously captured during loop:
                # build temporary arrays for Y displacements
                disp_top_y = np.zeros_like(disp_top)
                disp_bottom_y = np.zeros_like(disp_bottom)
                for ii in range(len(disp_top_y)):
                    disp_top_y[ii] = node_disps[step, ii+1] * 0.0  # placeholder; we need Y values
                # Note: above placeholder exists because we didn't store Y in node_disps; to keep structure identical to original,
                # we need Y peak drift as before — fetch Y displacements now with minimal calls:
                # Instead, collect Y displacements in a small temporary list (one extra pass) to compute Y drifts:
                y_vals = np.zeros(n_nodes)
                for i, node in enumerate(control_nodes):
                    y_vals[i] = ops.nodeDisp(node, 2)
                if len(y_vals) >= 2:
                    y_top = y_vals[1:]
                    y_bottom = y_vals[:-1]
                    y_drift_vals = np.abs(y_top - y_bottom) / h
                    for idx in range(len(y_drift_vals)):
                        if y_drift_vals[idx] > peak_drift[idx, 1]:
                            peak_drift[idx, 1] = y_drift_vals[idx]

            # increment step counter
            step += 1

        # End time-stepping loop

        # Trim arrays to actual steps
        if step < n_steps:
            node_disps = node_disps[:step, :].copy()
            node_accels = node_accels[:step, :].copy()

        # Now that the analysis is finished, get the maximum drift and location
        max_peak_drift = np.max(peak_drift) if peak_drift.size > 0 else 0.0
        if peak_drift.size > 0:
            ind = np.where(peak_drift == max_peak_drift)
            max_peak_drift_dir = 'X' if ind[1][0] == 0 else 'Y'
            max_peak_drift_loc = ind[0][0] + 1
        else:
            max_peak_drift_dir = 'X'
            max_peak_drift_loc = 0

        # Compute peak accelerations in-memory (from node_accels array)
        if node_accels.size > 0:
            # node_accels shape (step, n_nodes), we want per-floor peak accel; original code had shape len(top_nodes)+1
            # we'll compute peak at each floor from node_accels
            per_floor_peak = np.max(np.abs(node_accels), axis=0)  # already in g
            # Place into peak_accel first column
            # peak_accel has length len(top_nodes)+1, which equals n_nodes; map directly
            peak_accel[:, 0] = per_floor_peak
        else:
            peak_accel[:, 0] = 0.0

        # Determine max peak accel and location
        max_peak_accel = np.max(peak_accel)
        ind_a = np.where(peak_accel == max_peak_accel)
        if ind_a[0].size > 0:
            max_peak_accel_dir = 'X' if ind_a[1][0] == 0 else 'Y'
            max_peak_accel_loc = ind_a[0][0]
        else:
            max_peak_accel_dir = 'X'
            max_peak_accel_loc = 0

        # Feedback
        if conv_index == -1:
            print('------ ANALYSIS FAILED --------')
        else:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')

        if pFlag:
            print('Final state = {:d} (-1 for non-converged, 0 for stable)'.format(conv_index))
            print('Maximum peak storey drift {:.3f} radians at storey {:d} in the {:s} direction (Storeys = 1, 2, 3,...)'.format(
                max_peak_drift, max_peak_drift_loc, max_peak_drift_dir))
            print('Maximum peak floor acceleration {:.3f} g at floor {:d} in the {:s} direction (Floors = 0(G), 1, 2, 3,...)'.format(
                max_peak_accel, max_peak_accel_loc, max_peak_accel_dir))

        # Optional animation (downsample frames for speed)
        if save_animation_path is not None:
            try:
                print("\nGenerating NRHA animation...")
                # time array based on dt_ansys and actual steps
                time_array = np.arange(step) * dt_ansys
                # Read ground acceleration (assume first file present) and convert to g if in m/s2
                acc_input_full = np.loadtxt(fnames[0])
                # If gm dt differs from dt_ansys we may need to resample / trim
                # Simple approach: resample (nearest) to time_array using integer factor
                # Determine gm length and dt_gm -> assume user provided dt_gm consistent
                # Use numpy.interp to resample
                gm_time = np.arange(len(acc_input_full)) * dt_gm
                acc_resampled = np.interp(time_array, gm_time, acc_input_full) / 9.81  # convert to g

                # --- INSERT SYNCHRONIZATION HERE ---
                min_len = min(len(time_array), len(acc_resampled), node_disps.shape[0], node_accels.shape[0])

                time_array = time_array[:min_len]
                acc_resampled = acc_resampled[:min_len]
                node_disps = node_disps[:min_len, :]
                node_accels = node_accels[:min_len, :]
                # ------------------------------------

                # Decide downsampling for animation frames (keep animation reasonably fast)
                # Aim ~200-600 frames depending on duration; choose factor so frames <= 400
                max_frames = 200   # instead of 400–600
                frame_step = max(1, len(time_array) // max_frames)
                frames = np.arange(0, len(time_array), frame_step)

                # Call animate_nrha with downsample info by sending the full arrays; animate_nrha will create animation.
                # If animate_nrha supports internal downsampling, it can be used; else it will be given full arrays.
                pl = plotter()
                # pl.animate_nrha(control_nodes=control_nodes,
                #                 acc=acc_resampled,
                #                 dts=time_array,
                #                 nrha_disps=node_disps,
                #                 nrha_accels=node_accels,
                #                 drift_thresholds=drift_thresholds,
                #                 export_path=save_animation_path)

                pl.animate_nrha(control_nodes=control_nodes,
                                acc=acc_resampled[frames],
                                dts=time_array[frames],
                                nrha_disps=node_disps[frames, :],
                                nrha_accels=node_accels[frames, :],
                                drift_thresholds=drift_thresholds,
                                export_path=save_animation_path)
            except Exception as e:
                print(f"Animation generation failed: {e}")


        # Return outputs (same order as original)
        return (control_nodes, conv_index, peak_drift, peak_accel,
                max_peak_drift, max_peak_drift_dir, max_peak_drift_loc,
                max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp)


    def do_incremental_dynamic_analysis(self, fnames, dt_gm, t_max, dt_ansys,
                                        target_drift=0.05, initial_sf = 0.1, hunt_step =2.0,
                                        max_fill_gap=0.2, max_runs =15, capping_drift = 0.10, xi=0.05, pFlag=False):
        """
        Performs Incremental Dynamic Analysis (IDA) using the 'Hunt, Trace and Fill' algorithm as per Vamvatsikos and Cornell (2002, 2004).

        The algorithm first 'hunts' for the collapse capacity by increasing the scale
        factor (SF) geometrically. Once collapse or the target drift is reached, it
        'traces' back with smaller steps for the scaling factor and 'fills' the gaps between
        successful runs to refine the IDA curve.

        Parameters
        ----------
        fnames : list
            List of file paths to the ground motion records for each direction.

        dt_gm : float
            Time-step of the ground motion records.

        t_max : float
            The maximum time duration for each individual analysis run.

        dt_ansys : float
            The integration time-step at which the structural analysis will be conducted.

        target_drift : float, optional, default=0.05
            The drift ratio threshold considered as structural collapse (e.g., 5%).

        initial_sf : float, optional, default=0.1
            The starting scale factor for the first simulation run.

        hunt_step : float, optional, default=2.0
            The geometric multiplier used to increase the scale factor during the 'Hunt' phase.

        max_fill_gap : float, optional, default=0.2
            The maximum allowable gap between scale factors. If a gap is larger, the 'Fill' phase
            will bisect it.

        max_runs : int, optional, default=15
            Maximum total number of nonlinear time-history simulations allowed.

        capping_drift : float, optional, default=0.10
            The drift value assigned to failed or collapsed runs for visualization (flatlining).

        xi : float, optional, default=0.05
            The damping ratio used in the analysis (default is 5%).

        Returns
        -------
        ida_data : dict
            A dictionary where keys are scale factors and values are dictionaries containing
            results (peak drift, acceleration, convergence state, etc.) for that run.

        ordered_sfs : list
            A list of all scale factors tested, in the order they were executed.

        Note
        ----
        The current method assumes the acceleration time-history is in m/s2. Therefore, the acceleration
        values are multiple by a factor of g.

        References
        ----------
        [1] Vamvatsikos, D. and Cornell, C.A. (2002), Incremental dynamic analysis. Earthquake Engng. Struct. Dyn.,
            31: 491-514. https://doi.org/10.1002/eqe.141
        [2] Vamvatsikos D, Cornell CA. Applied Incremental Dynamic Analysis. Earthquake Spectra. 2004;20(2):523-553.
            doi:10.1193/1.1737737
        """

        ida_data = {}
        ordered_sfs = []
        self.run_count = 0

        def run_step(sf_value):
            """Helper function that executes a single simulation step and captures results."""
            if self.run_count >= max_runs:
                print(f"Execution Limit Reached: {max_runs} runs. Skipping SF {sf_value:.3f}")
                return None, None
            print(f" -- Run {self.run_count+1}/{max_runs} | SF: {sf_value:.3f}")

            # Reset environment and rebuild model for the current iteration
            ops.wipe()
            self.compile_model()
            self.do_gravity_analysis()

            # Execute the nonlinear time-history analysis
            res = self.do_nrha_analysis(fnames, dt_gm, sf_value*units.g, t_max, dt_ansys, pFlag=pFlag, xi=xi)

            # # Check convergence state and extract max drift
            raw_max_drift = res[4]
            conv_state = res[1]

            # Check the flatline: if solver failed (state == -1) or drift exceeds target then numerical instability occurred, set the drift a high value where
            # we are certain collapse is attained at (default = 0.10, 10% drift)
            if conv_state == -1 or raw_max_drift >= target_drift:
                final_drift = capping_drift
                print(f"Collapse/Target Reached! Capping drift at {final_drift}")
            else:
                final_drift = raw_max_drift

            self.run_count += 1
            ordered_sfs.append(sf_value)

            # Store results in the main data dictionary
            ida_data[sf_value] = {'control_nodes':      res[0],
                                  'conv_index':         conv_state,
                                  'peak_drift':         res[2],
                                  'peak_accel':         res[3],
                                  'max_peak_drift':     final_drift,
                                  'max_peak_drift_dir': res[5],
                                  'max_peak_drift_loc': res[6],
                                  'max_peak_accel':     res[7],
                                  'max_peak_accel_dir': res[8],
                                  'max_peak_accel_loc': res[9],
                                  'peak_disp':          res[10]}
            return final_drift, conv_state

        # Phase 1: Let's go hunting for collapse!
        # Rapidly increase SF to find the building's limit
        curr_sf = initial_sf
        while self.run_count < max_runs:
            m_drift, state = run_step(curr_sf)
            if m_drift is None:
                break
            if state == -1 or m_drift >= target_drift:
                print(f"Hunt terminated: Limit reached at SF = {curr_sf}")
                break
            curr_sf *=hunt_step

        # Phase 2: Let's trace and fill!
        # Refine the curve by filling in large gaps between existing scale factors
        while self.run_count < max_runs:
            sorted_sfs = sorted(ida_data.keys())
            refined    = False
            for i in range(len(sorted_sfs)-1):
                if self.run_count >= max_runs:
                    break
                sf_low, sf_high = sorted_sfs[i], sorted_sfs[i+1]

                # Only fill if the high side is not already a collapse/capped run, this avoids wasting runs in the flatline region
                if (sf_high - sf_low) > max_fill_gap and ida_data[sf_high]['max_peak_drift'] < 0.10:
                    mid_sf = (sf_low + sf_high)/2.0
                    run_step(mid_sf)
                    refined = True

            if not refined or self.run_count >= max_runs:
                break

        return ida_data, ordered_sfs

Installation
############

Follow these steps to install the ``oq-vmtk`` package and its dependencies. Note that
this procedure implies the installation of the OpenQuake engine dependencies.
This procedure was tested on Windows and Linux OS. It is highly recommended
to use a virtual environment to install this tool. A virtual environment is an
isolated Python environment that allows you to manage dependencies for this
project separately from your system's Python installation. This ensures that the
required dependencies for the OpenQuake engine do not interfere with other Python
projects or system packages, which could lead to version conflicts.

Clone the Repository
--------------------

Open your terminal, and run:

.. code-block:: bash

   cd <virtual_environment_directory>
   git clone https://github.com/GEMScienceTools/oq-vmtk.git
   cd oq-vmtk

Set Up a Virtual Environment
----------------------------

Create a virtual environment to manage dependencies:

.. code-block:: bash

   python -m venv .venv  # On Windows
   python3 -m venv .venv  # On Linux

Activate the virtual environment:

.. code-block:: bash

   .venv\Scripts\activate       # On Windows
   source .venv/Scripts/activate  # On Linux

Install Dependencies
--------------------

Install the appropriate requirements file based on your operating system and
Python version.

**For Windows users:**

.. code-block:: bash

   pip install -r requirements-py311-win64.txt  # Python 3.11
   pip install -r requirements-py312-win64.txt  # Python 3.12

**For Linux users:**

.. code-block:: bash

   pip install -r requirements-py311-linux.txt  # Python 3.11
   pip install -r requirements-py312-linux.txt  # Python 3.12

To check your current Python version, run:

.. code-block:: bash

   python --version

Install the Package
-------------------

Install the ``oq-vmtk`` package in editable mode:

.. code-block:: bash

   pip install -e .

.. note::

   **For macOS users:** OpenSeesPy does not currently support macOS versions running
   on arm64 processors, such as M1 and M2 chips. As a result, newer OpenSeesPy versions
   are not available for macOS. To use OpenSeesPy on a Mac, it is advised to run a virtual
   machine with Linux or Windows.

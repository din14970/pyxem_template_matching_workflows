# Pyxem template matching workflows

[![DOI](https://zenodo.org/badge/425115743.svg)](https://zenodo.org/badge/latestdoi/425115743)

This repository contains all the Jupyter notebook files and other files necessary to reproduce the results and figures in the paper *"Free, flexible and fast: orientation mapping using the multi-core and GPU-accelerated template matching capabilities in the python-based open source 4D-STEM analysis toolbox Pyxem"* which was recently submitted to the journal *Ultramicroscopy*. The paper preprint is available on [arXiv](https://arxiv.org/abs/2111.07347).

What these notebooks show:
* how to use diffsims, orix, and pyxem to perform template matching on a precession electron diffraction (PED) or nanobeam electron diffraction (NBED) dataset. 
* how template matching works under the hood, and which kinds of image pre-processing steps are helpful for getting good results.
* how to build custom workflows like a two-step indexation for finding the orientation of embedded precipitates

## Viewing the workflows

For static views of the notebooks visit:

* <https://nbviewer.org/github/din14970/pyxem_template_matching_workflows/blob/master/210903IndexOptimization.ipynb>
* <https://nbviewer.org/github/din14970/pyxem_template_matching_workflows/blob/master/210907Benchmarking.ipynb>
* <https://nbviewer.org/github/din14970/pyxem_template_matching_workflows/blob/master/210910AdvancedIndexing.ipynb>
* <https://nbviewer.org/github/din14970/pyxem_template_matching_workflows/blob/master/Generating%20figures.ipynb>

## Downloading the necessary data

A lot of data is already included in the `data` directory.
However, this repository does not include the large raw 4D-STEM datasets, they can be downloaded from here:
* [Cu-Ag dataset](https://doi.org/10.5281/zenodo.5595292)
* [G-phase dataset](https://doi.org/10.5281/zenodo.5597738)

Inside the respective notebooks, there are cells that download the correct file in case it is not available.

## Running the notebooks on your computer

### Option 1 (easy): getting the Docker image
Everything in this repository, the data and outputs, as well as all the necessary dependencies were put into a [docker container image](https://docs.docker.com/get-started/overview/).
With this option you don't need to download or configure anything else, everything should be self contained.
The steps to run the notebooks are:

1. [Download and install Docker Engine for your system](https://docs.docker.com/engine/install/). On Linux, ensure the docker daemon is loaded.
2. Pull the docker image with the command 
    ```
    $ docker pull nielscautaerts/pyxem_template_matching_workflows
    ```

3. Run the docker image with
    ```
    $ docker run -p 7000:8888 din14970/pyxem_template_matching_workflows
    ```
    You can change the 7000 port on local host to any port number you like.

4. Now visit <http://localhost:7000>. Enter and submit the token that you may find in the terminal when you ran the `docker run` command (it's a long string of numbers and letters after `token=`). You should now see all the necessary files.

#### Caveats
* I could not get Mayavi to work properly inside the Docker container in the `Generating figures` notebook. Running some of these cells may crash the kernel. This only pertains to a single subfigure of Figure 1 in the paper.
* Even if you have a CUDA enabled GPU, GPU code will not work inside the docker container. You will have to run template matching code on the CPU. I have already included outputs of most of the notebooks in the docker image so you aren't forced to rerun these processing steps in order to play with the data.

If you know how to solve these issues, feel free to modify the Dockerfile in this repository and make a pull request.

### Option 2 (harder): installing all dependencies in a virtual environment
To run the notebooks directly on your computer and get them working properly you may have to do a bit more work.
In order to save you some time I have already included an `environment.yml` file to install the right versions of dependencies.

1. Ensure you have `conda` installed, either with [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Miniconda is much more light weight and recommended.
2. Ensure you have [git](https://git-scm.com/downloads) installed.
3. Clone this repository to a destination of your choosing with
    ```
    $ git clone https://github.com/din14970/pyxem_template_matching_workflows.git
    ```

4. Go inside the repository folder with `cd` and create a new virtual environment from the `environment.yml` file using
    ```
    $ conda env create -f environment.yml
    ```

5. Go out of this folder and clone the repositories for pyxem and diffsims 
    ```
    $ git clone https://github.com/pyxem/diffsims.git
    ```

    and 

    ```
    $ git clone https://github.com/pyxem/pyxem.git
    ```

6. Go into these respective folders and check out specific commits. Then install the versions in your virtual environment.
    ```
    $ conda activate pyxenv
    $ cd diffsims
    $ git checkout 2b75329 
    $ python3 -m pip install -e .
    $ cd ../pyxem
    $ git checkout 30ddc1f5
    $ python3 -m pip install -e .
    ```
    These steps are necessary because the notebooks use specific features which have not yet been packaged into an official version of the software.

7. You should now be able to launch a Jupyter notebook server with `$ jupyter notebook` and open and run the notebooks. Always ensure the virtual environment is active.

If you want GPU acceleration to work you will need to also install `cupy` with a version >9.0.0.
Additionally you may have to install the appropriate NVIDIA drivers and the cuda toolkit.
If you can import `cupy` in a notebook and create an array with `cupy.array([1, 2, 3])` without errors, the GPU acceleration should work.

If Mayavi doesn't work on your system you may have to do some digging on required system packages.
Pyxem does not rely on Mayavi, only one image from the paper relies on this package.

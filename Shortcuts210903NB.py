'''
This file contains the same contents as the notebook but condensed into functional form so that certain things can easily be redone with a single line
'''

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
import skimage.filters as skifi

from diffsims.generators.rotation_list_generators import get_beam_directions_grid
import diffpy
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.diffraction_library import load_DiffractionLibrary
from orix.quaternion.rotation import Rotation
from orix.vector.vector3d import Vector3d
from orix.projections import StereographicProjection


def hspy_to_blo():
    data_file = hs.load("data/Cu-Ag_alloy.hspy", lazy=True)
    data_file.save("data/Cu-Ag_alloy-1000max.blo", intensity_scaling=(0, 1000))
    
    
def get_subset():
    if not os.path.isfile("data/Cu-Ag_alloy-1000max.blo"):
        print("No BLO file was found, attempting to convert the hspy file")
        hspy_to_blo()
    data_file = hs.load("data/Cu-Ag_alloy-1000max.blo", signal_type="electron_diffraction", lazy=True)
    subset = data_file.inav[7:147, 26:86]
    subset.center_direct_beam(method="interpolate",
        half_square_width=30,
        subpixel=True,
        sigma=1.5,
        upsample_factor=2,
        kind="linear",
        )
    subset.set_diffraction_calibration(0.01155)
    return subset


def subtract_background_dog(z, sigma_min, sigma_max):
    blur_max = gaussian_filter(z, sigma_max)
    blur_min = gaussian_filter(z, sigma_min)
    return np.maximum(np.where(blur_min > blur_max, z, 0) - blur_max, 0)
    

def process_image(image):
    median_cols = np.median(image, axis=0)
    image = image - median_cols
    image = image - image.min()
    image = subtract_background_dog(image, 3, 8)
    image = skifi.gaussian(image, sigma=1.5)
    image[image < 1.5] = 0
    image = image**0.5
    image = rescale_intensity(image)
    return image
    
    
def get_processed_subset():
    subset = get_subset()
    subset.change_dtype(np.float32)
    subset.map(process_image)
    return subset
    
    
def calculate_template_library():
    subset = get_subset()
    diffraction_calibration = 0.01155
    half_shape = (subset.data.shape[-2]//2, subset.data.shape[-1]//2)
    reciprocal_radius = np.sqrt(half_shape[0]**2 + half_shape[1]**2)*diffraction_calibration
    resolution = 0.3 
    grid_cub = get_beam_directions_grid("cubic", resolution, mesh="spherified_cube_edge")
    structure_cu = diffpy.structure.loadStructure("data/Cu.cif")
    diff_gen = DiffractionGenerator(accelerating_voltage=200,
                                    precession_angle=0,
                                    scattering_params=None,
                                    shape_factor_model="linear",
                                    minimum_intensity=0.1,
                                    )
    lib_gen = DiffractionLibraryGenerator(diff_gen)
    library_phases_cu = StructureLibrary(["cu"], [structure_cu], [grid_cub])
    diff_lib_cu = lib_gen.get_diffraction_library(library_phases_cu,
                                               calibration=diffraction_calibration,
                                               reciprocal_radius=reciprocal_radius,
                                               half_shape=half_shape,
                                               with_direct_beam=False,
                                               max_excitation_error=0.1)
    diff_lib_cu.pickle_library("data/Cu_lib_0.3deg_0.1me.pickle")
    return diff_lib_cu
    

def load_template_library():
    if os.path.isfile("data/Cu_lib_0.3deg_0.1me.pickle"):
        return load_DiffractionLibrary("data/Cu_lib_0.3deg_0.1me.pickle", True)
    else:
        print("No diffraction library found, calculating templates... This can take a while.")
        return calculate_template_library()
    
    
def grid_to_xy(grid, pole=-1):
    s = StereographicProjection(pole=pole)
    rotations_regular =  Rotation.from_euler(np.deg2rad(grid))
    rot_reg_test = rotations_regular*Vector3d.zvector()
    x, y = s.vector2xy(rot_reg_test)
    return x, y


def load_indexation_result():
    if not os.path.isfile('outputs/210903ResultCu.pickle'):
        print("No indexation result was found, attempting to perform calculation")
        subset = get_processed_subset()
        diff_lib_cu = load_template_library()
        delta_r = 1
        delta_theta = 1 
        max_r = 250
        intensity_transform_function = None
        find_direct_beam = False
        direct_beam_position = None
        normalize_image = False
        normalize_templates = True
        frac_keep = 1
        n_keep = None
        n_best = 5
        try:
            result, phasedict = iutls.index_dataset_with_template_rotation(subset,
                                                            diff_lib_cu,
                                                            n_best = n_best,
                                                            frac_keep = frac_keep,
                                                            n_keep = n_keep,
                                                            delta_r = delta_r,
                                                            delta_theta = delta_theta,
                                                            max_r = max_r,
                                                            intensity_transform_function=intensity_transform_function,
                                                            normalize_images = normalize_image,
                                                            normalize_templates=normalize_templates,
                                                            target="gpu",
                                                            )
        except Exception:
            print("No GPU was found, attempting calculation on CPU")
            result, phasedict = iutls.index_dataset_with_template_rotation(subset,
                                                            diff_lib_cu,
                                                            n_best = n_best,
                                                            frac_keep = frac_keep,
                                                            n_keep = n_keep,
                                                            delta_r = delta_r,
                                                            delta_theta = delta_theta,
                                                            max_r = max_r,
                                                            intensity_transform_function=intensity_transform_function,
                                                            normalize_images = normalize_image,
                                                            normalize_templates=normalize_templates,
                                                            target="cpu",
                                                            )
        with open('outputs/210903ResultCu.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return result
    else:
        with open('outputs/210903ResultCu.pickle', 'rb') as handle:
            return pickle.load(handle)
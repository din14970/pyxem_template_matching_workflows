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
import pyxem.utils.indexation_utils as iutls


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


def to_fundamental(data_sol):
    data_sol = np.abs(data_sol)
    data_sol = np.sort(data_sol, axis=-1)
    column = data_sol[...,0].copy()
    data_sol[..., 0] = data_sol[...,1]
    data_sol[..., 1] = column
    return data_sol


def get_ipf_color(vectors):
    # the following column vectors should map onto R [100], G [010], B[001], i.e. the identity. So the inverse of
    # this matrix maps the beam directions onto the right color vector
    color_corners = np.array([[0, 1/np.sqrt(2), 1/np.sqrt(3)],
                              [0, 0, 1/np.sqrt(3)],
                              [1, 1/np.sqrt(2), 1/np.sqrt(3)]])
    color_corners = np.array([[0, 1, 1],
                              [0, 0, 1],
                              [1, 1, 1]])
    color_mapper = np.linalg.inv(color_corners)
    # a bit of wrangling
    data_sol = to_fundamental(vectors.data)
    flattened = data_sol.reshape(np.product(data_sol.shape[:-1]), 3).T
    rgb_mapped = np.dot(color_mapper, flattened)
    rgb_mapped = np.abs(rgb_mapped / rgb_mapped.max(axis=0)).T
    rgb_mapped = rgb_mapped.reshape(data_sol.shape)
    return rgb_mapped

def get_processed_gdataset():
    data_file = hs.load("data/201009A17-FullData4xbin.hspy", lazy=True)
    data_file.data = data_file.data.rechunk(("auto", "auto", None, None))
    subset = data_file

    subset.change_dtype(np.float32)

    subset.center_direct_beam(
            method="blur",
            half_square_width=50,
            sigma=1.5,
            )

    from pyxem.utils.expt_utils import convert_affine_to_transform, apply_transformation

    transform = convert_affine_to_transform(np.array([[ 0.93356802, -0.04315628,  0.        ],
                                                    [-0.02749365,  0.96883687,  0.        ],
                                                    [ 0.        ,  0.        ,  1.        ]]), subset.data.shape[-2:])
    subset.map(apply_transformation, transformation=transform, keep_dtype=True)

    from scipy.ndimage import gaussian_filter
    from skimage.exposure import rescale_intensity

    def subtract_background_dog(z, sigma_min, sigma_max):
        blur_max = gaussian_filter(z, sigma_max)
        blur_min = gaussian_filter(z, sigma_min)
        return np.maximum(np.where(blur_min > blur_max, z, 0) - blur_max, 0)

    import skimage.filters as skifi

    def process_image(image):
        image = subtract_background_dog(image, 3, 8)
        image[image < 30] = 0
        image = image**0.5
        image = rescale_intensity(image)
        return image

    subset.map(process_image)
    return subset

def get_mask_gdataset():
    average = hs.load("data/210913AverageExport.hspy")
    from skimage.morphology import area_opening, area_closing, dilation, erosion, disk
    mask = average.data < 2e-4
    for _ in range(3):
        mask = dilation(mask, selem=disk(1))
    for _ in range(1):
        mask = erosion(mask, selem=disk(3))
        mask = erosion(mask, selem=disk(2))
    return mask

def get_austenite_result():
    import pickle
    with open('outputs/210921ResultAus.pickle', 'rb') as handle:
        return pickle.load(handle)

def get_g_result():
    import pickle
    with open('outputs/210921ResultG.pickle', 'rb') as handle:
        return pickle.load(handle)

def get_masked_gdataset():
    from skimage.filters import gaussian
    subset = get_processed_gdataset()
    mask = get_mask_gdataset()
    def mask_image(image):
        return image*mask
    subset.map(mask_image)
    # boost intensities and blur a bit
    subset.map(gaussian, sigma=1)
    # we subtract a small amount to punish spots in vacuum
    subset.map(lambda x: x**0.3-0.05)
    return subset


def get_diff_lib_g():
    from diffsims.libraries.diffraction_library import load_DiffractionLibrary
    import copy
    diff_lib_g = load_DiffractionLibrary("data/g_lib_0.5deg_0.01me_1e-7mi.pickle", safety=True)
    # set all intensities to 1, this is helper function
    def func_to_intensity(simulations, function, *args, **kwargs):
        new_sims = []
        for i in simulations:
            new_sim = copy.deepcopy(i)
            new_sim.intensities = function(new_sim.intensities, *args, **kwargs)
            new_sims.append(new_sim)
        return new_sims
    sims = func_to_intensity(diff_lib_g["g"]["simulations"], lambda x: (x>0)*1)
    diff_lib_g["g"]["simulations"] = sims
    return diff_lib_g


def get_diff_lib_aus():
    from diffsims.libraries.diffraction_library import load_DiffractionLibrary
    diff_lib_g = load_DiffractionLibrary("data/Aus_lib_1deg_0.1me.pickle", safety=True)
    return diff_lib_g


def get_stereo_triangle():
    from scipy.interpolate import griddata
    from diffsims.generators.rotation_list_generators import get_beam_directions_grid
    grid_cub = get_beam_directions_grid("cubic", 1, mesh="spherified_cube_edge")

    def ori_to_vec(eulers):
        from orix.quaternion.rotation import Rotation
        from orix.vector.vector3d import Vector3d
        rotations_regular =  Rotation.from_euler(np.deg2rad(eulers))
        return rotations_regular*Vector3d.zvector()

    xy = np.array(grid_to_xy(grid_cub)).T
    colors = get_ipf_color(ori_to_vec(grid_cub))
    reds = colors[:, 0]
    greens = colors[:, 1]
    blues = colors[:, 2]
    sampling=0.001
    gridx, gridy = np.mgrid[-0.05:0.42:sampling, -0.05:0.45:sampling]
    t_rd = griddata(xy, reds, (gridy, gridx), method="linear")
    t_gn = griddata(xy, greens, (gridy, gridx), method="linear")
    t_bl = griddata(xy, blues, (gridy, gridx), method="linear")
    t_alpha = np.invert(np.isnan(t_rd))
    t_rd[np.isnan(t_rd)] = 0
    t_bl[np.isnan(t_bl)] = 0
    t_gn[np.isnan(t_gn)] = 0
    triangle = np.stack([t_rd, t_gn, t_bl, t_alpha], axis=-1)
    triangle[triangle<0]=0
    triangle[triangle>1]=1
    return triangle

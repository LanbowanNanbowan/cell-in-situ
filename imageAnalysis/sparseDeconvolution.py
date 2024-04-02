import os
import numpy as np
from skimage import io, restoration
from skimage.util import img_as_float
from skimage import io, img_as_ubyte, img_as_uint
from skimage import exposure
from matplotlib import pyplot as plt


root_folder = 'F:/1/20240112-1'


for i in range(1, 32):
    sub_folder_path = os.path.join(root_folder, str(i))

    fitc_folder = os.path.join(sub_folder_path, 'FITC')
    tritc_folder = os.path.join(sub_folder_path, 'TRITC')

    output_fitc_folder = os.path.join(sub_folder_path, 'sparseDeconv', 'FITC')
    output_tritc_folder = os.path.join(sub_folder_path, 'sparseDeconv', 'TRITC')

    if not os.path.exists(output_fitc_folder):
        os.makedirs(output_fitc_folder)
    if not os.path.exists(output_tritc_folder):
        os.makedirs(output_tritc_folder)


    for filename in os.listdir(fitc_folder):
        if filename.endswith('.tif'):
            image_path = os.path.join(fitc_folder, filename)
            image = img_as_float(io.imread(image_path))

            psf = np.ones((3, 3)) / 9
            deconvolved_image = restoration.wiener(image, psf, 1)
            deconvolved_image = np.clip(deconvolved_image, 0, 1)
            deconvolved_image_uint16 = img_as_uint(deconvolved_image)

            output_path = os.path.join(output_fitc_folder, filename)
            io.imsave(output_path, deconvolved_image_uint16)


    for filename in os.listdir(tritc_folder):
        if filename.endswith('.tif'):
            image_path = os.path.join(tritc_folder, filename)
            image = img_as_float(io.imread(image_path))

            psf = np.ones((3, 3)) / 9
            deconvolved_image = restoration.wiener(image, psf, 1)
            deconvolved_image = np.clip(deconvolved_image, 0, 1)
            deconvolved_image_uint16 = img_as_uint(deconvolved_image)

            output_path = os.path.join(output_tritc_folder, filename)
            io.imsave(output_path, deconvolved_image_uint16)




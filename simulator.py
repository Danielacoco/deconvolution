import argparse
import numpy as np
import os
import glob
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import tifffile
import tensorflow as tf
from psf_generator import ScalarSphericalPropagator
from scipy.ndimage import convolve as nd_convolve


def pad_psf(psf, target_shape):
    """
    Pad a psf array to the target shape with zeros, centering the original kernel.
    """
    padded = np.zeros(target_shape, dtype=psf.dtype)
    psf_shape = psf.shape
    start_indices = [(t - k) // 2 for t, k in zip(target_shape, psf_shape)]
    slices = tuple(slice(start, start + k) for start, k in zip(start_indices, psf_shape))
    padded[slices] = psf
    return padded

def normalize_psf(psf):
    """
    Normalize the psf so that the sum of all coefficients equals one.
    """
    psf_sum = np.sum(psf)
    if psf_sum != 0:
        return psf / psf_sum
    return psf

def add_gaussian_noise(data, sigma_g):
    """
    Adding Gaussian noise to the data.
    """
    noise = np.random.normal(loc=0.0, scale=sigma_g, size=data.shape)
    return data + noise

def add_poisson_noise(data, alpha_p):
    """
    Adding Poisson noise to the data.
    """
    scaled = data * alpha_p
    scaled[scaled < 0] = 0
    return np.random.poisson(scaled) / alpha_p

def convolution_sample_psf(data, psf):
    """
    Convolve the sample data with the PSF using the FFT-based if the psf and data have same shape.
    otherwise use tensorflow 3d convolution to leverage gpus.
    """
    if data.shape == psf.shape:
        # Use FFT-based convolution if the PSF and data are the same size.
        return fftconvolve(data, psf, mode='same')
    else:
        # Use scipy.ndimage.convolve for data and PSF of different shapes.
        return nd_convolve(data, psf, mode='reflect')
        # # Use TensorFlow 3D convolution.
        # # Convert the data and PSF to TensorFlow tensors.
        # data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
        # psf_tf = tf.convert_to_tensor(psf, dtype=tf.float32)
        
        # # The data we hace is [z, y, x]. Dimensions of the OME-TIFF file: (30, 256, 256)
        # # Reshape the sample data to [batch, depth, height, width, channels].
        # # Here, we assume a single sample with one channel.
        # data_tf = tf.reshape(data_tf, [1] + list(data.shape) + [1])
        
        # # PSF shape is [depth, height, width]. and the once we created is (101, 256, 256)
        # # Reshape the PSF to [filter_depth, filter_height, filter_width, in_channels, out_channels].
        # # We assume a single-channel PSF that produces one output channel.
        # psf_tf = tf.reshape(psf_tf, list(psf.shape) + [1, 1])
        
        # # Perform the 3D convolution using TensorFlow.
        # result = tf.nn.conv3d(data_tf, psf_tf, strides=[1, 1, 1, 1, 1], padding='SAME')
        
        # # Remove the batch and channel dimensions.
        # result = tf.squeeze(result)
        
        # # Convert the TensorFlow tensor back to a NumPy array.
        # return result.numpy()
        

def simulate_microscopy_image(data, psf, sigma_g, alpha_p, background, psf_scale):
    """
    Simulate a microscopy image by convolving 3D data with a PSF and adding noise.

    """

    # TODO: Scale the PSF with the extra factor to control the degree of blur? THEN WE NEED TO NORMALIZE IT?
    psf = psf * psf_scale
    psf = normalize_psf(psf)

    # Convolve the data with the PSF
    result = convolution_sample_psf(data, psf)

    # TODO: Add a constant background value to the convolved image? This would be after the convolution?
    result += background
    
    if not alpha_p:
        print("No Poisson noise will be added.")
    else:
        result = add_poisson_noise(result, alpha_p)
    
    if not sigma_g:
        print("No Gaussian noise will be added.")
    else:        
        result = add_gaussian_noise(result, sigma_g)       
    return result

def load_ome_tif(filename):
    """
    load 3D data from an OME-TIFF file.
    """
    return tifffile.imread(filename) #np.ndarray

def load_data(filename):
    """
    helper to load 3D data from a .npy file.
    """
    return np.load(filename)

def save_data(filename, data):
    """
    Save data to a .npy file.
    """
    np.save(filename, data)

def save_ome_tiff(filename, data):
    """
    Save data to an OME-TIFF file.
    """
    tifffile.imwrite(filename, data)

def generate_default_psf():
    """
    Generate a default PSF using the PSF generator using Vasilikis example
  
    """
    kwargs = {
        'n_pix_pupil': 127,    # Number of pixels for the pupil function
        'n_pix_psf': 256,      # Number of pixels for the PSF
        'na': 1.3,             # Numerical aperture
        'wavelength': 480,     # Wavelength in nm
        'fov': 25600,          # Field of view in nm
        'defocus_min': -5000,  # Minimum defocus in nm
        'defocus_max': 5000,   # Maximum defocus in nm
        'n_defocus': 101,      # Number of defocus slices
    }
    propagator = ScalarSphericalPropagator(**kwargs)
    focus_field = propagator.compute_focus_field()
    psf = np.abs(focus_field).squeeze()**2
    return psf

def process_file(input_file, output_dir, psf, sigma_g, alpha_p, pad_psf, visualize, background, psf_scale):
    """
    Process a single OME-TIFF file.
    """
    data = load_ome_tif(input_file)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    npy_output_file = os.path.join(output_dir, f"{basename}_simulated.npy")
    tiff_output_file = os.path.join(output_dir, f"{basename}_simulated.ome.tif")
    
    if pad_psf:
        psf = pad_psf(psf, data.shape)
        print(f"PSF padded to match sample dimensions: {data.shape}")
    
    simulated = simulate_microscopy_image(data, psf, sigma_g, alpha_p, background, psf_scale)
    save_data(npy_output_file, simulated)
    save_ome_tiff(tiff_output_file, simulated)
    print(f"Simulated image saved to {npy_output_file}")
    print(f"Simulated OME-TIFF saved to {tiff_output_file}")
    
    if visualize:
        visualize_simulation(simulated, basename)

def visualize_simulation(simulated, basename):
    """
    Visualize the middle slice of the simulated image if requested.
    """
    slice_index = simulated.shape[0] // 2
    plt.imshow(simulated[slice_index, :, :], cmap='gray')
    plt.title(f"Simulated Image (Slice at index {slice_index}) for {basename}")
    plt.colorbar()
    plt.show()

def visualize_ome_tiff(filename):
    """
    Visualize the middle slice of an OME-TIFF
    """
    data = load_ome_tif(filename)
    slice_index = data.shape[0] // 2
    plt.imshow(data[slice_index, :, :], cmap='gray')
    plt.title(f"OME-TIFF Image (Slice at index {slice_index}) for {filename}")
    plt.colorbar()
    plt.show()

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simulator to model microscopy images using 3D data. The simulator takes in 3D data (the sample), convolves it with a PSF generated by the PSF generator, and adds Gaussian and Poisson noise using parameters alpha_g and alpha_p."
    )
    parser.add_argument('--input', type=str,
                        help='Input one 3D data file (.ome.tif) format).')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing OME-TIFF files (.ome.tif). All such files will be processed.')
    parser.add_argument('--psf', type=str,
                        help='PSF file (.npy format). If not provided, a default PSF will be generated using the PSF generator.')
    parser.add_argument('--output_dir', type=str, default='simulated_images',
                        help='Output directory where the simulated images will be saved (used when processing multiple files).')
    parser.add_argument('--sigma_g', type=float,
                        help='Gaussian noise standard deviation (read noise). If not provided, no Gaussian noise will be added.')
    parser.add_argument('--alpha_p', type=float,
                        help='Scaling factor for converting intensities into photon counts (shot noise). If not provided, no Poisson noise will be added.')
    parser.add_argument('--visualize', action='store_true',
                        help='Display a middle slice of the simulated image.')
    parser.add_argument('--pad_psf', action='store_true',
                        help='Pad the PSF to match the sample dimensions. This is when we use fftconvolve as convolution method.')
    ##### TODO: CONFIRM WITH TEAM BEFORE ADDING THESE PARAMETERS...
    parser.add_argument('--background', type=float, default=100.0,
                        help='Constant background value to add to the convolved image (simulates baseline detector signal).')
    parser.add_argument('--psf_scale', type=float, default=1.0,
                        help='Additional scaling factor applied to the PSF to control the degree of blur independently of the PSF generator parameters.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Check that at least one input option is provided.
    if not args.input and not args.input_dir:
        print("Error: You must provide either --input or --input_dir.")
        return
    
    # Create output directory.
    if args.input_dir or args.input:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # Load or generate the PSF.
    if args.psf:
        psf = load_data(args.psf) #expecting .npy file
    else:
        psf = generate_default_psf()

    # If --input is provided we only porcess that single single file.
    if args.input:
        print("Processing single file as --input argument was provided. Pricessing only one input file")
        ext = os.path.splitext(args.input)[1].lower()
        if ext not in ['.ome.tif', '.ome.tiff']: #rn only supporting .ome.tif files
            print(f"Error: --input file must be an OME-TIFF file (.ome.tif or .ome.tiff). Provided: {ext}")
            return
        process_file(args.input, args.output_dir, psf, args.sigma_g, args.alpha_p, args.pad_psf, args.visualize, args.background, args.psf_scale)
    # If processing all OME-TIFF files in a directory.
    elif args.input_dir:
        ome_files = glob.glob(os.path.join(args.input_dir, '*.ome.tif')) #find all files to process
        if not ome_files:
            print("No .ome.tif files found in the provided directory.")
            return
        for file in ome_files:
            print(f"Processing {file}...")
            process_file(file, args.output_dir, psf, args.sigma_g, args.alpha_p, args.pad_psf, False, args.background, args.psf_scale)
        return
    else:
        print("No input file or directory provided!")
        return

if __name__ == '__main__':
    main()
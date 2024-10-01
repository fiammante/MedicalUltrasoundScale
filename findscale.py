import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import morphology, io,filters, draw,transform
from PIL import Image, ImageDraw
from scipy.fft import fft, fftfreq, fftshift
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# method to get scale in pixels/cm from a medical ultrasound images
def getscale(img):
    # Compute the 2D FFT of the image
    f_transform = np.fft.fft2(img)
    # Shift the zero frequency component to the center
    f_shift = np.fft.fftshift(f_transform)

    # Create a mask with the same size as the image, 
    # but only keep the low frequencies so that only the regularly spaced dots are kept
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-100:crow+100, ccol-100:ccol+100] = 1

    # Apply the mask to the shifted DFT
    f_shift_filtered = f_shift * (1-mask)

    # Inverse shift and compute the inverse DFT
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # apply threshold to get a clearer image
    img_back=np.where(img_back>63,255,0)
    
    
    # Apply the dilation to increase size of scale dots
    # Create a structuring element (e.g., a disk of radius 2)
    selem = morphology.disk(2)
    img_backlow = morphology.dilation(img_back, selem)
    # now find the upper dot in the top center of image where the center of the probe is
    left=ccol-5
    img_backlow_middle=img_backlow[:,ccol-5:ccol+5]
    img_backlow_middle_sum=np.sum(img_backlow_middle,axis=-1) 
    non_zero_indices_y = np.nonzero(img_backlow_middle)
    cx,cy=(non_zero_indices_y[0][0],non_zero_indices_y[1][0]+left)
    
    # now that we have the center make all rays from center horizontal with a polar warp
    img_backlow_polar=transform.warp_polar(img_backlow, center=(cx,cy))
    # only keep the noise free zones (remove too close or too far from center
    img_backlow_polar[:,0:25]=0
    img_backlow_polar[:,-200:-1]=0
    # Fuse points on a line by applying dilation with a large horizontal structuring element 
    selem = morphology.ellipse(width=25, height=1)
    dilated_image = morphology.dilation(img_backlow_polar, selem)
    # find the longest line 
    sumx=np.sum(dilated_image,axis=1)
    max_index = np.argmax(sumx)
    print("max index",max_index)
    longest=img_backlow_polar[ max_index,:]
    # Apply Gaussian filter to only keep major peaks as some noise still occur 
    sigma = 9.0  # Standard deviation for Gaussian kernel
    smoothed_array = gaussian_filter(longest, sigma=sigma)
    peaks, _ = find_peaks(smoothed_array)
    print(peaks)
    pixel_by_cm=(peaks[-1]-peaks[0])/(len(peaks)-1)
    #  
    # Plot the original and the filtered image the scale line and peaks information
    plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(222), plt.imshow(img_backlow, cmap='gray'), plt.title('Low Frequencies Image')
    
    plt.subplot(224), plt.plot(np.arange(len(smoothed_array)), smoothed_array), plt.title('image peaks: '+str(len(peaks))+', pixels/cm: '+str(pixel_by_cm))
    plt.subplot(223), plt.imshow(img_backlow_polar, cmap='gray'), plt.title('Polar Image')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
   
import glob

folder_dir = "your_medical_ultrasound_path"
for video_path in glob.iglob(f"{folder_dir}/MOVIE*.mp4"): 
    print(video_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame only to find scale on image
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if ret:
        (height,width)=frame.shape[:2]
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(frame)
        # Create a mask image with the same size as the original image

        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        # mask all data outside of a 90° pie slide containing the ultrasound image.
        # Define the bounding box for the pie slice
        bbox = [0, -height, width, height]
        # Draw the 90° pie slice
        draw.pieslice(bbox, start=45, end=135, fill=255)
        # Apply the mask to the image
        result = Image.composite(image, Image.new('RGB', (width, height), (0, 0, 0)), mask)
        # Load the image in grayscale
        img = cv2.cvtColor(np.array(result), cv2.COLOR_BGR2GRAY)
        
        getscale(img)
    else:
        print("Failed to read the video")

    cap.release()
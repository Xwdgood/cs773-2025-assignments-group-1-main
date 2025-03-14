 
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import cv2  #only used for convolution


from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth


# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(r_pixel_array,g_pixel_array,b_pixel_array,image_width,image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):

    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage


# function to apply sobel filter to the image and show the result ----by Wenduo
def sobelFilter(px_array_left_original, px_array_right_original):
    #preparation for the sobel filter and convolution
    print("Wenduo:the type of original output is ", type(px_array_left_original))
    px_array_left_original = np.array(px_array_left_original, dtype=np.float32)
    px_array_right_original = np.array(px_array_right_original, dtype=np.float32)
    print("Wenduo:the shift type of output is ", type(px_array_left_original))

    #define sobel filter
    sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32)

    sobel_y_kernel = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=np.float32)
    
    #implement sobel filter at x axis and y axis, 
    IxLeft = cv2.filter2D(px_array_left_original, cv2.CV_32F, sobel_x_kernel)
    IxRight = cv2.filter2D(px_array_right_original, cv2.CV_32F, sobel_x_kernel)

    IyLeft = cv2.filter2D(px_array_left_original, cv2.CV_32F, sobel_y_kernel)
    IyRight = cv2.filter2D(px_array_right_original, cv2.CV_32F, sobel_y_kernel)

    #show sobel filter images
    fig2, axs2 = pyplot.subplots(2, 2)

    axs2[0,0].set_title('X_Sobel Left Image (Grayscale)')
    axs2[0,1].set_title('X_Sobel Right Image (Grayscale)')
    axs2[0,0].imshow(IxLeft, cmap='gray')
    axs2[0,1].imshow(IxRight, cmap='gray')

    axs2[1,0].set_title('Y_Sobel Left Image (Grayscale)')
    axs2[1,1].set_title('Y_Sobel Right Image (Grayscale)')
    axs2[1,0].imshow(IyLeft, cmap='gray')
    axs2[1,1].imshow(IyRight, cmap='gray')

    #add space between subplots
    pyplot.subplots_adjust(hspace=0.5)
    pyplot.show()
    
    return IxLeft, IxRight, IyLeft, IyRight

def WDharrisCornerDetector(Ix, Iy, image_width, image_height, alpha, maxpoints):
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # compute the harris response
    detM = (Ixx * Iyy) - (Ixy ** 2)
    traceM = Ixx + Iyy
    C = detM - alpha * (traceM ** 2)

    
    # Convert C to a 1D array and get sorted indices
    C_flat = C.flatten()
    sorted_indices = np.argsort(C_flat)[::]  # Sort in descending order by response value

    # Select the top maxpoints indices
    if maxpoints < len(sorted_indices):
        sorted_indices = sorted_indices[:maxpoints]

    # Convert 1D indices to (row, col) coordinates
    coordinates = np.array(np.unravel_index(sorted_indices, (image_height, image_width))).T

    return coordinates  # Return a list of corner points in (y, x) coordinates

    


# This is our code skeleton that performs the stitching
def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    (image_width, image_height, px_array_left_original)  = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)
    
    #apply sobel filter to the image and show the result ----by Wenduo
    Ixl,Ixr,Iyl,Iyr=sobelFilter(px_array_left_original, px_array_right_original)
    
    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)
    end = timer()
    print("3x3 elapsed time image smoothing: ", end - start)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    #region play for 5x5, 7x7, 15x15 Gaussian Averaging ----by Wenduo
    start = timer()
    px_array_left5 = IPSmooth.WDcomputeGaussianAveraging5x5(px_array_left_original, image_width, image_height)
    px_array_right5 = IPSmooth.WDcomputeGaussianAveraging5x5(px_array_right_original, image_width, image_height)
    end = timer()
    print("5x5 elapsed time image smoothing: ", end - start)
    px_array_left5 = IPPixelOps.scaleTo0And255AndQuantize(px_array_left5, image_width, image_height)
    px_array_right5 = IPPixelOps.scaleTo0And255AndQuantize(px_array_right5, image_width, image_height)

    start = timer()
    px_array_left7 = IPSmooth.WDcomputeGaussianAveraging7x7(px_array_left_original, image_width, image_height)
    px_array_right7 = IPSmooth.WDcomputeGaussianAveraging7x7(px_array_right_original, image_width, image_height)
    end = timer()
    print("7x7 elapsed time image smoothing: ", end - start)
    px_array_left7 = IPPixelOps.scaleTo0And255AndQuantize(px_array_left7, image_width, image_height)
    px_array_right7 = IPPixelOps.scaleTo0And255AndQuantize(px_array_right7, image_width, image_height)

    start = timer()
    px_array_left15 = IPSmooth.WDcomputeGaussianAveraging15x15(px_array_left_original, image_width, image_height)
    px_array_right15 = IPSmooth.WDcomputeGaussianAveraging15x15(px_array_right_original, image_width, image_height)
    end = timer()
    print("15x15 elapsed time image smoothing: ", end - start)
    px_array_left15 = IPPixelOps.scaleTo0And255AndQuantize(px_array_left15, image_width, image_height)
    px_array_right15 = IPPixelOps.scaleTo0And255AndQuantize(px_array_right15, image_width, image_height)

    #display the images after using different Gaussian Averaging cores ----by Wenduo
    fig3, axs3 = pyplot.subplots(4, 2)
    axs3[0,0].set_title('3x3 Gaussian Averaging left ')
    axs3[0,1].set_title('3x3 Gaussian Averaging right')
    axs3[0,0].imshow(px_array_left, cmap='gray')
    axs3[0,1].imshow(px_array_right, cmap='gray')

    axs3[1,0].set_title('5x5 Gaussian Averaging left ')
    axs3[1,1].set_title('5x5 Gaussian Averaging right')
    axs3[1,0].imshow(px_array_left5, cmap='gray')
    axs3[1,1].imshow(px_array_right5, cmap='gray')
    
    axs3[2,0].set_title('7x7 Gaussian Averaging left ')
    axs3[2,1].set_title('7x7 Gaussian Averaging right')
    axs3[2,0].imshow(px_array_left7, cmap='gray')
    axs3[2,1].imshow(px_array_right7, cmap='gray')

    axs3[3,0].set_title('15x15 Gaussian Averaging left ')
    axs3[3,1].set_title('15x15 Gaussian Averaging right')
    axs3[3,0].imshow(px_array_left15, cmap='gray')
    axs3[3,1].imshow(px_array_right15, cmap='gray')

    #add space between subplots
    pyplot.subplots_adjust(hspace=0.7)
    pyplot.show()
    #endregion

    #Create the harris corner detector
    alpha = 0.06
    maxpoints = 1000
    corners_left = WDharrisCornerDetector(Ixl, Iyl, image_width, image_height, alpha, maxpoints)
    corners_right = WDharrisCornerDetector(Ixr, Iyr, image_width, image_height, alpha, maxpoints)
    # some simplevisualizations
    fig1, axs1 = pyplot.subplots(1, 2)

    axs1[0].set_title('Harris response left overlaid on orig image')
    axs1[1].set_title('Harris response right overlaid on orig image')
    axs1[0].imshow(px_array_left, cmap='gray')
    axs1[1].imshow(px_array_right, cmap='gray')

    # plot red points in the center of each image
    for y, x in corners_left:
        circle = Circle((x, y), 3.5, color='r')
        axs1[0].add_patch(circle)

    for y, x in corners_right:
        circle = Circle((x, y), 3.5, color='r')
        axs1[1].add_patch(circle)

    pyplot.show()

    # a combined image including a red matching line as a connection patch artist (from matplotlib)

    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)

    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")

    pointA = (image_width/2, image_height/2)
    pointB = (3*image_width/2, image_height/2)
    connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
    ax.add_artist(connection)

    pyplot.show()



if __name__ == "__main__":
    main()


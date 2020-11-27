import sys
sys.path.append("../hw1")
import hw1
import numpy as np
import cv2
import queue

def processImage(path, color='gray'):
    img = cv2.imread(path,1)
    if color == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def logFilter(kernel_size=5, sigma=1):

    def logValue(x,y,sigma):
        rs = x**2 + y**2 
        hr = np.exp(-rs/(2.*sigma**2))
        w = ((rs-(2.0*sigma**2))/sigma**4)
        normal = 1 / (2.0 * np.pi * sigma**2)
        kernel = (hr*w)/ normal
        return kernel

    half = int(kernel_size/2)
    filter = np.zeros((kernel_size,kernel_size))

    for i in range(half+1):
        for j in range(half+1):
            value = logValue(i,j,sigma)
            filter[i+half,j+half] = value
            filter[-i+half,j+half]  = value
            filter[i+half,-j+half]  = value
            filter[-i+half, -j+half] = value
    # filter = filter - np.mean(filter)
    # filter /= sigma
    return filter

def sobel(degree):
    if degree == 0:
        sobel = [[-1,-2,-1],[0,0,0],[1,2,1]]
    elif degree == 45:
        sobel = [[-2,-1,0],[-1,0,1],[0,1,2]]
    elif degree == 90:
        sobel = [[-1,0,1],[-2,0,2],[-1,0,1]]
    elif degree == 135:
        sobel = [[0,1,2],[-1,0,1],[-2,-1,0]]
    return sobel

def prewitt(degree):
    if degree == 0:
        prewitt = [[-1,-1,-1],[0,0,0],[1,1,1]]
    elif degree == 45:
        prewitt = [[-1,-1,0],[-1,0,1],[0,1,1]]
    elif degree == 90:
        prewitt = [[-1,0,1],[-1,0,1],[-1,0,1]]
    elif degree == 135:
        prewitt = [[0,1,1],[-1,0,1],[-1,-1,0]]
    return prewitt

def gradient(image, filter='sobel', degree=0):

    if filter == 'sobel':
        sobel_filter = np.array(sobel(degree))
        new_image = convolve2D(image, sobel_filter)
        return new_image
    elif filter == 'prewitt':
        prewitt_filter = np.array(prewitt(degree))
        new_image = convolve2D(image, prewitt_filter)
        return new_image

def Gaussian(s, t, variance):
    r2 = s**2 + t**2
    return np.exp(-1*(r2/2*variance))


# correlation
def getGaussian(filter_size, variance):
    Gfilter = np.empty((filter_size,filter_size))
    offset = int(filter_size/2)
    # generate filter
    for i in range(filter_size):
        for j in range(filter_size):
            Gfilter[i][j] = Gaussian(i-offset, j-offset, variance)
    Gfilter = np.divide(Gfilter, np.sum(Gfilter))
    return Gfilter

def GBlur(image, kernel_size=5, sigma=1.4):
    """
    args:
    image : The input image
    kernel_size : kernel size
    sigma : The width parameter of the Gaussian filter
    """
    g_filter = getGaussian(kernel_size, sigma)
    new_image = convolve2D(image, g_filter)
    new_image = transferType(new_image)
    return new_image

def LoG(image, kernel_size=5, sigma=1.4):
    """
    args:
    image : The input image
    kernel_size : kernel size
    sigma : The width parameter of the Gaussian filter
    """
    log_filter = logFilter(kernel_size, sigma)
    new_image = convolve2D(image, log_filter)
    new_image = transferType(new_image)
    return new_image

def convolve2D(image, kernel, padding=0, pad_method='0'):
    kernel = np.flipud(np.fliplr(kernel))
    kernel_size = kernel.shape[0]
    img_height = image.shape[0]
    img_width = image.shape[1]
    new_image_height = int((img_height-kernel_size+2*padding)+1)
    new_image_width = int((img_width-kernel_size+2*padding)+1)
    new_image = np.zeros((new_image_height,new_image_width))
    if padding != 0:
        if pad_method == '0':
            pad_img = np.pad(image, padding, constant_value=0)
        else:
            pad_img = np.pad(image, padding, mode=pad_method)
    else:
        pad_img = image

    for row in range(img_height):
        if row > img_height - kernel_size:
            break
        for col in range(img_width):
            if col > img_width - kernel_size:
                break
            local = pad_img[row:row+kernel_size, col:col+kernel_size]
            new_image[row , col] = (local*kernel).sum()
    return new_image

def transferType(image):
    mn = image.min()
    mx = image.max()
    mx -= mn
    image = ((image - mn)/mx) * 255
    return image.astype(np.uint8)

def normalize(image):
    max = np.max(image)
    image /= max
    return image

def threshold(image, threshold=0):
    new_image = image.copy()
    if threshold == 0:
        new_image[new_image>0] = 255
        return new_image
    else:
        new_image[new_image>=threshold] = 255
        new_image[new_image<threshold] = 0
        return new_image

def canny(image, threshold1, threshold2):
    # 1.gaussian smooth
    image = GBlur(image)
    #image = transferType(image)
    # 2.gradient
    Gx = gradient(image, filter='sobel', degree=0)
    Gy = gradient(image, filter='sobel', degree=90)
    #magnitude = np.sqrt(Gx**2+Gy**2)
    magnitude = transferType(np.abs(Gx) + np.abs(Gy))
    # gradient direction
    theta = ((np.arctan(Gy/Gx))/np.pi) * 180
    theta[theta < 0] += 180

    # 3.non-maximum suppresion
    nms = magnitude.copy()
    for i in range(theta.shape[0]):
        if i == 0 or i == (theta.shape[0]-1) : continue
        for j in range(theta.shape[1]):
            if j == 0 or j == (theta.shape[1]-1) : continue
            # degree 0
            if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                if(magnitude[i, j] <= magnitude[i, j-1]) and (magnitude[i, j] <= magnitude[i, j+1]):
                    nms[i, j] = 0
            # degree 45
            elif (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                if(magnitude[i, j] <= magnitude[i+1, j+1]) and (magnitude[i, j] <= magnitude[i-1, j-1]):
                    nms[i, j] = 0
            # degree 90
            elif (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                if(magnitude[i, j] <= magnitude[i+1, j]) and (magnitude[i, j] <= magnitude[i-1, j]):
                    nms[i, j] = 0
            # degree 135
            elif (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                if(magnitude[i, j] <= magnitude[i+1, j-1]) and (magnitude[i, j] <= magnitude[i-1, j+1]):
                    nms[i, j] = 0
    

    # 4.double threshold
    strong = nms.copy()
    weak = nms.copy()
    # strong edge
    strong[strong>threshold2] = 255
    strong[strong<threshold1] = 0
    # weak edge
    weak[weak<threshold1] = 0
    weak[weak>threshold2] = 0

    # 5. Hystersis
    edgeImage = strong.copy()
    for i in range(edgeImage.shape[0]):
        if i == 0 or i == (edgeImage.shape[0]-1): continue
        for j in range(theta.shape[1]):
            if j == 0 or j == (edgeImage.shape[1]-1): continue
            if(edgeImage[i,j]!=0):
                # degree 0
                if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                    if(weak[i,j-1]!=0):
                        edgeImage[i,j-1] = 255
                    if(weak[i,j+1]!=0):
                        edgeImage[i,j+1] = 255
                # degree 45
                elif (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                    if(weak[i-1,j-1]!=0):
                        edgeImage[i-1,j-1] = 255
                    if(weak[i+1,j+1]!=0):
                        edgeImage[i+1,j+1] = 255
                # degree 90
                elif (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                    if(weak[i-1,j]!=0):
                        edgeImage[i-1,j] = 255
                    if(weak[i+1,j]!=0):
                        edgeImage[i+1,j] = 255
                # degree 135
                elif (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                    if(weak[i+1,j+1]!=0):
                        edgeImage[i+1,j+1] = 255
                    if(weak[i-1,j-1]!=0):
                        edgeImage[i-1,j-1] = 255
    return nms, edgeImage

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------Display----------------
def show_histogram(image ,model=None, title=None):
    dim = image.ndim
    height = image.shape[0]
    width = image.shape[1]
    pix_range = np.arange(256)
    if(dim==2):
        hist = np.zeros(256)
        for row in range(height):
            for col in range(width):
                value = image[row][col]
                hist[value] += 1
        hist = np.divide(hist, height*width)
        fig, ax = plt.subplots()
        ax.set_ylim(0,0.2)
        ax.set_title(title)
        ax.bar(pix_range, hist)
        plt.show()
    else:
        channel = image.shape[2]
        hist = hist = np.zeros((channel,256))
        for ch in range(channel):
            for row in range(height):
                for col in range(width):
                    value = image[row][col][ch]
                    hist[ch,value] += 1
        hist = np.divide(hist, height*width)
        fig, ax = plt.subplots(channel)
        for ch in range(channel):
            ax[ch].set_title(model[ch])
            ax[ch].set_ylim(0,0.2)
            ax[ch].bar(pix_range, hist[ch])
        plt.show()




# ------------Transformation----------------
def histogramEqualize(image):

    height = image.shape[0]
    width = image.shape[1]
    histogram = np.zeros([255])
    cdf = np.empty([255])
    L = 255 

    # histogram compute
    v = image.flatten()
    for item in v:
        histogram[item-1] += 1
    histogram = np.divide(histogram, len(v))
    # cdf
    for i in range(len(histogram)):
        cdf[i] = np.sum(histogram[:(i+1)])
    # compute s 
    for i in range(height):
        for j in range(width):
            image[i][j] = L * cdf[image[i][j]-1]

def my_powerLaw(image, c, gamma):
    image = np.array(255*(c*(image/255)**gamma), dtype='uint8')
    return image

# piecewise transform
def piecewise(pixel, r1, r2, s1, s2):
    if(pixel < r1):
        return round(pixel/r1)*s1
    elif(pixel < r2):
        return round( (pixel-r1) / (r2-r1) * (s2-s1) + s1)
    else:
        return round( (pixel-r2) / (255-r2) * (255-s2) + s2)

def pw_trans(image, r1, r2, s1, s2):
    img_dim = image.ndim
    height = image.shape[0]
    width = image.shape[1]
    if(img_dim==2):
        new_img = np.empty((height,width))
        for row in range(height):
            for col in range(width):
                new_img[row, col] = piecewise(image[row, col], r1, r2, s1, s2)
        return new_img.astype('uint8')
    else:
        channel = image.shape[2]
        new_img = np.empty((height,width,channel))
        for ch in range(channel):
            for row in range(height):
                for col in range(width):
                    new_img[row, col, ch] = piecewise(image[row, col, ch], r1, r2, s1, s2)
        return new_img.astype('uint8')

# ------------Filter----------------

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

# Gaussian filter
def My_GaussianFilter(image, Filter_size, variance):
    img_dim = image.ndim
    height = image.shape[0]
    width = image.shape[1]
    Gfilter = getGaussian(Filter_size, variance)
    pad_width = int(Filter_size/2)
    #zero padding
    if(img_dim == 2):
        pad_img = np.pad(image, pad_width, constant_values=0)
        new_img = np.empty((height, width))
        for row in range(height):
            row_end = row + Filter_size
            for col in range(width):   
                col_end = col + Filter_size
                local = pad_img[row : row_end, col:col_end]
                value = np.dot(local.flatten(),Gfilter.flatten())
                new_img[row,col] = value
        return new_img.astype('uint8')
    else: 
        channel = image.shape[2]
        new_img = np.empty((height, width,channel))
        for ch in range(channel):
            pad_img = np.pad(image[:,:,ch], pad_width, constant_values=0)
            for row in range(height):
                row_end = row + Filter_size
                for col in range(width):   
                    col_end = col + Filter_size
                    local = pad_img[row : row_end, col:col_end]
                    value = np.dot(local.flatten(),Gfilter.flatten())
                    new_img[row,col,ch] = value
        return new_img.astype('uint8')

    

# Averaging filter
def My_AvgFilter(image, filter_size):
    filter = np.full([filter_size,filter_size], 1)
    filter /= np.sum(filter)
    return np.round(np.dot(image.flatten(), filter.flatten()))
    
# Adaptive local noise reduction filter
def My_ALNRF(image, filter_size, noise_variance):
    height = image.shape[0]
    width = image.shape[1]
    avg_filter = np.full([filter_size,filter_size], 1)
    avg_filter = np.divide(avg_filter, np.sum(avg_filter))
    pad_width = int(filter_size/2)
    image = np.pad(image, pad_width, constant_values=0)
    new_img = np.empty((height, width))
    for i in range(pad_width, height+pad_width):
        for j in range(pad_width, width+pad_width):
            local_var = np.var(image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1])
            local_mean = np.mean(image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1])
            var_ration = np.abs(local_var/noise_variance)
            print(local_var)
            if(var_ration == 1):
                new_img[i-pad_width][j-pad_width] = np.round(np.dot(image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1].flatten(), avg_filter.flatten()))
            else:
                new_img[i-pad_width][j-pad_width] = image[i][j] - (noise_variance/local_var)*(image[i][j]-local_mean)
    return new_img

# median filter
def findMid(image, filter_size):
    mid = int(filter_size/2) - 1
    sorted_seq = np.sort(image.flatten())
    return sorted_seq[mid]

def My_MedianFilter(image, filter_size):
    img_dim = image.ndim
    height = image.shape[0]
    width = image.shape[1]
    pad_width = int(filter_size/2)
    if(img_dim==2):
        new_img = np.empty((height,width))
        pad_img = np.pad(image, pad_width, constant_values=0)
        for row in range(height):
            for col in range(width):
                row_end = row+pad_width+1
                col_end = col+pad_width+1
                new_img[row,col] = findMid(pad_img[row : row_end, col : col_end], filter_size)
        return new_img.astype('uint8')
    else:
        channel = image.shape[2]
        new_img = np.empty((height,width,channel))
        for ch in range(channel):
            pad_img = np.pad(image[:,:,ch], pad_width, constant_values=0)
            for row in range(height):
                for col in range(width):
                    row_end = row+pad_width+1
                    col_end = col+pad_width+1
                    new_img[row,col,ch] = findMid(pad_img[row : row_end, col : col_end], filter_size)
        return new_img.astype('uint8')
#Laplacian filter
def Laplacian(image, negative=False, filter_number='0'):
    img_dim = image.ndim
    height = image.shape[0]
    width = image.shape[1]
    # create filter
    if(filter_number=='0'):
        filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    elif(filter_number=='1'):
        filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    if(negative):
        filter *= -1
    # different cases of single channel or multiple channels
    if(img_dim==2):
        new_img = np.empty((height,width))
        pad_img = np.pad(image, 1, constant_values=0)
        for row in range(height):
            row_end = row + 3  
            for col in range(width):
                col_end = col + 3
                local = pad_img[row : row_end, col : col_end]
                cor = np.dot(local.flatten(),filter.flatten())
                new_img[row, col] = cor
        new_img = np.where(new_img < 0., 0., new_img)
        new_img = np.where(new_img > 255., 255., new_img)
        new_img = new_img.astype('uint8')
        return new_img
    else:
        channel = image.shape[2]
        new_img = np.empty((height, width, channel))
        # correlation
        for ch in range(channel):
            pad_img = np.pad(image[:,:,ch], 1, constant_values=0)
            for row in range(height):
                row_end = row + 3  
                for col in range(width):
                    col_end = col + 3
                    local = pad_img[row : row_end, col : col_end]
                    cor = np.dot(local.flatten(),filter.flatten())
                    new_img[row, col, ch] = cor

        new_img = np.where(new_img < 0., 0., new_img)
        new_img = np.where(new_img > 255., 255., new_img)
        new_img = new_img.astype('uint8')
        return new_img

def weightedGaussian(local, filter_size, variance_d, variance_i):
    """
    variance_d = variance of distance
    variance_i = variance of intensity
    """
    offset = int(filter_size/2)
    WGfilter = np.empty((filter_size, filter_size))
    GF_d = getGaussian(filter_size, variance_d)

    diff_i = np.abs(local-local[offset, offset])
    GF_i = np.exp(-1*( diff_i**2 / 2 * variance_i))            
    WGfilter = np.multiply(GF_d, GF_i)
    return WGfilter

# the implement is only available for single channel
def Bilateral(image, filter_size, variance_d, variance_i):
    #img_dim = image.ndim
    height = image.shape[0]
    width = image.shape[1]
    pad_width = int(filter_size/2)
    pad_img = np.pad(image, pad_width, constant_values=0)
    new_img = np.empty((height, width))
    for row in range(height):
        row_end = row + filter_size
        for col in range(width):
            col_end = col + filter_size
            local = pad_img[row : row_end ,col : col_end]
            WGfilter = weightedGaussian(local, filter_size, variance_d, variance_i)
            WGfilter = np.divide(WGfilter, np.sum(WGfilter))
            new_img[row, col] = np.dot(local.flatten(), WGfilter.flatten())
    return new_img.astype('uint8')



# ------------Zoom----------------


# Bilinear interpolation 
def Bilinear_Zoom(image, new_height, new_width):
    img_dim = image.ndim
    oringin_height = image.shape[0]
    oringin_width = image.shape[1]
    height_ratio = float(oringin_height)/float(new_height)
    width_ratio = float(oringin_width)/float(new_width)
    #single channel image
    if(img_dim==2):
        new_img = np.empty((new_height, new_width))
    else:
        channel = image.shape[2]
        new_img = np.empty((new_height, new_width, channel))
    for row in range(new_height):
        for col in range(new_width):
            # calculate position in original image
            orin_x = col * width_ratio
            orin_y = row * height_ratio            
            orin_xi = int(orin_x)
            orin_yi = int(orin_y)
            orin_xf = orin_x - orin_xi
            orin_yf = orin_y - orin_yi
            orin_xi_plus_one = min(orin_xi+1, oringin_width-1)
            orin_yi_plus_one = min(orin_yi+1, oringin_height-1)
            # four corners in original image
            tl = image[orin_yi, orin_xi]
            tr = image[orin_yi, orin_xi_plus_one]
            bl = image[orin_yi_plus_one, orin_xi]
            br = image[orin_yi_plus_one, orin_xi_plus_one]
            # calculate interpolation
            top = (1. - orin_xf) * tl + orin_xf * tr
            bottom = (1. - orin_yf) * bl + orin_yf * br
            new_value = (1. - orin_yf)*top + orin_yf * bottom
            new_img[row, col] = np.around(new_value)
    return new_img.astype('uint8')

globals().clear()

import cv2
import pymeanshift as pms
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

GSVimagesRoot = r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs\test'
GSVsegRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test_results\P1_B1"

test_seg = cv2.imread(r'D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test_results\P1_B1\P1_B1_18_60.png',cv2.IMREAD_UNCHANGED)
# 看一下颜色
cv2.imshow("seg_image resized",test_seg)
cv2.waitKey(0)
cv2.destroyAllWindows()

green = [107, 142, 35]  # tree:red green blue
light_green = [152, 251, 152]  # bush

from PIL import Image
import pymeanshift as pms

def graythresh(array, level):
    '''array: is the numpy array waiting for processing
    return thresh: is the result got by OTSU algorithm
    if the threshold is less than level, then set the level as the threshold
    by Xiaojiang Li
    '''

    import numpy as np

    maxVal = np.max(array)
    minVal = np.min(array)

    #   if the inputImage is a float of double dataset then we transform the data
    #   in to byte and range from [0 255]
    if maxVal <= 1:
        array = array * 255
        # print "New max value is %s" %(np.max(array))
    elif maxVal >= 256:
        array = np.int((array - minVal) / (maxVal - minVal))
        # print "New min value is %s" %(np.min(array))

    # turn the negative to natural number
    negIdx = np.where(array < 0)
    array[negIdx] = 0

    # calculate the hist of 'array'
    dims = np.shape(array)
    hist = np.histogram(array, range(257))
    P_hist = hist[0] * 1.0 / np.sum(hist[0])

    omega = P_hist.cumsum()

    temp = np.arange(256)
    mu = P_hist * (temp + 1)
    mu = mu.cumsum()

    n = len(mu)
    mu_t = mu[n - 1]

    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))

    # try to found if all sigma_b squrered are NaN or Infinity
    indInf = np.where(sigma_b_squared == np.inf)

    CIN = 0
    if len(indInf[0]) > 0:
        CIN = len(indInf[0])

    maxval = np.max(sigma_b_squared)

    IsAllInf = CIN == 256
    if IsAllInf != 1:
        index = np.where(sigma_b_squared == maxval)
        idx = np.mean(index)
        threshold = (idx - 1) / 255.0
    else:
        threshold = level

    if np.isnan(threshold):
        threshold = level

    return threshold

def VegetationClassification(Img):
    '''
    This function is used to classify the green vegetation from GSV image,
    This is based on object based and otsu automatically thresholding method
    The season of GSV images were also considered in this function
        Img: the numpy array image, eg. Img = np.array(Image.open(StringIO(response.content)))
        return the percentage of the green vegetation pixels in the GSV image

    '''

    import pymeanshift as pms
    import numpy as np

    Img = r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\POI268\input\P3_B2_268_120.jpg'
    # Load the original image
    Img = Image.open(Img)

    # use the meanshift segmentation algorithm to segment the original GSV image
    (segmented_image, labels_image, number_regions) = pms.segment(Img, spatial_radius=6,
                                                                  range_radius=7, min_density=40)

    I = segmented_image / 255.0

    red = I[:, :, 0]
    green = I[:, :, 1]
    blue = I[:, :, 2]

    # calculate the difference between green band with other two bands
    green_red_Diff = green - red
    green_blue_Diff = green - blue

    ExG = green_red_Diff + green_blue_Diff
    diffImg = green_red_Diff * green_blue_Diff

    redThreImgU = red < 0.6
    greenThreImgU = green < 0.9
    blueThreImgU = blue < 0.6

    shadowRedU = red < 0.3
    shadowGreenU = green < 0.3
    shadowBlueU = blue < 0.3
    del red, blue, green, I

    greenImg1 = redThreImgU * blueThreImgU * greenThreImgU
    greenImgShadow1 = shadowRedU * shadowGreenU * shadowBlueU
    del redThreImgU, greenThreImgU, blueThreImgU
    del shadowRedU, shadowGreenU, shadowBlueU

    greenImg3 = diffImg > 0.0
    greenImg4 = green_red_Diff > 0
    threshold = graythresh(ExG, 0.1)

    if threshold > 0.1:
        threshold = 0.1
    elif threshold < 0.05:
        threshold = 0.05

    greenImg2 = ExG > threshold
    greenImgShadow2 = ExG > 0.05
    greenImg = greenImg1 * greenImg2 + greenImgShadow2 * greenImgShadow1



    del ExG, green_blue_Diff, green_red_Diff
    del greenImgShadow1, greenImgShadow2

    # calculate the percentage of the green vegetation
    greenPxlNum = len(np.where(greenImg != 0)[0])
    greenPercent = greenPxlNum / (400.0 * 400) * 100
    del greenImg1, greenImg2
    del greenImg3, greenImg4

    return greenPercent

# Path to the image
Img = r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\POI268\input\P3_B2_268_120.jpg'
# Load the original image
original_image = Image.open(Img)
# use the meanshift segmentation algorithm to segment the original GSV image
(segmented_image, labels_image, number_regions) = pms.segment(original_image,
                                                              spatial_radius=6,
                                                              range_radius=7,
                                                              min_density=40)
I = segmented_image / 255.0
red = I[:, :, 0]
green = I[:, :, 1]
blue = I[:, :, 2]
# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
# Plot red band
axes[0].imshow(red, cmap='Reds')  # Use Reds colormap for the red band
axes[0].set_title('red band',fontsize=22)
axes[0].axis('off')
# Plot green band
axes[1].imshow(green, cmap='Greens')  # Use Greens colormap for the green band
axes[1].set_title('green band',fontsize=22)
axes[1].axis('off')
# Plot blue band
axes[2].imshow(blue, cmap='Blues')  # Use Blues colormap for the blue band
axes[2].set_title('blue band',fontsize=22)
axes[2].axis('off')
# Adjust layout
plt.tight_layout()
# Save the figure
plt.savefig(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\POI268\output\result_diagrams.png')
# Show the plots
plt.show()


import numpy as np
Img_1 = np.array(original_image)
# Use the VegetationClassification function to get the segmented vegetation mask
vegetation_mask = VegetationClassification(Img_1)

# Plot original GSV image
plt.figure(figsize=(8, 8))
plt.imshow(Img_1)
plt.title('Original GSV Image')
plt.axis('off')
plt.show()

# Plot segmented vegetation area on the original image
plt.figure(figsize=(8, 8))
plt.imshow(greenImg, cmap='Greens')
plt.title('Segmented Vegetation Area',fontsize= 22)
plt.axis('off')
plt.savefig(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\POI268\output\result_green.png')
plt.show()


"""
-----------------对sementations作GVI
"""
def calculate_ratio(color,img):
    # Here, you define your target color as
    # a tuple of three values: RGB
    # You define an interval that covers the values
    # in the tuple and are below and above them by 20
    diff = 0
    # Be aware that opencv loads image in BGR format,
    # that's why the color values have been adjusted here:
    boundaries = [([color[2], color[1]-diff, color[0]-diff],
               [color[2]+diff, color[1]+diff, color[0]+diff])]
    # Scale your BIG image into a small one:
    scalePercent = 1
    # Calculate the new dimensions
    width = int(img.shape[1] * scalePercent)
    height = int(img.shape[0] * scalePercent)
    newSize = (width, height)
    # Resize the image:
    seg_image = cv2.resize(img, newSize, None, None, None, cv2.INTER_AREA)

    # check out the image resized:
    #cv2.imshow("seg_image resized", seg_image)
    #cv2.waitKey(0)

    # for each range in your boundary list:
    for (lower, upper) in boundaries:
        # You get the lower and upper part of the interval:
        lower = np.array(lower)
        upper = np.array(upper)

        # cv2.inRange is used to binarize (i.e., render in white/black) an image
        # All the pixels that fall inside your interval [lower, uipper] will be white
        # All the pixels that do not fall inside this interval will
        # be rendered in black, for all three channels:
        mask = cv2.inRange(img, lower, upper)
        # Check out the binary mask:
        #cv2.imshow("binary mask", mask)
        #cv2.waitKey(0)

        # Now, you AND the mask and the input image
        # All the pixels that are white in the mask will
        # survive the AND operation, all the black pixels
        # will remain black
        output = cv2.bitwise_and(img, img, mask=mask)

        # Check out the ANDed mask:
        #cv2.imshow("ANDed mask", output)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # You can use the mask to count the number of white pixels.
        # Remember that the white pixels in the mask are those that
        # fall in your defined range, that is, every white pixel corresponds
        # to a green pixel. Divide by the image size and you got the
        # percentage of green pixels in the original image:
        ratio_green = cv2.countNonZero(mask)/(seg_image.size/3)
        # This is the color percent calculation, considering the resize I did earlier.
        colorPercent = (ratio_green * 100) / scalePercent

        # Print the color percent, use 2 figures past the decimal point
        print('green pixel percentage:', np.round(colorPercent, 2))
        return(colorPercent)

# 测试一下能不能行 - 能行
a = calculate_ratio(green,test_seg)
# numpy's hstack is used to stack two images horizontally,
# so you see the various images generated in one figure:
#cv2.imshow("images", np.hstack([seg_image, output]))
#cv2.waitKey(0)

Root = r'D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test_results'
outputPath = r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG'

def GVI_SEG_final(GSV_file):
    GVI_df = pd.DataFrame(columns=['part_id', 'batch_id', 'object_id', 'head', 'GVI'])
    GSVsegRoot = os.path.join(Root,GSV_file)
    for filename in os.listdir(GSVsegRoot):
        filepath = os.path.join(GSVsegRoot, filename)
        img = cv2.imread(filepath,cv2.COLOR_BGR2HSV)
        print(filepath)
        # checking if it is a file
        #if os.path.isfile(img):
        percent_green = calculate_ratio(green,img)
        percent_light = calculate_ratio(light_green,img)
        tot_percent = percent_green+percent_light

        part_id, batch_id, object_id, head = filename.replace('.', '_').split('_')[:-1]
        row = {'part_id': part_id, 'batch_id': batch_id, 'object_id': object_id, 'head': head,
               'GVI': tot_percent}
        row = pd.DataFrame(row, index=[0])
        GVI_df = pd.concat([GVI_df, row], axis=0)

    part_id_final = part_id
    # 然后计算新的字段'total GVI'
    tot_GVI_df = GVI_df.groupby(by=['object_id'],as_index=False).agg({'object_id':'first',
                                              'part_id':'count', # 这里返回的是每个点的head的图片数量
                                              'batch_id':'first',
                                              'GVI':'sum'}) # 计算相同object_id的图片数量
    tot_GVI_df['totGVI'] = tot_GVI_df.apply(lambda x: x.GVI/x.part_id,axis=1) # 防止没有6张
    csv_name = part_id_final  + '_SEG.csv' # 这个脚本直接算整个part
    tot_GVI_df['object_id'] = tot_GVI_df['object_id'].astype('int64')
    tot_GVI_df=tot_GVI_df.sort_values('object_id')
    tot_GVI_df.to_csv(os.path.join(outputPath,csv_name),encoding='utf-8')

GVI_SEG_final('P1') # 'P1' - GSV_file
GVI_SEG_final('P2')
GVI_SEG_final('P3')
GVI_SEG_final('P4')
GVI_SEG_final('P5')
GVI_SEG_final('P6')
GVI_SEG_final('P7')

"""
lower = np.array([40,192,142])
upper = np.array([40,192,142])
# 原图
hsv_green = cv2.cvtColor(seg_image,cv2.COLOR_BGR2HSV)
print(hsv_green) #[40,192,142] R:G:B
cv2.imshow("hsv_green", hsv_green)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Define threshold color range to filter
# 低于这个值的值 -> 0
# 高于这个值的值 -> 0
# 两者之间的 -> 255
mask = cv2.inRange(hsv_green, lower, upper)
cv2.imshow("mask", mask)
cv2.waitKey(1)
cv2.destroyAllWindows()

# Bitwise-AND mask and original image
res = cv2.bitwise_and(seg_image, seg_image, mask=mask)
ratio = cv2.countNonZero(mask)/(seg_image.size/3)
print('pixel percentage:', np.round(ratio*100, 2))
"""
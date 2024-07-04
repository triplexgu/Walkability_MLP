globals().clear()

import cv2
import pymeanshift as pms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
# -------------方法1：使用pms,根据光学反射原理算
"""
original_image = cv2.imread(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P1\P1_B1_15_72.925.jpg',cv2.IMREAD_UNCHANGED)
(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6,range_radius=4.5, min_density=50)
# 看一下颜色
#cv2.imshow('blue',segmented_image[:,:,0])
cv2.imshow('blue',segmented_image)
cv2.waitKey(1)
cv2.destroyAllWindows()

"""
# 迭代所有文件夹中的图片
# 先按照batch来做
"""
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
    np.seterr(divide='ignore', invalid='ignore')

    #path = cv2.imread(Img,cv2.IMREAD_UNCHANGED)
    # use the meanshift segmentation algorithm to segment the original GSV image
    (segmented_image, labels_image, number_regions) = pms.segment(Img,spatial_radius=6,range_radius=7, min_density=40)

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

def get_names(filepath):
    # 返回的是路径下图片的object_id和角度
    # file name with extension
    file_name = os.path.basename(filepath)
    # file name without extension
    part_num,batch_num,id,head = os.path.splitext(file_name)[0].split('_')
    return part_num,int(id),float(head)

"""
读取GSV file作为input，然后遍历所有img，读取gvi by调用<VegetationClassification>
"""
outputPath = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift"

def get_split_GVI_pymeanshift(part_num,Root,output_Root):
    split_df = pd.DataFrame(columns=['object_id',
                                     'top_right', 'top_middle', 'top_left',
                                     'middle_right', 'middle_middle', 'middle_left',
                                     'low_right', 'low_middle', 'low_left',
                                     'top', 'middle', 'low'])
    # 遍历一个文件夹内所有的文档（注意：这里需要的是segmentation之后的照片！！！！！）
    for filename in os.listdir(Root):
        f = os.path.join(Root, filename)
        print(f)
        part_num,id, head = get_names(f)
        img = cv2.imread(f)
        im = cv2.resize(img, (900, 900))
        # 应该存3个dictionaries！
        top_right = im[0:300, 0:300, :]
        top_middle = im[0:300, 300:600, :]
        top_left = im[0:300, 600:900, :]

        middle_right = im[300:600, 0:300, :]
        middle_middle = im[300:600, 300:600, :]
        middle_left = im[300:600, 600:900, :]

        low_right = im[600:900, 0:300, :]
        low_middle = im[600:900, 300:600, :]
        low_left = im[600:900, 600:900, :]

        # 算每个角度的tot GVI
        top_right_GVI = VegetationClassification(top_right)
        top_middle_GVI = VegetationClassification(top_middle)
        top_left_GVI = VegetationClassification(top_left)

        middle_right_GVI = VegetationClassification(middle_right)
        middle_middle_GVI = VegetationClassification(middle_middle)
        middle_left_GVI = VegetationClassification(middle_left)

        low_right_GVI = VegetationClassification(low_right)
        low_middle_GVI = VegetationClassification(low_middle)
        low_left_GVI = VegetationClassification(low_left)

        top = top_right_GVI+top_middle_GVI+top_left_GVI
        middle = middle_right_GVI+middle_left_GVI
        low = low_right_GVI+low_middle_GVI+low_left_GVI

        #把所有的内容存在df中
        row = {'object_id':id,
             'top_right':top_right_GVI,'top_middle':top_middle_GVI,'top_left':top_left_GVI,
             'middle_right':middle_right_GVI,'middle_middle':middle_middle_GVI,'middle_left':low_right_GVI,
             'low_right':low_right_GVI,'low_middle':low_middle_GVI,'low_left':low_left_GVI,
            'top':top,'middle':middle,'low':low}
        row = pd.DataFrame(row, index=[0])
        split_df = pd.concat([split_df, row], axis=0)

    df_name = '{}_GVI_pymeanshift.csv'.format(part_num)
    output_Root_full = os.path.join(output_Root,df_name)
    split_df.to_csv(output_Root_full)


# P1
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P1"
get_split_GVI_pymeanshift(1,GSVimagesRoot,outputPath)

# P2
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P2"
get_split_GVI_pymeanshift(2,GSVimagesRoot,outputPath)

# P3
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P3"
get_split_GVI_pymeanshift(3,GSVimagesRoot,outputPath)

# P4
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P4"
get_split_GVI_pymeanshift(4,GSVimagesRoot,outputPath)

# P5
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P5"
get_split_GVI_pymeanshift(5,GSVimagesRoot,outputPath)

# P6
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P6"
get_split_GVI_pymeanshift(6,GSVimagesRoot,outputPath)

# P7
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs\P7"
get_split_GVI_pymeanshift(7,GSVimagesRoot,outputPath)
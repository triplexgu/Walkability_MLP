globals().clear()

import cv2
import pymeanshift as pms
import numpy as np
import matplotlib.pyplot as plt
"""
# -------------方法1：使用pms,根据光学反射原理算
"""
original_image = cv2.imread(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs\P1_B1\P1_B1_12_180.jpg',cv2.IMREAD_UNCHANGED)
(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6,range_radius=4.5, min_density=50)
# 看一下颜色
#cv2.imshow('blue',segmented_image[:,:,0])
cv2.imshow('blue',labels_image)
cv2.waitKey(1)
cv2.destroyAllWindows()

"""
# 迭代所有文件夹中的图片
# 先按照batch来做
"""
# import required module
import os

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

"""
读取GSV file作为input，然后遍历所有img，读取gvi by调用<VegetationClassification>
"""
def GreenViewComputing_ogr_6Horizon(GSVimagesRoot,outputPath):
    import os
    import pandas as pd
    #GSVimagesRoot = r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs\test'
    # 设置两个初始值
    GVI_df = pd.DataFrame(columns=['part_id','batch_id','object_id','head','GVI'])
    greenPercent = 0.0
    for filename in os.listdir(GSVimagesRoot):
        img = os.path.join(GSVimagesRoot, filename)
        # checking if it is a file
        if os.path.isfile(img):
            im = cv2.imread(img) #cv2.IMREAD_UNCHANGED
            percent = VegetationClassification(im)

            part_id,batch_id,object_id,head = filename.replace('.', '_').split('_')[:-1]
            row = {'part_id': part_id, 'batch_id': batch_id, 'object_id': object_id, 'head': head,
                   'GVI':percent}
            row = pd.DataFrame(row, index=[0])
            GVI_df = pd.concat([GVI_df, row], axis=0)

    # 然后计算新的字段'total GVI'
    tot_GVI_df = GVI_df.groupby(by=['object_id'],as_index=False).agg({'object_id':'first',
                                              'part_id':'count',
                                              'batch_id':'first',
                                              'GVI':'sum'}) # 计算相同object_id的图片数量
    tot_GVI_df['totGVI'] = tot_GVI_df.apply(lambda x: x.GVI/x.part_id,axis=1)
    csv_name = part_id + batch_id + '.csv'
    tot_GVI_df['object_id'] = tot_GVI_df['object_id'].astype('int64')
    tot_GVI_df=tot_GVI_df.sort_values('object_id')
    tot_GVI_df.to_csv(os.path.join(outputPath,csv_name),encoding='utf-8')
    return(tot_GVI_df)

# P1 ✔
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P1"
# P2: 6.40 ✔
# 包含了P2+P3第一部分
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P2"
# P3：7.07 - 7.18 ✔
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P3"
# P4：7.18 - 7.38 ✔
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P4"
# P5：13.09 - 13.28 ✔
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P5"
# P6: 13.31 - 13.48 ✔
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P6"
# P7：13.48 - 14.11 ✔
GSVimagesRoot = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\test\P7"

outputPath = r"D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift"
#greenmonth = ['05', '06', '07', '08', '09', '10']

GreenViewComputing_ogr_6Horizon(GSVimagesRoot=GSVimagesRoot,outputPath=outputPath)
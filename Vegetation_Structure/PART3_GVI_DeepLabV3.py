"""
#-----------PART 3 计算GVI-----------------------
"""
globals().clear()

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
import numpy as np

# 计算GVI
def calculate_ratio(color,img):
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

    # for each range in your boundary list:
    for (lower, upper) in boundaries:
        # You get the lower and upper part of the interval:
        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(img, lower, upper)

        output = cv2.bitwise_and(img, img, mask=mask)

        ratio_green = cv2.countNonZero(mask)/(seg_image.size/3)
        # This is the color percent calculation, considering the resize I did earlier.
        colorPercent = (ratio_green * 100) / scalePercent

        print('green pixel percentage:', np.round(colorPercent, 2))
        return(colorPercent)

def calculate_ratio_tot(img):
    green = [107, 142, 35]  # tree:red green blue
    light_green = [152, 251, 152]  # bush
    green = calculate_ratio(green,img)
    light_green = calculate_ratio(light_green,img)
    tot = green+light_green
    return tot

def get_names(filepath):
    # 返回的是路径下图片的object_id和角度
    # file name with extension
    file_name = os.path.basename(filepath)
    # file name without extension
    part_num,batch_num,id,head = os.path.splitext(file_name)[0].split('_')
    return part_num,int(id),float(head)

def get_split_GVI(part_num):
    Root = r"D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\new_vs_output"
    output_Root = r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split'
    Root = os.path.join(Root,part_num)

    split_df = pd.DataFrame(columns=['object_id',
                                     'top_right', 'top_middle', 'top_left',
                                     'mid_up_right', 'mid_up_middle', 'mid_up_left',
                                     'mid_down_right', 'mid_down_middle', 'mid_down_left',
                                     'low_right', 'low_middle', 'low_left',
                                     'top', 'low'])
    # 遍历一个文件夹内所有的文档（注意：这里需要的是segmentation之后的照片！！！！！）
    for filename in os.listdir(Root):
        f = os.path.join(Root, filename)
        print(f)

        part_num,id, head = get_names(f)
        img = cv2.imread(f)
        im = cv2.resize(img, (900, 900))
        # 应该存3个dictionaries！
        top_right = im[0:225, 0:300, :]
        top_middle = im[0:225, 300:600, :]
        top_left = im[0:225, 600:900, :]

        mid_up_right = im[225:450, 0:300, :]
        mid_up_middle = im[225:450, 300:600, :]
        mid_up_left = im[225:450, 600:900, :]

        mid_down_right = im[450:675, 0:300, :]
        mid_down_middle = im[450:675, 300:600, :]
        mid_down_left = im[450:675, 600:900, :]

        low_right = im[675:900, 0:300, :]
        low_middle = im[675:900, 300:600, :]
        low_left = im[675:900, 600:900, :]

        # 算每个角度的tot GVI
        top_right_GVI = calculate_ratio_tot(top_right)
        top_middle_GVI = calculate_ratio_tot(top_middle)
        top_left_GVI = calculate_ratio_tot(top_left)

        mid_up_right_GVI = calculate_ratio_tot(mid_up_right)
        mid_up_middle_GVI = calculate_ratio_tot(mid_up_middle)
        mid_up_left_GVI = calculate_ratio_tot(mid_up_left)

        mid_down_right_GVI = calculate_ratio_tot(mid_down_right)
        mid_down_middle_GVI = calculate_ratio_tot(mid_down_middle)
        mid_down_left_GVI = calculate_ratio_tot(mid_down_left)

        low_right_GVI = calculate_ratio_tot(low_right)
        low_middle_GVI = calculate_ratio_tot(low_middle)
        low_left_GVI = calculate_ratio_tot(low_left)

        # 等一下，这边需要除以5吗
        top = (top_right_GVI+top_middle_GVI+top_left_GVI + mid_up_right_GVI+mid_up_left_GVI)/5
        low = (low_right_GVI+low_middle_GVI+low_left_GVI + mid_down_right_GVI+mid_down_left_GVI)/5

        #把所有的内容存在df中
        row = {
            'object_id': id,
            'top_right': top_right_GVI,
            'top_middle': top_middle_GVI,
            'top_left': top_left_GVI,
            'mid_up_right': mid_up_right_GVI,
            "mid_up_middle":mid_up_middle_GVI,
            'mid_up_left': mid_up_left_GVI,
            'mid_down_right': mid_down_right_GVI,
            "mid_down_middle":mid_down_middle_GVI,
            'mid_down_left': mid_down_left_GVI,
            'low_right': low_right_GVI,
            'low_middle': low_middle_GVI,
            'low_left': low_left_GVI,
            'top': top,
            'low': low
        }
        # Create a DataFrame with a single row
        row = pd.DataFrame([row])
        split_df = pd.concat([split_df, row], axis=0)

    df_name = '{}_GVI_DeepLabV3.csv'.format(part_num)
    df_name = 'test.csv'
    output_Root = 'D:\WUR\master\MLP\master thesis\data\数据汇总\图\Methodology'
    output_Root_full = os.path.join(output_Root,df_name)
    split_df.to_csv(output_Root_full)

# 使用DeepLabV3+来获得每个split的GVI
get_split_GVI('P1')
get_split_GVI('P2')
get_split_GVI('P3')
get_split_GVI('P4')
get_split_GVI('P5')
get_split_GVI('P6')
get_split_GVI('P7')

# Your existing code
f = r'D:\WUR\master\MLP\master thesis\data\DLmodel\deeplabv3plus-pytorch-master\new_vs_output\P5\P5_B1_423_89.443.png'
part_num, id, head = get_names(f)
img = cv2.imread(f)
im = cv2.resize(img, (900, 900))
plt.imshow(im)
plt.show()

# Subplots
fig, axs = plt.subplots(4, 3, figsize=(12, 9))

# Top row
axs[0, 0].imshow(im[0:225, 0:300, :])
axs[0, 1].imshow(im[0:225, 300:600, :])
axs[0, 2].imshow(im[0:225, 600:900, :])

# TOP2 row
axs[1, 0].imshow(im[225:450, 0:300, :])
axs[1, 1].imshow(im[225:450, 300:600, :])
axs[1, 2].imshow(im[225:450, 600:900, :])

# BOTTOM1 row
axs[2, 0].imshow(im[450:675, 0:300, :])
axs[2, 1].imshow(im[450:675, 300:600, :])
axs[2, 2].imshow(im[450:675, 600:900, :])

# BOTTOM2 subplots
axs[3, 0].imshow(im[675:900, 0:300, :])
axs[3, 1].imshow(im[675:900, 300:600, :])
axs[3, 2].imshow(im[675:900, 600:900, :])

# Hide the axes labels and ticks
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Adjust layout to prevent clipping of subplot titles
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig(r'D:\WUR\master\MLP\master thesis\data\数据汇总\图\Methodology\vegetationStructure.png')
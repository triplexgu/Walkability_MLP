globals().clear()


import os
import os.path
import json
import re
from typing import List, Optional
import requests
from pydantic import BaseModel
from requests.models import Response
import pandas as pd

"""
------------------------PART 1-----------------------------
functions setting
"""
class Panorama(BaseModel):
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float]
    roll: Optional[float]
    date: Optional[str]

def coordns(poi_test): # 这里的输入不一样！
    """
    从csv中读取lon/lat
    """
    lonlst = []
    latlst = []
    object_id_lst = []
    for index, row in poi_test.iterrows():
        lon = row['X']
        lat = row['Y']
        object_id = row['OBJECTID']
        lonlst.append(lon)
        latlst.append(lat)
        object_id_lst.append(object_id)
    return lonlst,latlst,object_id_lst

def make_search_url(lat: float, lon: float) -> str:
    """
    Builds the URL of the script on Google's servers that returns the closest
    panoramas (ids) to a give GPS coordinate.
    """
    url = (
        "https://maps.googleapis.com/maps/api/js/"
        "GeoPhotoService.SingleImageSearch"
        "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10"
        "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4"
        "!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=callbackfunc"
    )
    return url.format(lat, lon)

def search_request(lat: float, lon: float) -> Response:
    """
    Gets the response of the script on Google's servers that returns the
    closest panoramas (ids) to a give GPS coordinate.
    返回的是response
    """
    url = make_search_url(lat, lon)
    return requests.get(url)

def extract_panoramas_updated(text: str) -> List[Panorama]:
    b = []
    """
    Given a valid response from the panoids endpoint, return a list of all the
    panoids
    """
    # The response is actually javascript code. It's a function with a single
    # input which is a huge deeply nested array of items.
    blob = re.findall(r"callbackfunc\( (.*) \)$", text)[0]
    data = json.loads(blob)

    if data == [[5, "generic", "Search returned no images."]]:
        return []

    subset = data[1][5][0]

    raw_panos = subset[3][0]

    if len(subset) < 9 or subset[8] is None:
        raw_dates = []
    else:
        raw_dates = subset[8]

    # For some reason, dates do not include a date for each panorama.
    # the n dates match the last n panos. Here we flip the arrays
    # so that the 0th pano aligns with the 0th date.
    raw_panos = raw_panos[::-1]
    raw_dates = raw_dates[::-1]

    dates = [f"{d[1][0]}-{d[1][1]:02d}" for d in raw_dates]

    for i, pano in enumerate(raw_panos):
        pano_id=pano[0][1]
        lat=pano[2][0][2]
        lon=pano[2][0][3]
        heading=pano[2][2][0]
        pitch=pano[2][2][1] if len(pano[2][2]) >= 2 else None
        roll=pano[2][2][2] if len(pano[2][2]) >= 3 else None
        date=dates[i] if i < len(dates) else None
        a = (pano_id,lat,lon,heading,pitch,roll,date)
        b.append(a)
    return b

# 不要用chunksize，还是乖乖设置一个function来手动把df分成好几块吧
def chuncker(df,num):
    # specify number of rows in each chunk
    n = num
    # split DataFrame into chunks
    list_df = [df[i:i + n] for i in range(0, len(df), n)]
    return list_df

def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None

def batch_process(batch_df):
    df_name = get_variable_name(batch_df)
    df_list = chuncker(batch_df,50)
    n = 0
    for subdf in df_list:
        n = n+1
        batch_name = '%s_Batch%d_PanoID_collection.text' % (df_name,n)
        output_file = os.path.join(output_root, batch_name)
        df = subdf.copy()
        latlst, lonlst, object_id_lst = coordns(subdf) # 这里记得改！
        with open(output_file, 'w') as panoInfoText:
            for i in range(len(latlst)):
                batchID = n  # 这里需要注意一下batch的序列
                lat_x = float(latlst[i])
                lon_y = float(lonlst[i])
                object_id = object_id_lst[i] # # 这里控制的是csv中点的序号，确保是同一个点
                # lon - 25.004901 lat - 60.263337 测试用
                qst = search_request(lon=lat_x, lat=lon_y) # 调用功能<search_request>
                test = extract_panoramas_updated(qst.text) # 调用功能<extract_panoramas_updated>
                #print(test)
                # 把这个点上的信息从tuple摘出来，写进文本
                for j in range(len(test)):
                    panoLat = test[j][1]
                    panoLon = test[j][2]
                    panoId = test[j][0] # 在google api中定位用的PanoID
                    panoHeading = test[j][3]
                    panoPitch = test[j][4]
                    panoDate = test[j][-1]
                    print('The coordinate (%s,%s), panoId is: %s, panoDate is: %s' % (panoLon, panoLat, panoId, panoDate))
                    lineTxt = 'batchID: %s object_id: %s panoID: %s panoDate: %s longitude: %s latitude: %s Heading: %s Pitch: %s \n' % (
                        batchID, object_id, panoId, panoDate, panoLon, panoLat, panoHeading, panoPitch)
                    panoInfoText.write(lineTxt)
        panoInfoText.close()

"""
------------------------PART 1-----------------------------
真正的操作部分
"""
pois = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\POI_25m_withAngles.csv')
pois_p1,pois_p2,pois_p3,pois_p4,pois_p5,pois_p6,pois_p7 = chuncker(pois,100)

output_root = r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_pano_info'

# 对所有的点进行一个pano的metadata提取
batch_process(pois_p1)
batch_process(pois_p2)
batch_process(pois_p3)
batch_process(pois_p4)
batch_process(pois_p5)
batch_process(pois_p6)
batch_process(pois_p7)


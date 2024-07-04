globals().clear()

import geopandas as gpd
import requests
import csv
import pandas as pd

pois = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\VegetationStructure_POI_50.geojson').\
    drop(columns=['geometry','FIRST_naam', 'FIRST_MAX_name', 'SUM_SUM_Shape_Length', 'ORIG_FID', 'OBJECTID_1', 'FIRST_OBJECTID'])
# 'OBJECTID', 'ORIG_FID', 'OBJECTID_1', 'FIRST_OBJECTID', 'FIRST_PC4',
#        'FIRST_naam', 'FIRST_MAX_name', 'SUM_SUM_Shape_Length', 'Directions',
#        'X', 'Y', 'geometry'

meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
api_key = 'AIzaSyC-mpY1oGP7OCofxP3VzeNJTXqfXFV1aVI'

filename = pois.iloc[0:15,]
for idx,row in filename.iterrows():
    id = int(row.OBJECTID)
    lat = row.Y
    lon = row.X
    heading = row.Directions
    location = str(lat) + ',' + str(lon)
    pic_params = {'key': api_key,
                  'location': location,
                  'size': "400x400",
                  'heading': str(heading),
                  'fov': '60',  # zoom-in level
                  'pitch': '0'}
    meta_params = {'key': api_key,
                  'location': location,
                  'size': "400x400",
                  'heading': str(heading),
                  'fov': '60',  # zoom-in level
                  'pitch': '0'}
    # obtain the metadata of the request (this is free)
    meta_response = requests.get(meta_base, params=meta_params)
    print(meta_response.json())

    pic_response = requests.get(pic_base, params=pic_params)
    for key, value in pic_response.headers.items():
      #print(f"{key}: {value}")
      #print(pic_response.ok)
      import os
      Root = r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\imgs'
      file = str(id)+'_'+str(heading)+'.jpg'
      final_file = os.path.join(Root,file)
      with open(final_file, 'wb') as f: #coordination+pitch+heading
        f.write(pic_response.content)

# remember to close the response connection to the API
pic_response.close()


"""
------------------------PART 2-----------------------------
根据晒出来的metadate下载GSV（greenmonth）
"""
globals().clear()

import csv
import os, os.path
import json
import re
from typing import List, Optional
import requests
from pydantic import BaseModel
from requests.models import Response
import pandas as pd

GSVinfoRoot = r"D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\GSV"
GVI_root = r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs'
# name of the output image file
greenmonth = ['05', '06', '07', '08', '09','10']  # months used

def GVS_image_downloader(GSVinfoRoot, filename,greenmonth,part_num):
    """
    This function is used to download the GSV from the information provide
    by the gsv info txt, and save the result to a shapefile
    """
    final_df = pd.DataFrame(columns=['part_num','batch_id','object_id','panoID',
                                     'Year','Month','Lon','Lat','Head','Pitch'])
    #allTxtFiles = os.listdir(GSVinfoRoot,filename)
    textpath = os.path.join(GSVinfoRoot, filename)

    #for txtfile in allTxtFiles:
        #if txtfile.endswith('.text'):
            #path = os.path.join(GSVinfoRoot,txtfile)
    lines = open(textpath, "r")
    # loop all lines in the txt files
    for line in lines:
        print(line)
        metadata = line.split(" ")
        if metadata[7] != 'None':
            Year,Month = metadata[7].split('-')
            if Month in greenmonth:
                part_num = part_num
                batch_id = metadata[1]
                object_id = metadata[3]
                panoID = metadata[5]

                lon = metadata[9]
                lat = metadata[11]
                head = metadata[13]
                pitch = metadata[15]

                row = {'part_num':part_num,'batch_id':batch_id,'object_id':object_id,'panoID':panoID,
                       'Year':Year,'Month':Month,
                       'Lon':lon,'Lat':lat,'Head':head,'Pitch':pitch}
                row = pd.DataFrame(row,index=[0])
                final_df = pd.concat([final_df,row],axis=0)

            # 这时候把所有能搜寻到的，在green month里的点都抽出来放在csv中进行整理
            csv_name = 'csv_' + filename.split('.')[0]+'.csv'
            final_df.to_csv(os.path.join(GVI_root, csv_name),encoding='utf-8')

            headingArr = range(0,360,60) #自己加的
            for heading in headingArr:
                print("Heading is: ", heading)

                for index, row in final_df.iterrows():
                    panoID = row.panoID
                    new_lat = row.Lat
                    new_lon = row.Lon
                    object_id = row.object_id
                    batch_id = row.batch_id

                    # construct the orginal URL
                    URL = "http://maps.googleapis.com/maps/api/streetview?size=400x400&pano=%s&fov=60&heading=%s&sensor=false&key=AIzaSyC-mpY1oGP7OCofxP3VzeNJTXqfXFV1aVI" % (
                    panoID, heading)
                    # add the decoded google security key into the URL
                    #URL = sign_url(URL, 'put your signing signature here')

                    # Download and save the GSV images
                    #name = os.path.join(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs','GSV_{}_{}.jpg'.format(new_lat, new_lon, heading))
                    #name = os.path.join(
                        #r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs',
                        #'P_{}_B_{}_{}.jpg'.format(batch_id,object_id, heading))
                    name = os.path.join(
                        r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs',
                        'P{}_B{}_{}_{}.jpg'.format(part_num,batch_id, object_id, heading))
                    try:
                        #name = os.path.join(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs','GSV_%d_%d.jpg'%(id, heading))
                        if not os.path.exists(name):
                            response = requests.get(URL)
                            with open(name, 'wb') as f:
                                f.write(response.content)
                    except:
                        print('downloading ' + name + ' failed ')
                        #print('downloading failed ')
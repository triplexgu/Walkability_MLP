"""
------------------------PART 2-----------------------------
根据晒出来的metadate下载GSV（greenmonth）
"""
globals().clear()

import os, os.path
import requests
import pandas as pd

def chuncker(df,num):
    # specify number of rows in each chunk
    n = num
    # split DataFrame into chunks
    list_df = [df[i:i + n] for i in range(0, len(df), n)]
    return list_df

GSVinfoRoot = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_pano_info"
GVI_root = r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_imgs'

pois = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\POI_25m_withAngles.csv')
pois_p1,pois_p2,pois_p3,pois_p4,pois_p5,pois_p6,pois_p7 = chuncker(pois,100)

# name of the output image file
greenmonth = ['05', '06', '07', '08', '09','10']  # months used

def GVS_image_downloader(GSVinfoRoot, filename,greenmonth,part_num,poi_df):
    final_df = pd.DataFrame(columns=['part_num','batch_id','object_id','panoID',
                                     'Year','Month','Lon','Lat','Head','Pitch'])
    textpath = os.path.join(GSVinfoRoot, filename)

    #textpath = r"D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_pano_info\pois_p7_Batch1_PanoID_collection.text"

    lines = open(textpath, "r")
    #part_num = 7
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
            #csv_name = 'csv_' + filename.split('.')[0]+'.csv'
            #final_df.to_csv(os.path.join(GVI_root, csv_name),encoding='utf-8')

        for index,row in final_df.iterrows():
            panoID = row.panoID
            new_lat = row.Lat
            new_lon = row.Lon
            object_id = int(row.object_id)
            #object_id.append(int(row.object_id))
            batch_id = row.batch_id

            head = list(poi_df['Directions'][poi_df['TARGET_FID'] == object_id])

            if len(head) != 0:
                # construct the orginal URL
                URL = "http://maps.googleapis.com/maps/api/streetview?size=900x900&pano=%s&fov=60&heading=%s&sensor=false&key=AIzaSyC-mpY1oGP7OCofxP3VzeNJTXqfXFV1aVI" % (
                panoID, head[0])

                # Download and save the GSV images
                name = os.path.join(
                    GVI_root,
                    'P{}_B{}_{}_{}.jpg'.format(part_num,batch_id, object_id, head[0]))
                try:
                    #name = os.path.join(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\imgs','GSV_%d_%d.jpg'%(id, heading))
                    if not os.path.exists(name):
                        response = requests.get(URL)
                        with open(name, 'wb') as f:
                            f.write(response.content)
                except:
                    print('downloading ' + name + ' failed ')
                    #print('downloading failed ')

# poi_p1 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p1_Batch1_PanoID_collection.text',greenmonth,1,pois_p1)
GVS_image_downloader(GSVinfoRoot,'pois_p1_Batch2_PanoID_collection.text',greenmonth,1,pois_p1)

# poi_p2 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p2_Batch1_PanoID_collection.text',greenmonth,2,pois_p2)
GVS_image_downloader(GSVinfoRoot,'pois_p2_Batch2_PanoID_collection.text',greenmonth,2,pois_p2)

# poi_p3 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p3_Batch1_PanoID_collection.text',greenmonth,3,pois_p3)
GVS_image_downloader(GSVinfoRoot,'pois_p3_Batch2_PanoID_collection.text',greenmonth,3,pois_p3)

# poi_p4 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p4_Batch1_PanoID_collection.text',greenmonth,4,pois_p4)
GVS_image_downloader(GSVinfoRoot,'pois_p4_Batch2_PanoID_collection.text',greenmonth,4,pois_p4)

# poi_p5 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p5_Batch1_PanoID_collection.text',greenmonth,5,pois_p5) # ✔
GVS_image_downloader(GSVinfoRoot,'pois_p5_Batch2_PanoID_collection.text',greenmonth,5,pois_p5)

# poi_p6 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p6_Batch1_PanoID_collection.text',greenmonth,6,pois_p6)
GVS_image_downloader(GSVinfoRoot,'pois_p6_Batch2_PanoID_collection.text',greenmonth,6,pois_p6) # ✔

# poi_p7 ✔
GVS_image_downloader(GSVinfoRoot,'pois_p7_Batch1_PanoID_collection.text',greenmonth,7,pois_p7)
GVS_image_downloader(GSVinfoRoot,'pois_p7_Batch2_PanoID_collection.text',greenmonth,7,pois_p7) # ✔
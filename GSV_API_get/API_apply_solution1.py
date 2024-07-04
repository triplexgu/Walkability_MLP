# Import google_streetview for the api module
#在terminal！！！！
#pip install google_streetview
import google_streetview.api
# 但是api不好用，弃用

import requests
import csv
import pandas as pd

filename = 'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Sampling_coords.csv'
pois = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Sampling_coords.csv')

meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
api_key = 'AIzaSyC-mpY1oGP7OCofxP3VzeNJTXqfXFV1aVI'

with open(filename) as f:
  reader = csv.reader(f)
  next(reader) #跳过heading
  latitudes = []
  longitudes = []
  for row in reader:
    latitude = str(row[-1]) #Y - 五十多的
    latitudes.append(latitude)
    longitude = str(row[-2]) #X - 四点几的
    longitudes.append(longitude)
    num = 3
    #num = len(longitudes) #这里是设置每个点的数量

    #location = '52.34589,4.888348'
    #location = '52.34547,4.899404'

  #for i in range(len(latitudes)): #iterate all locations
  for i in [5,30,45,80]:  # iterate all locations
    location = latitudes[i]+','+longitudes[i]
    for j in range(0,360,60): #heading:6 horizontal angles；0-》NORTH

        pic_params = {'key': api_key,
              'location': location,
              'size': "512x512",
              'heading':str(j),
              'fov':'60', # zoom-in level
              'pitch':'0'}

        meta_params = {'key': api_key,
              'location': location,
              'size': "512x512",
              'heading':str(j),
              'fov':'60',
              'pitch':'0'}
        # obtain the metadata of the request (this is free)
        meta_response = requests.get(meta_base, params=meta_params)
        print(meta_response.json())

        pic_response = requests.get(pic_base, params=pic_params)
        for key, value in pic_response.headers.items():
          print(f"{key}: {value}")
          print(pic_response.ok)
          with open('D:\WUR\master\MLP\master thesis\data\GSV_API_GET\GSVs\测试'+location+'_'+'angle0'+'_'+str(j)+'.jpg', 'wb') as f: #coordination+pitch+heading
            f.write(pic_response.content)

# remember to close the response connection to the API
pic_response.close()

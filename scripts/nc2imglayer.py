
import os
import cv2  
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt



import logging

# 配置日志记录
logging.basicConfig(filename='error.log', level=logging.ERROR)
path = r'/root/cloud/cloudflower/vitdata08/high/'
if not os.path.exists(path):
    os.makedirs(path)
path = r'/root/cloud/cloudflower/vitdata08/low/'
if not os.path.exists(path):
    os.makedirs(path)
path = r'/root/cloud/cloudflower/vitdata08/mid/'
if not os.path.exists(path):
    os.makedirs(path)
path = r'/root/cloud/cloudflower/vitdata/cloud_image08/'
if not os.path.exists(path):
    os.makedirs(path)

for root, dirs, files in os.walk(r"/root/cloud/cloudflower/data/201906/"):
    dirs.sort()  
    files.sort()  
   
    current_dir_name = root.split('/')[-1]
    
  
    if True:
        for file_name in files:
            file_path = os.path.join(root, file_name)  
            print(f"File exists: {file_path}")
            data_time = '-'.join(file_path.split('_')[2:4])  
            try:
                dataset = Dataset(file_path)  

                
                lats = dataset.variables['latitude'][:]  
                lons = dataset.variables['longitude'][:]  
                
                hb_lat = 36.71
                hb_lng = 113.90

              
                region_size = 50
                half_region = region_size // 2

               
                hb_lat_idx = int(round((hb_lat - lats[0]) / (lats[1] - lats[0])))
                hb_lng_idx = int(round((hb_lng - lons[0]) / (lons[1] - lons[0])))

                
                lat_start = max(0, hb_lat_idx - half_region)
                lat_end = min(len(lats), hb_lat_idx + half_region)
                lng_start = max(0, hb_lng_idx - half_region)
                lng_end = min(len(lons), hb_lng_idx + half_region)

              
                try:
                   
                    tbb_13 = dataset.variables['tbb_13'][lat_start:lat_end, lng_start:lng_end]
                    r, g, b = tbb_13, tbb_13, tbb_13

                    cloud_img = np.stack([r, g, b], axis=2)
                    cloud_img = cv2.normalize(cloud_img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                    

                    output_path =r'/root/cloud/cloudflower/vitdata/cloud_image08/' + file_name[7:20] + '.png'
                
                    cv2.imwrite(output_path, cloud_img)
                except KeyError:
                   
                    print(f"Variable 'tbb_13' not found in {file_path}, skipping...")
                    continue  


              
                cloud_img_mid = cloud_img.copy()
                cloud_img_high = cloud_img.copy()
                cloud_img_low = cloud_img.copy()

             
                mask_mid = (tbb_13 > 273.15) | (tbb_13 < 253.15)
                cloud_img_mid[mask_mid] = [255, 0, 0] 
                mid_output_path = r'/root/cloud/cloudflower/vitdata08/mid/' + 'mid_' + file_name[7:20] + '.png'
                cv2.imwrite(mid_output_path, cloud_img_mid)
                print(f"Processed and saved: {mid_output_path}")

               
                mask_high = (tbb_13 >= 253.15)
                cloud_img_high[mask_high] = [255, 0, 0]  
                high_output_path = r'/root/cloud/cloudflower/vitdata08/high/' + 'high_' + file_name[7:20] + '.png'
                cv2.imwrite(high_output_path, cloud_img_high)
                print(f"Processed and saved: {high_output_path}")

               
                mask_low = (tbb_13 <= 273.15)
                cloud_img_low[mask_low] = [255, 0, 0] 
                low_output_path = r'/root/cloud/cloudflower/vitdata08/low/' + 'low_' + file_name[7:20] + '.png'
                cv2.imwrite(low_output_path, cloud_img_low)
                print(f"Processed and saved: {low_output_path}")

            except OSError as e:
              
                logging.error(f"Error processing file {file_path}: {e}")
                continue  


         
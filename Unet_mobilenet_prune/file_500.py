import os
import shutil

# 把图像数据分为300一份
img_dir = r'D:\data\process_data\save'
save_dir = r'D:\data\process_data\line_400'
img_dir_list = os.listdir(img_dir)
img_list = []
count = 0
for img in img_dir_list:
    if img.split('.j')[-1]=='pg':
        img_list.append(img)
for indx, img_name in enumerate(img_list):
    count = indx//300
    origin_file = img_dir + '/' + img_name
    save_img_dir = save_dir + '/' + str(count)
    if os.path.exists(save_img_dir):
        print(str(indx))
    else:
        os.mkdir(save_img_dir)
    shutil.copy(img_dir + '/' + img_name, save_img_dir + '/' + img_name)
    shutil.copy(img_dir + '/' + img_name.replace('.jpg', '.json'), save_img_dir + '/' + img_name.replace('.jpg', '.json'))

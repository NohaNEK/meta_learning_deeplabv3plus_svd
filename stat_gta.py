import os 
import numpy as np

import cv2

from collections import namedtuple

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])
train_id_to_color = np.array(train_id_to_color)
id_to_train_id = np.array([c.train_id for c in classes])
print(id_to_train_id)
# Create a dictionary to index by train_id
indexed_classes = {cls.train_id: cls for cls in classes}
print(indexed_classes[2].name)


from collections import Counter


def read_images_from_folder(folder_path):
    images = []
    cpt =0
    for filename in os.listdir(folder_path) :
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img= np.array(img)
            lb_ids=id_to_train_id[img]
            lb_ids= np.unique(lb_ids)
         
            lb_ids = lb_ids[lb_ids != 255]
            images.append(lb_ids)
            # cpt+= 1
            # if cpt>10:
            #     break

    return images

# Example usage
folder_path = '/media/fahad/Crucial X81/gen_data/label_ids/s1/'
images = read_images_from_folder(folder_path)
print(len(images))
f2='/media/fahad/Crucial X81/gen_data/label_ids/s2/'
im2 = read_images_from_folder(f2)
print(len(im2))
f3='/media/fahad/Crucial X81/gen_data/label_ids/s3/'
im3 = read_images_from_folder(f3)
print(len(im3))
f4='/media/fahad/Crucial X81/gen_data/label_ids/s4/'
im4 = read_images_from_folder(f4)
print(len(im4))
f5='/media/fahad/Crucial X81/gen_data/label_ids/s5/'
im5 = read_images_from_folder(f5)
print(len(im5))
im3=images+im2+ im3 +im4+ im5
flat_array = np.concatenate(im3)

import matplotlib.pyplot as plt
import numpy as np
# Count the occurrences of each number
counts = Counter(flat_array)

# Extract numbers and their counts
numbers = list(counts.keys())
counts = list(counts.values())
print(numbers)
print(counts)
# Create an array with values from 0 to 19
train_id_to_name= np.array([indexed_classes[id].name for id in numbers ] )
print(train_id_to_name)
# train_id_to_name= np.unique(train_id_to_name)
# print(train_id_to_name)


# Create a bar chart
plt.figure(figsize=(12, 6))  # Optional: Adjust the figure size for better readability
#plt.bar(range(len(train_id_to_name)), values, tick_label=train_id_to_name)
plt.bar(range(len(numbers)),counts,tick_label= train_id_to_name)
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add labels and title
plt.xlabel('Classes')
plt.ylabel('Value')
plt.title('statistical distribution of genearated 4k classes ')
plt.savefig('stat_rand_sd_4k_val_final.png', bbox_inches='tight')

# Display the plot
plt.tight_layout()  # Adjust layout to make room for rotated labels
plt.show()


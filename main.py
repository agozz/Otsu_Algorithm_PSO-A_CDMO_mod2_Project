import cv2
from my_utils import basicOtsu_vs_psoOtsu
import os

path_test_images = './lego_marvel_test'

list_of_images = []

for filename in os.listdir(path_test_images):
    img = cv2.imread(os.path.join(path_test_images, filename), cv2.IMREAD_GRAYSCALE)

    if img is not None:
        list_of_images.append(img)

avg_rel_times = basicOtsu_vs_psoOtsu(list_of_images, stamp=True)

print("\nAverage relationship between times:", avg_rel_times, "s")


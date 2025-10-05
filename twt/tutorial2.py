import cv2
import random

img = cv2.imread('assets/823f.jpg', -1)

# look at the 400th pixel in the 250th row
# print("400th pixel at the 250th row: ",img[250][400])
# print("Width (Cols) of the current image:",img.shape[1])
# print(len(img[0]))  # width of the img (cols)


# manipulating indv pixels (values of ndarray)
# for each 100 rows we look throw each of the cols (entire width of the img)
# for i in range(100):
#    for j in range(img.shape[1]): # cols
#       # img[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
#       img[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

# COPYING part of a img and pasting it to another
# find the part of the array, copy that and paste 

img2 = cv2.imread('assets/my-notion-face-portrait.png', -1)

print("Shape of image 2:",img2.shape)
# from row 500-700 of the img copy the cols 600-900
logo = img2[500:1000, 600:1100]


#paste: when we paste the copied part it must be the same dimension as the copied part
img2[100:600, 950:1450] = logo


cv2.imshow("Image2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


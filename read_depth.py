import cv2
import numpy as np
import random

filename = "datademo-depth55.png"
img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
# img = img/0.7071
# img = img*5
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
# print(img2)
# img2 = img2/0.7071
# print(img2)
print("image shapeeeeeeeeeeeeeeeeeeee", img2.shape)
#img = img[:,1280:,0]

#img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

maxvalue = np.max(img)
minvalue = np.min(img)
print(maxvalue)
print(minvalue)



ext_mat = [[-0.0000,  1.0000, -0.0000,  0.0000],
        [-0.6968,  0.0000,  0.7173,  3.2278],
        [ 0.7173,  0.0000,  0.6968, -6.6236],
        [-0.0000, -0.0000, -0.0000,  1.0000]] 

ext_mat = ext_mat[:3]

# int_mat = [[0.6665, 0.0000,  0.0000,  0.0000],
#         [0.0000, 0.8887,  0.0000,  0.0000],
#         [0.0000, 0.0000, -1.0020, -0.2002],
#         [0.0000, 0.0000, -1.0000,  0.0000]]


# int_mat = [[0.011997, 0.0000,  320],
#         [0.0000, 0.011997,  240],
#         [0.0000, 0.0000, 1]]

int_mat = [[213.2899136013455, 0, 320.0], 
        [0, 213.2899136013455, 240.0], 
        [0, 0, 1]]

# int_mat = int_mat[:3]
projection_mat = np.dot(int_mat, ext_mat)



pt1 = np.array([4, 1.3, 0, 1])

pt2 = np.array([4, -1.3, 0, 1])

pt3 = np.array([4, -1.3, 2.4, 1])

pt4 = np.array([4, 1.3, 2.4, 1])

points = [pt1, pt2, pt3, pt4]


transformedpts = []

for pt in points:

    image_zero = np.dot(projection_mat, pt)
    print(projection_mat)
    print(image_zero)

    u = int(image_zero[0]/image_zero[2])
    v = int(image_zero[1]/image_zero[2])

    print(u,v)
    transformedpts += [[u,v]]
    cv2.circle(img, (u,v), 3, (60000), 1)

transformedpts = np.array(transformedpts)
cv2.fillPoly(img, pts=[transformedpts], color=(61000))

show = img<60000
print(show.shape)


img2[show] = 0
print(img2.shape)

# img2 = img2/0.7071

# print(img2.dtype)
# img2 = np.array(img2/0.7071).astype("uint16")
# print(img2.dtype)

show3 = img3>=3200
show4 = img3<=3500

img2[show3] = 0
# img2[show4] = 0
# print(np.max(img2))
# print(np.min(img2))

img2 = img2*5



# pt5 = np.array([3, 1.3, 0, 1])

# pt6 = np.array([3, -1.3, 0, 1])

# pt7 = np.array([3, -1.3, 2.4, 1])

# pt8 = np.array([3, 1.3, 2.4, 1])

# points2 = [pt5, pt6, pt7, pt8]
# transformedpts2 = []

# for pt in points2:

#     image_zero = np.dot(projection_mat, pt)
#     print(projection_mat)
#     print(image_zero)

#     u = int(image_zero[0]/image_zero[2])
#     v = int(image_zero[1]/image_zero[2])

#     print(u,v)
#     transformedpts2 += [[u,v]]
#     cv2.circle(img, (u,v), 3, (60000), 1)

# transformedpts2 = np.array(transformedpts2)
# print(transformedpts2)
# cv2.fillPoly(img2, pts=[transformedpts2], color=(0))

cv2.imshow("demo", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("d2_"+filename, img4)


# for i in range(1):
#     # x = random.randint(0, img.shape[1])
#     # y = random.randint(0, img.shape[0])
#     #value = str(np.round((img[y][x]/255)*1550, 1))
#     # value = str(img[y][x])
#     cv2.circle(img, (u,v), 3, (60000), 1)
#     # cv2.circle(img, (x,y), 2, (60000), 1)
#     # cv2.putText(img, value, (x,y), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (60000), 1, cv2.LINE_AA) 
# cv2.imshow("demo", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

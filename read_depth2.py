import cv2
import numpy as np
import random


filename1 = "Depth0040.png"
# filename1 = "datademo-depth1_40degreescam.png"
filename2 = "datademo-depth55.png"
filenames = [filename1, filename2]
outfiles = []
basemask = []


# Save point cloud as PLY file
def save_ply(file_name, points):
    """ Save points to a PLY file. """
    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
'''.format(len(points))

    with open(file_name, 'w') as f:
        f.write(header)
        for point in points:
            f.write('{} {} {}\n'.format(point[0], point[1], point[2]))


for idx, filename in enumerate(filenames):
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    img2 = img.copy()

    maxvalue = np.max(img)
    minvalue = np.min(img)
    print(maxvalue)
    print(minvalue)

    ####### intrinsic matrix get from blender
    #### 90 degrees camera
    int_mat = np.array([[213.2899136013455, 0, 320.0], 
            [0, 213.2899136013455, 240.0], 
            [0, 0, 1]])

    #### alternative 90 degrees camera (shd be wrong)
    # int_mat = np.array([[213.2899136013455, 0, 320.0], 
    #         [0, 239.9512, 240.0], 
    #         [0, 0, 1]])

    #### 40 degrees camera
    # int_mat = np.array([[888.8888888888889, 0, 320.0], 
    #         [0, 1000.0000, 240.0], 
    #         [0, 0, 1]])

    ####90 degrees camera inverse extrinsic matrix(cam2world)
    int_mat_inverse =np.linalg.inv(int_mat)

    ################# extrinsic matrix get from blender
    #### 90 degrees camera
    ext_mat = [[-0.0000,  1.0000, -0.0000,  0.0000],
            [-0.6968,  0.0000,  0.7173,  3.2278],
            [ 0.7173,  0.0000,  0.6968, -6.6236],
            [-0.0000, -0.0000, -0.0000,  1.0000]] 

    #### flipping the sign
    # ext_mat = [[-0.0000,  1.0000, -0.0000,  0.0000],
    #     [0.6968,  0.0000,  -0.7173,  -3.2278],
    #     [ -0.7173,  0.0000,  -0.6968, 6.6236],
    #     [-0.0000, -0.0000, -0.0000,  1.0000]] 

    ####90 degrees camera inverse extrinsic matrix(cam2world)
    ext_mat_inverse = np.linalg.inv(ext_mat)

    #### override ext_mat_inverse with translation vector (-6999,0,-2999)
    # ext_mat_inverse = np.array([[ 0., -0.69676549, 0.71726447, -6999.89262],
    #                         [1., 0., 0., 0.],
    #                         [0., 0.71726447, 0.69676549, -2299.90963],
    #                         # [0., 0., 0., 1.]
    #                         ])

    #### override ext_mat_inverse with translation vector (0,0,0)
    ext_mat_inverse = np.array([[ 0., -0.69676549, 0.71726447, 0],
                            [1., 0., 0., 0.],
                            [0., 0.71726447, 0.69676549, 0],
                            # [0., 0., 0., 1.]
                            ])

    #### override ext_mat_inverse with 40 degrees camera 
    # ext_mat_inverse = np.array([[-0.0000, -0.0665, 0.9978, 0],
    #                         [1.0000,  0.0000, 0.0000, 0.0000],
    #                         [-0.0000,  0.9978, 0.0665, 0],
    #                         # [0., 0., 0., 1.]
    #                         ])


    ext_mat = ext_mat[:3]
    ext_mat_inverse = ext_mat_inverse[:3]
    print(ext_mat)
    print(ext_mat_inverse)

    projection_mat = np.dot(int_mat, ext_mat)

    boolmap = np.zeros(img.shape).astype(int)
    print(boolmap.shape)
    xmap0 = np.zeros(img.shape).astype(float)
    ymap0 = np.zeros(img.shape).astype(float)
    zmap0 = np.zeros(img.shape).astype(float)

    xmap = np.zeros(img.shape).astype(float)
    ymap = np.zeros(img.shape).astype(float)
    zmap = np.zeros(img.shape).astype(float)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            z = img[y][x]
            vect = np.array([x*z,y*z,z])
            point3d = np.dot(int_mat_inverse,vect)
            point3d = np.append(point3d, [1])

            xmap0[y][x] = point3d[0]
            zmap0[y][x] = point3d[2]
            ymap0[y][x] = point3d[1]

            point3d = np.dot(ext_mat_inverse, point3d)

            xmap[y][x] = point3d[0]
            zmap[y][x] = point3d[2]
            ymap[y][x] = point3d[1]

            if idx ==0:
                if point3d[0] > 4000:
                    boolmap[y][x] = 1
            else:
            # if point3d[0] > 3000 and point3d[0] < 4000:
                if point3d[0] < 4000:
                    # print("false")
                    boolmap[y][x] = 1


    print("min max")
    print(np.min(xmap), np.max(xmap))
    print(np.min(zmap), np.max(zmap))
    print(np.min(ymap), np.max(ymap))
    print(np.min(xmap0), np.max(xmap0))
    print(np.min(zmap0), np.max(zmap0))
    print(np.min(ymap0), np.max(ymap0))


    pointcloud = []
    for y in range(xmap.shape[0]):
        for x in range(xmap.shape[1]):
            pc = (xmap[y][x], ymap[y][x], zmap[y][x])
            pointcloud += [pc]

    save_ply(filename[:-4]+".ply", pointcloud)
   
    xmap = xmap.astype(np.uint16)
    ymap = ymap.astype(np.uint16)
    zmap = zmap.astype(np.uint16)

    # time.sleep(1000)
    # boolmap = boolmap.astype(np.uint16)
    # boolmap = boolmap*60000

    boolmap = boolmap.astype(bool)
    # img2[boolmap==False] = 0

    # theta = np.radians(-45)
    # r =np.array([[1,0,0], 
    #             [0, np.cos(theta), -np.sin(theta)], 
    #             [0, np.sin(theta), np.cos(theta)]])
    # t = np.array([0,0.00,0])

    # h = int_mat.dot(t+r.dot(int_mat_inverse))
    
    # img2_out =cv2.warpPerspective(img2, h, (img2.shape[1]+300, img2.shape[0]))
    # xmap_out =cv2.warpPerspective(xmap, h, (xmap.shape[1]+300, xmap.shape[0]))
    # zmap_out =cv2.warpPerspective(zmap, h, (zmap.shape[1]+300, zmap.shape[0]))

    outfiles += [img2]

    for i in range(100):
        x = random.randint(0, xmap.shape[1]-1)
        y = random.randint(0, xmap.shape[0]-1)
        # print("random", x,y)
        # value = str(np.round((img[y][x]/255)*1550, 1))
        value = str(img2[y][x])
        cv2.circle(img2, (x,y), 2, (60000), 1)
        cv2.putText(img2, "here", (515, 479), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (60000), 1, cv2.LINE_AA) 
        cv2.putText(img2, value, (x,y), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (60000), 1, cv2.LINE_AA) 

    img2 = img2*5
    xmap = xmap*5
    ymap = ymap*5
    zmap = zmap*5

    cv2.imshow("demo", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("d2_"+filename, img2)


# basemask = outfiles[0]
# basesum = np.count_nonzero(basemask)
# # basesum = basemask.sum()
# print("basesum", basesum)

# out = outfiles[1]
# out[basemask==False] = 0
# outcount = np.count_nonzero(out)
# print("outcount", outcount)

# print("percentage", outcount/basesum)
# # diff = outfiles[1] - outfiles[0]
# # print(np.max(diff))
# # print(np.min(diff))

# # for i in range(100):
# #     x = random.randint(0, diff.shape[1])
# #     y = random.randint(0, diff.shape[0])
# #     # value = str(np.round((img[y][x]/255)*1550, 1))
# #     value = str(diff[y][x])
# #     cv2.circle(diff, (x,y), 2, (60000), 1)
# #     cv2.putText(diff, value, (x,y), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (60000), 1, cv2.LINE_AA) 


# cv2.imshow("demo", xmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#     # for i in range(1):
#     #     # x = random.randint(0, img.shape[1])
#     #     # y = random.randint(0, img.shape[0])
#     #     #value = str(np.round((img[y][x]/255)*1550, 1))
#     #     # value = str(img[y][x])
#     #     cv2.circle(img, (u,v), 3, (60000), 1)
#     #     # cv2.circle(img, (x,y), 2, (60000), 1)
#     #     # cv2.putText(img, value, (x,y), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (60000), 1, cv2.LINE_AA) 
#     # cv2.imshow("demo", img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


######################################################################################
# import open3d as o3d


# filename1 = "datademo-depth1_container.png"

# color_raw = o3d.io.read_image("image_0_0001.png")
# depth_raw = o3d.io.read_image("datademo-depth1_container.png")
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color_raw, depth_raw)

# camera_intrinsics = [213.289, 239.951, 320, 240]


# fx, fy, cx, cy = camera_intrinsics

# # Create Open3D point cloud
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image,
#     o3d.camera.PinholeCameraIntrinsic(
#         width= 640,
#         height= 480,
#         fx=fx, fy=fy,
#         cx=cx, cy=cy
#     )
# )

# o3d.visualization.draw_geometries([pcd])
# o3d.io.write_point_cloud("carton55.pcd", pcd)

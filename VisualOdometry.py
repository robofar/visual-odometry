import cv2 as cv
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy
import open3d

# ---------------------------------------- PRVI DIO ---------------------------------------------------- #
def loadGroundTruth():
    gt = open("/home/faris/Desktop/Diplomski/VO/IMU/poses/00.txt","r")
    t_ground_truth = []
    for line in gt:
        data = line.split()
        q = []
        q.append(float(data[3]))
        q.append(float(data[7]))
        q.append(float(data[11]))
        t_ground_truth.append(q)

    return t_ground_truth

def loadImages(s):
    q=''
    if(s=='left'):
        q = '/home/faris/Desktop/Diplomski/VO/Mape/sequences/00/image_0/*.png'

    if(s=='right'):
        q = '/home/faris/Desktop/Diplomski/VO/Mape/sequences/00/image_1/*.png'

    path = glob.glob(q)
    path.sort()
    images = []
    i = 0
    for a in path:
        images.append(cv.imread(a,1))
        print(i)
        i = i + 1
    return images

# Projection matrix
def loadProjectionMatrix(s):
    P_left = [[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
          [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
          [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]
    P_right = copy.deepcopy(P_left)
    P_right[0][3] = -3.861448000000e+02

    P_left = np.array(P_left)
    P_right = np.array(P_right)

    if(s=='left'):
        return P_left

    return P_right

def decomposeProjectionMatrix(P):
    K, R, t, _, _, _, _ = cv.decomposeProjectionMatrix(P)
    t = (t / t[3])[:3]
    t = t.round(4)
    return K,R,t

def initialState(R_w_0,t_w_0):
    C0 = np.eye(4)
    C0[:3, :3] = R_w_0
    C0[:3, 3] = t_w_0.reshape(1, -1)
    return C0


# ----------------------------------- DRUGI DIO ------------------------------------------------ #
# Break pose into coordinates
def coordinates(q):
    x = []
    y = []
    z = []
    for a in q:
        x.append(a[0])
        y.append(a[1])
        z.append(a[2])
    return x,y,z

# Plotting trajectory
def plot(horizontal,vertical,name):
    plt.scatter(horizontal[0], vertical[0], c='green')
    plt.plot(horizontal, vertical, c='red')
    plt.savefig(name + '.png')
    plt.clf()

def plot_both(gt_h,gt_v,vo_h,vo_v,name):
    plt.scatter(gt_h[0], gt_v[0], c='k')
    plt.scatter(vo_h[0], vo_v[0], c='g')
    plt.plot(gt_h, gt_v, c='r',label='gt')
    plt.plot(vo_h, vo_v, c='b',label = 'vo')
    plt.legend()
    plt.savefig(name + '.png')


# ----------------------------------- TRECI DIO ------------------------------------------------ #

# Absolute and relative scale
def absoluteScale(a,b,c,d,e,f):
    x_truth_prev = a
    y_truth_prev = b
    z_truth_prev = c

    x_truth_curr = d
    y_truth_curr = e
    z_truth_curr = f

    abs_scale = np.sqrt((x_truth_curr-x_truth_prev)**2 + (y_truth_curr-y_truth_prev)**2 + (z_truth_curr-z_truth_prev)**2)

    return abs_scale

def relativeScale(pcd1,pcd2):
    X1_1 = pcd1[5]
    X2_1 = pcd1[10]

    X1_2 = pcd2[5]
    X2_2 = pcd2[10]

    difference_1 = np.sqrt( (X1_1[0] - X2_1[0])**2 + (X1_1[1] - X2_1[1])**2 + (X1_1[2] - X2_1[2])**2 )
    difference_2 = np.sqrt( (X1_2[0] - X2_2[0])**2 + (X1_2[1] - X2_2[1])**2 + (X1_2[2] - X2_2[2])**2 )

    rs = difference_1/difference_2

    return rs

# ----------------------------------- CETRVTI DIO ------------------------------------------------ #

# Disparity and depth - for computing 3D points
def disparityMap(img_left,img_right,matcher_name='sgbm'):
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher = 0
    if matcher_name == 'bm':
        matcher = cv.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                    )

    elif matcher_name == 'sgbm':
        matcher = cv.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8 * 1 * block_size ** 2,
                                        P2=32 * 1 * block_size ** 2,
                                        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
                                        )

    disparity_map = matcher.compute(img_left, img_right).astype(np.float32) / 16
    return disparity_map


def depthMap(disparity_map,K_left,b):
    # Get focal length of x axis for left camera (x axis because disparity is horizontal difference)
    f = K_left[0][0]

    # Avoid instability and division by zero
    # Ne znamo koji je disparitet u ovim tackama zato su vrijednosti dispariteta -1 i 0
    # Tako da za ove nepoznate disparitete stavicemo im male vrijednosti da bi depth u tim pixelima bila velika
    # i onda tako mozemo znati da su to oni pixeli za koje nismo znali disparitet
    disparity_map[disparity_map == 0.0] = 0.1
    disparity_map[disparity_map == -1.0] = 0.1

    # Make empty depth map then fill with depth
    depth_map = np.ones(disparity_map.shape)
    depth_map = f * b / disparity_map

    return depth_map

def depthMapMaskIndex(depth_map):
    index = 0
    for i,pixel_value in enumerate(depth_map[0]):
        if pixel_value < depth_map.max():
            index = i
            break

    return index

# MOZE BITI DOBRA MASKA ZA FEATURE DETACTION I TO
def depthMapMask(shape,index):
    mask = np.zeros(shape)
    xmax = shape[0]
    ymax = shape[1]
    cv.rectangle(mask, (0, 0), (xmax, ymax), (255), thickness=-1)
    return mask

# ----------------------------------- PETI DIO ------------------------------------------------ #

# Triangulation
def triangulation(intrinsic,prev_points,curr_points,Cn_prev,Cn):
    Rn_prev = [ [Cn_prev[0][0],Cn_prev[0][1],Cn_prev[0][2]] , [Cn_prev[1][0],Cn_prev[1][1],Cn_prev[1][2]] , [Cn_prev[2][0],Cn_prev[2][1],Cn_prev[2][2]]  ]
    tn_prev = [ [Cn_prev[0][3]] , [Cn_prev[1][3]] , [Cn_prev[2][3]] ]
    Rntn_prev = np.hstack((Rn_prev,tn_prev))
    Pn_prev = np.matmul(intrinsic,Rntn_prev)

    Rn = [ [Cn[0][0],Cn[0][1],Cn[0][2]] , [Cn[1][0],Cn[1][1],Cn[1][2]] , [Cn[2][0],Cn[2][1],Cn[2][2]]  ]
    tn = [ [Cn[0][3]] , [Cn[1][3]] , [Cn[2][3]] ]
    Rntn = np.hstack((Rn,tn))
    Pn = np.matmul(intrinsic,Rntn)

    pts_3d_hom = cv.triangulatePoints(Pn_prev,Pn,prev_points.reshape(2,-1),curr_points.reshape(2,-1))

    return pts_3d_hom

def from_homogenous_to_euclidian(p3d_hom):
    p3d = p3d_hom / p3d_hom[3]
    p3d = np.transpose(p3d)
    p3d = p3d[:, :3]

    return p3d

# ----------------------------------- SESTI DIO ------------------------------------------------ #

def lidarPC2image(pointcloud, imheight, imwidth, Tr, P0):
    '''
        Takes a pointcloud of shape Nx4 and projects it onto an image plane, first transforming
        the X, Y, Z coordinates of points to the camera frame with tranformation matrix Tr, then
        projecting them using camera projection matrix P0.

        Arguments:
        pointcloud -- array of shape Nx4 containing (X, Y, Z, reflectivity)
        imheight -- height (in pixels) of image plane
        imwidth -- width (in pixels) of image plane
        Tr -- 3x4 transformation matrix between lidar (X, Y, Z, 1) homogeneous and camera (X, Y, Z)
        P0 -- projection matrix of camera (should have identity transformation if Tr used)

        Returns:
        render -- a (imheight x imwidth) array containing depth (Z) information from lidar scan

        '''

    # We know the lidar X axis points forward, we need nothing behind the lidar, so we
    # ignore anything with a X value less than or equal to zero
    pointcloud = pointcloud[pointcloud[:, 0] > 0]

    # We do not need reflectance info, so drop last column and replace with ones to make
    # coordinates homogeneous for tranformation into the camera coordinate frame
    pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1, 1))])

    # Transform pointcloud into camera coordinate frame
    cam_xyz = Tr.dot(pointcloud.T)

    # Ignore any points behind the camera (probably redundant but just in case)
    # cam_xyz = cam_xyz[:, cam_xyz[2] > 0]

    # Extract the Z row which is the depth from camera
    depth_camera_frame = cam_xyz[2].copy()

    # Project coordinates in camera frame to flat plane at Z=1 by dividing by Z
    cam_xyz /= cam_xyz[2]

    # Add row of ones to make our 3D coordinates on plane homogeneous for dotting with P0
    cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])

    # Get pixel coordinates of X, Y, Z points in camera coordinate frame
    projection = P0.dot(cam_xyz)

    # Turn pixels into integers for indexing
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')

    # Limit pixel coordinates considered to those that fit on the image plane
    indices = np.where((pixel_coordinates[:, 0] < imwidth)
                       & (pixel_coordinates[:, 0] >= 0)
                       & (pixel_coordinates[:, 1] < imheight)
                       & (pixel_coordinates[:, 1] >= 0)
                       )


    pixel_coordinates = pixel_coordinates[indices]
    depth_camera_frame = depth_camera_frame[indices]

    # Establish empty render image, then fill with the depths of each point
    render = np.zeros((imheight, imwidth))
    # Ova provjera opet se vrsi za svaki slucaj
    for j, (u, v) in enumerate(pixel_coordinates):
        if u >= imwidth or u < 0:
            continue
        if v >= imheight or v < 0:
            continue
        render[v, u] = depth_camera_frame[j]
    # Fill zero values with large distance so they will be ignored. (Using same max value)
    render[render == 0.0] = 3861.45

    return render

# Point-cloud
# ----------------------------------- SEDMI DIO ------------------------------------------------ #
def writePCtoFile(pc):
    f = open("myfile.txt", "w")
    for i in range(0,len(pc)):
        f.write(str(pc[i][0]))
        f.write(" ")
        f.write(str(pc[i][1]))
        f.write(" ")
        f.write(str(pc[i][2]))
        f.write("\n")
    f.close()

def visualizePC2():
    pcd = open3d.io.read_point_cloud("/home/faris/Desktop/SVO/myfile.txt", format='xyz')
    open3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def visualizePC(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    

# ----------------------------------- OSMI DIO ------------------------------------------------ #
def feature_detection_extraction(image,detector='sift',mask=None):
    """
        Find keypoints and descriptors for the image

        Arguments:
        image -- a grayscale image

        Returns:
        kp -- list of the extracted keypoints (features) in an image
        des -- list of the keypoint descriptors in an image
        """
    det = None
    if detector == 'sift':
        det = cv.SIFT_create()
    elif detector == 'orb':
        det = cv.ORB_create()
    elif detector == 'surf':
        det = cv.xfeatures2d.SURF_create()

    kp, des = det.detectAndCompute(image, mask)

    return kp, des

# ----------------------------------- DEVETI DIO ------------------------------------------------ #
def feature_matching(des1, des2, matching='BF', detector='sift', sort=False, k=2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'
    detector -- (str) can be 'sift or 'orb'. Default is 'sift'
    sort -- (bool) whether to sort matches by distance. Default is True
    k -- (int) number of neighbors to match to each feature.

    Returns:
    matches -- list of matched features from two images. Each match[i] is k or less matches for
               the same query descriptor
    """
    matcher = None
    matches = None
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv.BFMatcher_create(cv.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv.BFMatcher_create(cv.NORM_HAMMING2, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=k)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=k)

    if sort:
        matches = sorted(matches, key=lambda x: x[0].distance)

    return matches

# ----------------------------------- DESETI DIO ------------------------------------------------ #
def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    for m, n in matches:
        if m.distance < dist_threshold * n.distance:
            filtered_match.append(m)

    return filtered_match

# ----------------------------------- JEDANAESTI DIO ------------------------------------------------ #
def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    """
    image_matches = cv.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    plt.pause(0.1)

# ----------------------------------- DVANAESTI DIO ------------------------------------------------ #
def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera intrinsic calibration matrix

    Optional arguments:
    depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
    max_depth -- Threshold of depth to ignore matched features. 3000 is default

    Returns:
    rmat -- estimated 3x3 rotation matrix
    tvec -- estimated 3x1 translation vector
    image1_points -- matched feature pixel coordinates in the first image.
                     image1_points[i] = [u, v] -> pixel coordinates of i-th match
    image2_points -- matched feature pixel coordinates in the second image.
                     image2_points[i] = [u, v] -> pixel coordinates of i-th match

    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    # Every feature in prev_frame has only one correspondent in curr_frame
    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])

    if depth1 is not None:
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        object_points = np.zeros((0, 3))
        delete = [] #indexes (indicies)

        # Extract depth information of query image at match points and build 3D positions
        for i, (u, v) in enumerate(image1_points):
            z = depth1[int(v), int(u)]
            # If the depth at the position of our matched feature is above 3000, then we
            # ignore this feature because we don't actually know the depth and it will throw
            # our calculations off. We add its index to a list of coordinates to delete from our
            # keypoint lists, and continue the loop. After the loop, we remove these indices
            if z >= max_depth:
                delete.append(i)
                continue

            # Use arithmetic to extract x and y (faster than using inverse of k) -> x,y of 3D point in camera coordinate sysyem
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            # object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])

        # Izbrisemo tacke za koje nismo imali informacije o depth-u njihovom
        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)

        # Use PnP algorithm with RANSAC for robustness to outliers
        _, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image2_points, k, None)
        # print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))

        # Above function returns axis angle rotation representation rvec, use Rodrigues formula
        # to convert this to our desired format of a 3x3 rotation matrix
        rmat = cv.Rodrigues(rvec)[0]

    else:
        # With no depth provided, use essential matrix decomposition instead. This is not really
        # very useful, since you will get a 3D motion tracking but the scale will be ambiguous
        # image1_points_hom = np.hstack([image1_points, np.ones(len(image1_points)).reshape(-1, 1)])
        # image2_points_hom = np.hstack([image2_points, np.ones(len(image2_points)).reshape(-1, 1)])
        E, outliers_inliers_mask = cv.findEssentialMat(image1_points, image2_points, k, method=cv.RANSAC,prob=0.999,threshold=0.1)

        # Mask from findEssentialMatrix separates outliers from inliers (dont know where to use it)
        # outliers_inliers_mask = np.ravel(outliers_inliers_mask)
        # image1_points_inliers = image1_points[outliers_inliers_mask == 1]
        # image2_points_inliers = image2_points[outliers_inliers_mask == 1]

        _, rmat, tvec, mask = cv.recoverPose(E, image1_points, image2_points, k)


    return rmat, tvec, image1_points, image2_points

# ----------------------------------- TRINAESTI DIO ------------------------------------------------ #
# We need a function to tell us how much error we have compared to the ground truth
# We will use Euclidean distance of each camera pose from the ground truth to give us
# Mean Squared Error (mse), Root Mean Squared Error (rmse), or Mean Absolute Error (mae)
def calculate_error(ground_truth, estimated, error_type='mse'):
    '''
    Takes arrays of ground truth and estimated poses of shape Nx3x4, and computes error using
    Euclidean distance between true and estimated 3D coordinate at each position.

    Arguments:
    ground_truth -- Nx3x4 array of ground truth poses
    estimated -- Nx3x4 array of estimated poses

    Optional Arguments:
    error_type -- (str) can be 'mae', 'mse', 'rmse', or 'all' to return dictionary of all 3

    Returns:
    error -- either a float or dictionary of error types and float values

    '''
    # Find the number of frames in the estimated trajectory to compare with
    nframes_est = estimated.shape[0]

    def get_mse(ground_truth, estimated):
        se = np.sqrt((ground_truth[nframes_est, 0, 3] - estimated[:, 0, 3]) ** 2
                     + (ground_truth[nframes_est, 1, 3] - estimated[:, 1, 3]) ** 2
                     + (ground_truth[nframes_est, 2, 3] - estimated[:, 2, 3]) ** 2) ** 2
        mse = se.mean()
        return mse

    def get_mae(ground_truth, estimated):
        ae = np.sqrt((ground_truth[nframes_est, 0, 3] - estimated[:, 0, 3]) ** 2
                     + (ground_truth[nframes_est, 1, 3] - estimated[:, 1, 3]) ** 2
                     + (ground_truth[nframes_est, 2, 3] - estimated[:, 2, 3]) ** 2)
        mae = ae.mean()
        return mae

    if error_type == 'mae':
        return get_mae(ground_truth, estimated)
    elif error_type == 'mse':
        return get_mse(ground_truth, estimated)
    elif error_type == 'rmse':
        return np.sqrt(get_mse(ground_truth, estimated))
    elif error_type == 'all':
        mae = get_mae(ground_truth, estimated)
        mse = get_mse(ground_truth, estimated)
        rmse = np.sqrt(mse)
        return {'mae': mae,
                'rmse': rmse,
                'mse': mse}

# ----------------------------------- CETRNAESTI DIO ------------------------------------------------ #
def transformationMatrix(Rk,tk):
    Rt = np.hstack((Rk,tk))
    homogenous_part = np.array([0,0,0,1])
    Tk = np.vstack((Rt, homogenous_part))

    return Tk






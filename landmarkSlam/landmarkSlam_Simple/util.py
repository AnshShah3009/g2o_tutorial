import copy
from operator import pos
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

np.random.seed(42)


def getColors(n):
    if (n > 67):
        raise ValueError(f'n must be atmost 67, instead got {n}')

    colors = [
        (1, 100, 211),
        (174, 212, 46),
        (100, 52, 183),
        (85, 188, 47),
        (190, 97, 233),
        (1, 197, 87),
        (226, 89, 219),
        (0, 153, 32),
        (232, 42, 163),
        (107, 221, 136),
        (248, 44, 151),
        (0, 117, 48),
        (177, 124, 255),
        (222, 164, 0),
        (122, 126, 255),
        (252, 147, 20),
        (70, 68, 175),
        (168, 147, 0),
        (117, 48, 164),
        (177, 209, 131),
        (179, 0, 135),
        (1, 219, 210),
        (232, 11, 80),
        (13, 217, 245),
        (255, 122, 39),
        (0, 96, 179),
        (208, 92, 0),
        (3, 184, 242),
        (163, 46, 0),
        (133, 160, 255),
        (188, 102, 0),
        (235, 135, 255),
        (59, 92, 26),
        (255, 98, 193),
        (1, 147, 107),
        (210, 0, 107),
        (127, 216, 174),
        (255, 71, 89),
        (1, 98, 150),
        (255, 135, 71),
        (149, 181, 255),
        (128, 111, 0),
        (254, 161, 255),
        (236, 192, 99),
        (121, 55, 134),
        (210, 200, 126),
        (155, 20, 106),
        (255, 157, 100),
        (76, 75, 142),
        (130, 66, 0),
        (212, 187, 252),
        (164, 19, 40),
        (76, 117, 165),
        (145, 52, 27),
        (249, 172, 245),
        (128, 64, 53),
        (255, 141, 215),
        (255, 153, 127),
        (134, 50, 111),
        (240, 167, 158),
        (146, 44, 84),
        (180, 145, 195),
        (255, 103, 163),
        (129, 72, 105),
        (255, 159, 165),
        (255, 165, 200),
        (0, 0, 0),
    ]
    return np.array(colors[n], dtype=np.float64) / 255

def getVertices(points):
    """
    Generating Cube vertices
    """
    vertices = []

    for ele in points:
        if(ele is not None):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.paint_uniform_color([0.9, 0.2, 0])

            trans = np.identity(4)
            trans[0, 3] = ele[0]
            trans[1, 3] = ele[1]
            trans[2, 3] = ele[2]

            sphere.transform(trans)
            vertices.append(sphere)

    return vertices, points


def getFrames(poses):
    """
    Generating Robot positions
    """
    # posei = ( x, y, z, thetaZ(deg) )
    frames = []

    for pose in poses:
        T = np.identity(4)
        T[0, 3], T[1, 3], T[2, 3] = pose[0], pose[1], pose[2]
        T[0:3, 0:3] = R.from_euler('z', pose[3], degrees=True).as_matrix()

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.2, origin=[0, 0, 0])
        frame.transform(T)
        frames.append(frame)

    return frames, poses


def getLocalCubes(points, poses):
    """
    Returns local point cloud cubes
    """
    points = np.array(points)
    poses = np.array(poses)

    nPoses, nPoints, pointDim = poses.shape[0], points.shape[0], points.shape[1]
    cubes = np.zeros((nPoses, nPoints, pointDim))

    for i, pose in enumerate(poses):
        cube = []

        T = np.identity(4)
        T[0, 3], T[1, 3], T[2, 3] = pose[0], pose[1], pose[2]
        T[0:3, 0:3] = R.from_euler('z', pose[3], degrees=True).as_matrix()

        for pt in np.hstack((points, np.ones((points.shape[0], 1)))):
            ptLocal = np.linalg.inv(T) @ pt.reshape(4, 1)

            cube.append(ptLocal.squeeze(1)[0:3])

        cubes[i] = np.asarray(cube)

    return cubes


def icpTransformations(cubes):
    # T1_2 : 2 wrt 1

    if cubes.shape[0] < 2:
        raise ValueError('Need at least 2 observations to register.')

    pcds = []
    for i in range(cubes.shape[0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cubes[i])
        pcds.append(pcd)

    corres = np.arange(cubes.shape[1]) # correspondences, no of points * 2 (because pairwise)
    corres = np.vstack((corres, corres)).T
    corres = o3d.utility.Vector2iVector(corres)

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    transforms = []
    for i in range(cubes.shape[0] - 1): # for every consecutive pair of observations
        tr = p2p.compute_transformation(pcds[i + 1], pcds[i], corres)
        transforms.append(tr)

    draw_registration_result(pcds[1], pcds[0], transforms[0])
    return np.array(transforms)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.get_render_option().point_size = 15
    vis.run()
    vis.destroy_window()


def registerCubes(trans, cubes):
    # Registering noisy cubes in first frame

    pcds = []
    for i in range(cubes.shape[0]):
        pcd = getCloud(cubes[i], getColors(i))
        pcds.append(pcd)

    pcdsTransformed = []
    for i, pcd in enumerate(pcds):
        tr = np.eye(4)
        for j in range(i):
            tr = tr @ trans[j]
        pcdsTransformed.append([
            ele.transform(tr) for ele in pcd
        ])
    o3d.visualization.draw_geometries([mesh for pcd in pcdsTransformed for mesh in pcd])


def getCloud(cube, color):
    vertices = []

    for ele in cube:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.paint_uniform_color(color)

        trans = np.identity(4)
        trans[0, 3] = ele[0]
        trans[1, 3] = ele[1]
        trans[2, 3] = ele[2]

        sphere.transform(trans)
        vertices.append(sphere)

    return vertices


def add_noise(arr, noise):
    noiseMat = arr + np.random.normal(0, noise, arr.shape)
    return noiseMat


def writeRobotPose(poses, GPI, g2o):
    poses = add_noise(poses, GPI)

    fromWorld = []
    for pose in poses:
        Tw = np.eye(4)
        Tw[0, 3], Tw[1, 3], Tw[2, 3] = pose[0], pose[1], pose[2]
        Tw[0:3, 0:3] = R.from_euler('z', pose[3], degrees=True).as_matrix()
        fromWorld.append(Tw)

    posesRobot = []
    for Tw in fromWorld:
        pose = [Tw[0, 3], Tw[1, 3], Tw[2, 3]] + \
            list(R.from_dcm(Tw[0:3, 0:3]).as_quat())
        posesRobot.append(pose)

    sp = ' '

    for i, (x, y, z, qx, qy, qz, qw) in enumerate(posesRobot):
        line = "VERTEX_SE3:QUAT " + str(i+1) + sp + str(x) + sp + str(y) + sp + str(
            z) + sp + str(qx) + sp + str(qy) + sp + str(qz) + sp + str(qw) + ' \n'
        g2o.write(line)


def writeOdom(trans, g2o):
    sp = ' '
    info = '20 0 0 0 0 0 20 0 0 0 0 20 0 0 0 20 0 0 20 0 20'

    for i, T in enumerate(trans):
        dx, dy, dz = T[0, 3], T[1, 3], T[2, 3]

        qx, qy, qz, qw = list(R.from_dcm(T[0:3, 0:3]).as_quat())

        line = "EDGE_SE3:QUAT " + str(i+1) + sp + str(i+2) + sp + str(dx) + sp + str(dy) + sp + str(
            dz) + sp + str(qx) + sp + str(qy) + sp + str(qz) + sp + str(qw) + sp + info + '\n'

        g2o.write(line)


def writeCubeVertices(points, GLI, g2o):
    points = add_noise(points, GLI)

    quat = "0 0 0 1 \n"
    sp = ' '

    for i, (x, y, z) in enumerate(points):
        line = "VERTEX_SE3:QUAT " + \
            str(i+6) + sp + str(x) + sp + str(y) + \
            sp + str(z) + sp + quat

        g2o.write(line)


def writeLandmarkEdge(cubes, g2o):
    quat = "0 0 0 1"
    sp = ' '
    info = '40 0 0 0 0 0 40 0 0 0 0 40 0 0 0 0.000001 0 0 0.000001 0 0.000001'

    for i, cube in enumerate(cubes):
        for j, (x, y, z) in enumerate(cube):
            line = "EDGE_SE3:QUAT " + str(i+1) + sp + str(j+6) + sp + str(
                x) + sp + str(y) + sp + str(z) + sp + quat + sp + info + '\n'

            g2o.write(line)


def writeG2o(poses, points, trans, cubes, GPI, GLI):
    g2o = open("noise.g2o", 'w')

    g2o.write('# Robot poses\n\n')

    writeRobotPose(poses, GPI, g2o)

    g2o.write("\n # Cube vertices\n\n")

    writeCubeVertices(points, GLI, g2o)

    g2o.write('\n# Odometry edges\n\n')

    writeOdom(trans, g2o)

    g2o.write('\n# Landmark edges\n\n')

    writeLandmarkEdge(cubes, g2o)

    g2o.write("\nFIX 1\n")

    g2o.close()


def readG2o(fileName, num_poses):
    f = open(fileName, 'r')
    A = f.readlines()
    f.close()

    poses = []
    poses_coord = []
    landmarks = []

    for line in A:
        if "VERTEX_SE3:QUAT" in line:
            (ver, ind, x, y, z, qx, qy, qz, qw, newline) = line.split(' ')

            if int(ind) <= num_poses:
                T = np.identity(4)
                T[0, 3], T[1, 3], T[2, 3] = x, y, z
                T[0:3, 0:3] = R.from_quat([qx, qy, qz, qw]).as_matrix()

                poses.append(T)
                poses_coord.append(np.array([x, y, z]))

            else:
                landmarks.append(np.array([x, y, z]))

    poses = np.asarray(poses)
    landmarks = np.asarray(landmarks, dtype=np.float)
    poses_coord = np.asarray(poses_coord, dtype=np.float)

    return poses, poses_coord, landmarks


def getRelativeEdge(poses):
    trans = []
    for i in range(poses.shape[0] - 1):
        tr = np.linalg.inv(poses[i]) @ poses[i+1]
        trans.append(tr)

    return np.array(trans)

import os
import util
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


np.random.seed(42)


def visualizeData(vertices, frames):
    geometries = []
    geometries = geometries + vertices + frames
    o3d.visualization.draw_geometries(geometries)


def optimize():
    cmd = "g2o -robustKernel Cauchy -robustKernelWidth 1 -o {} -i 50 {} > /dev/null 2>&1".format(
        "opt.g2o", "noise.g2o")
    os.system(cmd)


def plot(gt, init, opt, title):
     # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(gt[:, 0], gt[:, 1], gt[:, 2],
                 color="green", label="Ground Truth")
    ax.scatter3D(init[:, 0], init[:, 1], init[:, 2],
                 color="red", label="Initial Estimate")
    ax.scatter3D(opt[:, 0], opt[:, 1], opt[:, 2],
                 color="blue", label="Optimised")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPI', type=float, required=True)
    parser.add_argument('-RPO', type=float, required=True)
    parser.add_argument('-GLI', type=float, required=True)
    parser.add_argument('-RLO', type=float, required=True)
    args = parser.parse_args()

    # Generating Cube vertices
    points = [[0, 8, 8], [0, 0, 8], [0, 0, 0], [0, 8, 0],
              [8, 8, 8], [8, 0, 8], [8, 0, 0], [8, 8, 0]]
    vertices, points = util.getVertices(points)
    points = np.asarray(points)

    # Generating Robot positions
    poses = [[-12, 0, 0, 0], [-10, -4, 0, 30],
             [-8, -8, 0, 60], [-4, -12, 0, 75], [0, -16, 0, 80]]
    frames, poses = util.getFrames(poses)
    poses = np.asarray(poses)

    # Visualizing Ground Truth robot positions and cube vertices
    visualizeData(vertices, frames)

    gtCubes = util.getLocalCubes(points, poses)
    noisyCubes = util.add_noise(gtCubes, args.RLO)

    # Registering two point clouds using transformation obtained using ICP.
    trans = util.icpTransformations(noisyCubes)
    trans = util.add_noise(trans, args.RPO)

    # util.registerCubes(trans, noisyCubes)

    util.writeG2o(poses, points, trans, noisyCubes, args.GPI, args.GLI)
    optimize()

    _, initPoses_coord, initLandmarks = util.readG2o("noise.g2o")
    optPoses, optPoses_coord, optLandmarks = util.readG2o("opt.g2o")

    optEdges = util.getRelativeEdge(optPoses)
    util.registerCubes(optEdges, noisyCubes)

    # plotting gt, initial and optimised landmarks
    plot(points, initLandmarks, optLandmarks, "Landmarks")

    # plotting gt, initial and optimised robot poses
    plot(poses, initPoses_coord, optPoses_coord, "Robot Poses")

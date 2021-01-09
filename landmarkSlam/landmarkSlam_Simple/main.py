import os
import util
import numpy as np
import open3d as o3d

np.random.seed(42)

HIGH_NOISE = 2.0
LOW_NOISE = 0.2


def visualizeData(vertices, frames):
    geometries = []
    geometries = geometries + vertices + frames
    o3d.visualization.draw_geometries(geometries)


def addNoiseCubes(cubes, noise=0):
    noisyCubes = np.zeros(cubes.shape)
    for i in range(cubes.shape[0]):
        noiseMat = np.random.normal(
            0, noise, cubes[i].size).reshape(cubes[i].shape)
        noisyCubes[i] = cubes[i] + noiseMat

    return noisyCubes


def optimize():
    cmd = "g2o -robustKernel Cauchy -robustKernelWidth 1 -o {} -i 50 {} > /dev/null 2>&1".format(
        "opt.g2o", "noise.g2o")
    os.system(cmd)


if __name__ == "__main__":
    # Generating Cube vertices
    vertices, points = util.getVertices()
    # Generating Robot positions
    frames, poses = util.getFrames()
    # Visualizing Ground Truth robot positions and cube vertices
    visualizeData(vertices, frames)

    gtCubes = util.getLocalCubes(points, poses)
    noisyCubesHigh = addNoiseCubes(gtCubes, noise=HIGH_NOISE)
    noisyCubesLow = addNoiseCubes(gtCubes, noise=LOW_NOISE)

    # Registering two point clouds using transformation obtained using ICP.
    # Here by giving `noisyCubesHigh`, we are just saying our relative poses are very noisy.
    # Do not worry about how we are obtaining it (ICP), just think of it as some noisy odometry
    # sensor giving consecutive relative poses.
    trans = util.icpTransformations(noisyCubesHigh)

    util.registerCubes(trans, noisyCubesLow)

    util.writeG2o(trans, noisyCubesLow)
    optimize()

    optPoses = util.readG2o("opt.g2o")
    optEdges = util.getRelativeEdge(optPoses)
    util.registerCubes(optEdges, noisyCubesLow)

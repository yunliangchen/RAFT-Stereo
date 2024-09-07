import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
import glob
import h5py
import json
import os
import argparse
import pyzed.sl as sl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# import torch
# import torch.nn.functional as F
import PIL.Image as Image


import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cuda'



class RAFT:
    def __init__(self, args):
        self.model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        self.model.load_state_dict(torch.load(args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

        self.valid_iters = args.valid_iters

    def compute_disparity(self, left_image, right_image):
        """
        left_image: np array (uint8) of shape (H, W, 3)
        right_image: np array (uint8) of shape (H, W, 3)
        """
        left_image = torch.from_numpy(left_image).permute(2, 0, 1).float()[None].to(DEVICE)
        right_image = torch.from_numpy(right_image).permute(2, 0, 1).float()[None].to(DEVICE)
        with torch.no_grad():
            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            _, flow_up = self.model(left_image, right_image, iters=self.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

        return -flow_up.cpu().numpy().squeeze()

    def disparity_to_depth(self, disparity, focal_length, baseline):
        """
        disparity: np array of shape (H, W)
        focal_length: float
        """
        depth = focal_length * baseline / disparity
        return depth





# For DROID data format
def get_camera_extrinsic_matrix(calibration_6d):
    calibration_matrix = np.array(calibration_6d)
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    extrinsic_matrix = np.hstack((rotation_matrix, cam_pose.reshape(3, 1)))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
    return extrinsic_matrix


class StereoCamera:
    left_images: list[np.ndarray]
    right_images: list[np.ndarray]
    depth_images: list[np.ndarray]
    width: float
    height: float
    left_dist_coeffs: np.ndarray
    left_intrinsic_mat: np.ndarray

    right_dist_coeffs: np.ndarray
    right_intrinsic_mat: np.ndarray

    def __init__(self, recordings: Path, serial: int, baseline: float):
        
        
        init_params = sl.InitParameters()
        svo_path = recordings / "SVO" / f"{serial}.svo"
        init_params.set_from_svo_file(str(svo_path))
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Quality: NEURAL > ULTRA > QUALITY > PERFORMANCE
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = 0.1
        init_params.enable_right_side_measure = True
        if int(serial) < 20000000:
            init_params.camera_image_flip = sl.FLIP_MODE.ON
        else:
            init_params.camera_image_flip = sl.FLIP_MODE.OFF

        zed = sl.Camera()
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Error reading camera data: {err}")
        
        params = (
            zed.get_camera_information().camera_configuration.calibration_parameters
        )
        
        self.left_intrinsic_mat = np.array(
            [
                [params.left_cam.fx, 0, params.left_cam.cx],
                [0, params.left_cam.fy, params.left_cam.cy],
                [0, 0, 1],
            ]
        )
        self.right_intrinsic_mat = np.array(
            [
                [params.right_cam.fx, 0, params.right_cam.cx],
                [0, params.right_cam.fy, params.right_cam.cy],
                [0, 0, 1],
            ]
        )
        self.zed = zed
        self.params = params
        self.baseline = zed.get_camera_information().camera_configuration.calibration_parameters.get_camera_baseline()
        print("baseline:", self.baseline)

        # have RAFT as well
        args = argparse.Namespace(
            restore_ckpt="/home/lawchen/project/raft_stereo/RAFT-Stereo/models/raftstereo-middlebury.pth",
            mixed_precision=False,
            valid_iters=32,
            hidden_dims=[128]*3,
            corr_implementation="reg", # reg, alt, reg_cuda, alt_cuda
            shared_backbone=False,
            corr_levels=4,
            corr_radius=4,
            n_downsample=2,
            context_norm="batch",
            slow_fast_gru=False,
            n_gru_layers=3
        )
        self.raft = RAFT(args)


    def get_next_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Gets the the next from both cameras and maybe computes the depth."""

        
        left_image = sl.Mat()
        right_image = sl.Mat()
        depth_image = sl.Mat()
        point_cloud = sl.Mat()

        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.enable_fill_mode = True
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            left_image = np.array(left_image.numpy())

            self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            right_image = np.array(right_image.numpy())

            self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH) # Depth is aligned on the left image
            depth_image = np.array(depth_image.numpy())

            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            point_cloud = np.array(point_cloud.get_data())


            # use RAFT to compute disparity
            disparity = self.raft.compute_disparity(left_image[..., :3], right_image[..., :3])
            raft_depth_image = self.raft.disparity_to_depth(disparity, self.params.left_cam.fx, self.baseline)


            return (left_image[..., :3], right_image[..., :3], depth_image, point_cloud[..., :3], raft_depth_image)
        else:
            return None
        
    def close(self):
        self.zed.close()



def depth2cloud(depth_img, intrinsics, extrinsics, wrist=False):
    im_height, im_width = depth_img.shape
    ww = np.linspace(0, im_width - 1, im_width)
    hh = np.linspace(0, im_height - 1, im_height)
    xmap, ymap = np.meshgrid(ww, hh)
    points_2d = np.column_stack((xmap.ravel(), ymap.ravel()))
    depth_img = depth_img.flatten()

    # set depth value that are greater than 70 percentile to nan
    # if wrist:
    #     threshold1 = np.nanpercentile(depth_img, 90) 
    #     threshold2 = np.nanpercentile(depth_img, 0)
    #     print("wrist thresholds:", threshold2, threshold1)
    #     threshold1 = 1
    # else:
    #     threshold1 = np.nanpercentile(depth_img, 60) 
    #     threshold2 = np.nanpercentile(depth_img, 0)
    #     print("ext thresholds:", threshold2, threshold1)
    #     threshold1 = 1
    # depth_img[depth_img > threshold1] = np.nan
    # depth_img[depth_img < threshold2] = np.nan

    homogenous = np.pad(points_2d, ((0, 0), (0, 1)), constant_values=1.0)
    points_in_camera_axes = np.matmul(
        np.linalg.inv(intrinsics),
        homogenous.T * depth_img[None]
    )
    points_in_camera_axes_homogenous = np.pad(points_in_camera_axes, ((0, 1), (0, 0)), constant_values=1.0)
    points_in_world_frame_homogenous = np.matmul(
        # np.linalg.inv(extrinsics), 
        extrinsics,
        points_in_camera_axes_homogenous
    ) # (4, im_height * im_width)
    return points_in_world_frame_homogenous[:3, :].reshape(3, im_height, im_width), points_in_camera_axes


def cam_pointcloud_to_world_pointcloud(cam_pointcloud, extrinsics):
    """Converts a point cloud from the camera frame to the world frame.
    cam_pointcloud: (H, W, 3) -> (3, N)
    extrinsics: (4, 4)
    return: (3, H, W)
    """
    H, W = cam_pointcloud.shape[:2]
    cam_pointcloud = cam_pointcloud.reshape(-1, 3).T
    return (np.dot(extrinsics[:3, :3], cam_pointcloud) + extrinsics[:3, 3][:, None]).reshape(3, H, W)


def visualize_actions_and_point_clouds(visible_pcd, visible_rgb, cartesian_position_state,
                                       rand_inds=None, seg_mask=None):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: An array of shape (ncam, 3, H, W)
        visible_rgb: An array of shape (ncam, H, W, 3)
        cartesian_position_state: A 3D array of the cartesian position of the gripper.
    """

    cur_vis_pcd = np.transpose(visible_pcd, (0, 2, 3, 1)).reshape(-1, 3) # (ncam * H * W, 3)
    cur_vis_rgb = np.transpose(visible_rgb, (0, 1, 2, 3)).reshape(-1, 3)#[..., ::-1] # (ncam * H * W, 3)
    if rand_inds is None:
        rand_inds = np.random.choice(cur_vis_pcd.shape[0], 20000, replace=False)
        mask = (
                (cur_vis_pcd[rand_inds, 2] >= -0.1) &
                (cur_vis_pcd[rand_inds, 2] <= 0.6) &
                (cur_vis_pcd[rand_inds, 1] >= -0.6) &
                (cur_vis_pcd[rand_inds, 1] <= 0.6) &
                (cur_vis_pcd[rand_inds, 0] >= -0.1) &
                (cur_vis_pcd[rand_inds, 0] <= 1.2)
            )
        rand_inds = rand_inds[mask]
        # if seg_mask is not None:
        #     mask = seg_mask[0].flatten()[rand_inds] > 1
        # else:
        #     mask = (
        #         (cur_vis_pcd[rand_inds, 2] >= -0.0) &
        #         # (cur_vis_pcd[rand_inds, 1] >= -1) &
        #         (cur_vis_pcd[rand_inds, 1] <= 5) &
        #         # (cur_vis_pcd[rand_inds, 0] >= -1) &
        #         (cur_vis_pcd[rand_inds, 0] <= 5)
        #     )
        # rand_inds = rand_inds[mask]
    fig = plt.figure()
    canvas = fig.canvas
    ax = fig.add_subplot(projection='3d')
    # ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # breakpoint()
    ax.scatter(cur_vis_pcd[rand_inds, 0],
               cur_vis_pcd[rand_inds, 1],
               cur_vis_pcd[rand_inds, 2],
               c=np.clip(cur_vis_rgb[rand_inds].astype(float) / 255, 0, 1), s=15)
    # mask = seg_mask[0].flatten()[rand_inds] > 1
    # ax.scatter(cur_vis_pcd[rand_inds[mask], 0],
    #            cur_vis_pcd[rand_inds[mask], 1],
    #            cur_vis_pcd[rand_inds[mask], 2],
    #            c=cur_vis_rgb[rand_inds[mask]], s=1)

    # plot the gripper pose
    ax.scatter(cartesian_position_state[0], cartesian_position_state[1], cartesian_position_state[2], c='b', s=100)
    # plot the origin
    ax.scatter(0, 0, 0, c='g', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim([-0.9, 0.9])
    ax.set_xlim([0, 1.5])
    ax.set_zlim([-0.2, 0.7])

    fig.tight_layout()
    # make an interactive 3d plot
    # plt.show()

    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                          [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
    # for elev, azim in zip([10], [0]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        ax.set_ylim([-0.9, 0.9])
        ax.set_xlim([0, 1.5])
        ax.set_zlim([-0.2, 0.7])
        # add axes label
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110] # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1)
    ], axis=0)
    
    plt.close(fig)
    return images


def process_traj_folder(traj_folder_path):

    # find ".json" file in the folder
    # if file.endswith("errors.json") skip
    if len([file for file in os.listdir(traj_folder_path) if file.endswith("errors.json")]) == 0:
        pass
    json_path = [os.path.join(traj_folder_path, file) for file in os.listdir(traj_folder_path) if file.endswith(".json") and not file.endswith("errors.json")][0]
    meta_data = json.load(open(json_path))
    trajectory_h5_path = os.path.join(traj_folder_path, "trajectory.h5")

    try:
        with h5py.File(trajectory_h5_path, 'r') as traj:
            cartesian_position_state = traj['observation']['robot_state']['cartesian_position'][()]
            wrist_cam_left_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["wrist_cam_serial"]}_left'][()]
            wrist_cam_right_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["wrist_cam_serial"]}_right'][()]
            ext1_cam_left_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext1_cam_serial"]}_left'][()]
            ext1_cam_right_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext1_cam_serial"]}_right'][()]
            ext2_cam_left_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext2_cam_serial"]}_left'][()]
            ext2_cam_right_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext2_cam_serial"]}_right'][()]
            wrist_baseline = np.linalg.norm((wrist_cam_left_extrinsics[:, :3] - wrist_cam_right_extrinsics[:, :3]).mean(axis=0))
            ext1_baseline = np.linalg.norm((ext1_cam_left_extrinsics[:, :3] - ext1_cam_right_extrinsics[:, :3]).mean(axis=0))
            ext2_baseline = np.linalg.norm((ext2_cam_left_extrinsics[:, :3] - ext2_cam_right_extrinsics[:, :3]).mean(axis=0))
            print("wrist baseline:", wrist_baseline)
            print("ext1 baseline:", ext1_baseline)
            print("ext2 baseline:", ext2_baseline)
    except:
        return

    serial = {"wrist": meta_data["wrist_cam_serial"], 
              "ext1": meta_data["ext1_cam_serial"], 
              "ext2": meta_data["ext2_cam_serial"]}
    baselines = {"wrist": wrist_baseline, "ext1": ext1_baseline, "ext2": ext2_baseline}
    cameras = {}
    for camera_name in ["wrist", "ext1", "ext2"]:
        cameras[camera_name] = StereoCamera(
            traj_folder_path / "recordings",
            serial[camera_name],
            baselines[camera_name]
        )

    for timestep in range(len(wrist_cam_left_extrinsics)):
        print("Timestep", timestep)

        cartesian_position_state_timestep = cartesian_position_state[timestep]
        ext1_pose_left = get_camera_extrinsic_matrix(ext1_cam_left_extrinsics[timestep])
        ext1_pose_right = get_camera_extrinsic_matrix(ext1_cam_right_extrinsics[timestep])
        ext2_pose_left = get_camera_extrinsic_matrix(ext2_cam_left_extrinsics[timestep])
        ext2_pose_right = get_camera_extrinsic_matrix(ext2_cam_right_extrinsics[timestep])
        wrist_pose_left = get_camera_extrinsic_matrix(wrist_cam_left_extrinsics[timestep])
        wrist_pose_right = get_camera_extrinsic_matrix(wrist_cam_right_extrinsics[timestep])

        extrinsics_left = {"wrist": wrist_pose_left, "ext1": ext1_pose_left, "ext2": ext2_pose_left}

        all_points_zed = []
        all_points_raft = []
        all_points_pcd = []
        all_images = []
        for camera_name in ["ext1", "ext2"]:#, "wrist"]:
            left_image, right_image, depth_image, point_cloud, raft_depth_image = cameras[camera_name].get_next_frame()
            if left_image is None:
                continue
            # if camera_name == "wrist":
            #     left_image = left_image[::-1, ::-1, ...]
            #     right_image = right_image[::-1, ::-1, ...]
            #     depth_image = depth_image[::-1, ::-1, ...]
            #     raft_depth_image = raft_depth_image[::-1, ::-1, ...]
            #     # breakpoint()
            #     point_cloud = point_cloud[::-1, ::-1, ...]
                # point_cloud = point_cloud[ :, ::-1,...]
            # cv2.imshow(f"{camera_name} left", left_image)
            # cv2.imshow(f"{camera_name} right", right_image)
            # cv2.imshow(f"{camera_name} depth", depth_image)
            # cv2.waitKey(1)
            cv2.imshow(f"{camera_name} pcd", point_cloud[:, :, 2])
            # cv2.imshow(f"{camera_name} raft depth", raft_depth_image)
            cv2.waitKey(1)
            import time
            # time.sleep(10)
            # plt.imshow(depth_image, cmap='jet')
            # plt.show()

            # convert point_cloud to 3D points in the world frame
            points_in_world_frame_homogenous_zed, points_in_camera_axes_zed = depth2cloud(depth_image, 
                                                           cameras[camera_name].left_intrinsic_mat, 
                                                           extrinsics_left[camera_name],
                                                           wrist=(camera_name == "wrist"))
            points_in_world_frame_homogenous_raft, _ = depth2cloud(raft_depth_image, 
                                                           cameras[camera_name].left_intrinsic_mat, 
                                                           extrinsics_left[camera_name],
                                                           wrist=(camera_name == "wrist"))
            
            points_in_world_frame_homogenous_pcd = cam_pointcloud_to_world_pointcloud(point_cloud, extrinsics_left[camera_name])
            if camera_name == "wrist":
                print(np.nanmean((points_in_camera_axes_zed - point_cloud.transpose((2,0,1)).reshape(3, -1))))
                print(np.nanmean((points_in_world_frame_homogenous_pcd - points_in_world_frame_homogenous_zed)))
            all_points_zed.append(points_in_world_frame_homogenous_zed)
            all_points_raft.append(points_in_world_frame_homogenous_raft)
            all_points_pcd.append(points_in_world_frame_homogenous_pcd)
            all_images.append(left_image)
        all_points_zed = np.array(all_points_zed)
        all_points_raft = np.array(all_points_raft)
        all_points_pcd = np.array(all_points_pcd)
        all_points = (all_points_zed + all_points_raft) / 2
        all_images = np.array(all_images)
        # breakpoint()
        images_zed = visualize_actions_and_point_clouds(all_points_zed, all_images, cartesian_position_state_timestep[:3])
        cv2.imshow("all images zed", images_zed)
        cv2.waitKey(1)
        # images_raft = visualize_actions_and_point_clouds(all_points_raft, all_images, cartesian_position_state_timestep[:3])
        # cv2.imshow("all images raft", images_raft)
        # cv2.waitKey(1)
        # images_average = visualize_actions_and_point_clouds(all_points, all_images, cartesian_position_state_timestep[:3])
        # cv2.imshow("all images average", images_average)
        # cv2.waitKey(1)
        images_pcd = visualize_actions_and_point_clouds(all_points_pcd, all_images, cartesian_position_state_timestep[:3])
        cv2.imshow("all images pcd", images_pcd)
        cv2.waitKey(1)
    for camera in cameras.values():
        camera.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_folder_path", type=Path)
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    # parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')


    args = parser.parse_args()
    process_traj_folder(args.traj_folder_path)
    cv2.destroyAllWindows()
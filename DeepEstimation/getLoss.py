# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from PIL import Image  # using pillow-simd for increased speed
import json
import PIL.Image as pil
from utils import *
#from kitti_utils import *
from layers import *
from torchvision import transforms, datasets
import datasets
import networks
from IPython import embed

# 训练类化

class Valer:
    def __init__(self, options,image_path1,image_path2,model_path,img_ext = '.jpg'):
        self.opt = options

        self.models = {}
        self.parameters_to_train = []
        self.height = 192
        self.width = 640

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = 2
        self.num_pose_frames = 2
        # 检验id
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=Image.ANTIALIAS)

        self.disable_automasking = True
        # 获取模型 模型数组话旁边使用， 并设立训练变量
        self.use_pose_net = True

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)


        # 获取pose模型所做
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)


                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared": # 否
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn": # 否
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)

        self.load_model(model_path)

        self.loader = pil_loader
        self.inputs = {}

        def get_color(self, folder, frame_index, side, do_flip):
            color = self.loader(self.get_image_path(folder, frame_index))
            return color

        image1 = self.loader(image_path1).resize((self.width, self.height), pil.LANCZOS)
        self.inputs[("color", 0, 0)] = transforms.ToTensor()(image1).unsqueeze(0).to(self.device)
        image2 = self.loader(image_path2).resize((self.width, self.height), pil.LANCZOS)
        self.inputs[("color", 1, 0)] = transforms.ToTensor()(image2).unsqueeze(0).to(self.device)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        scale = 0
        K = self.K.copy()

        K[0, :] *= self.width // (2 ** scale) # 重点
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        self.inputs[("K", scale)] = torch.from_numpy(K).to(self.device)
        self.inputs[("inv_K", scale)] = torch.from_numpy(inv_K).to(self.device)
        '''
        self.preprocess(self.inputs)
        
        for i in [0,1]:
            del self.inputs[("color", i, -1)]
        '''
        '''
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        '''


        # 预选项加载
        #self.writers = {}
        #for mode in ["val"]:
        #    self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in [0]: # scale 的真正应用？
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(1, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(1, h, w)
            self.project_3d[scale].to(self.device)


        #self.save_opts()


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()


    def process_batch(self, inputs): # 每个batch所做
        """Pass a minibatch through the network and generate images and losses
        """
        #for key, ipt in inputs.items(): # 遍历字典,换入gpu
        #    inputs[key]= ipt.to(self.device)

        if self.opt.pose_model_type == "shared": # 不用
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color", 0, 0])
            outputs = self.models["depth"](features)

        #if self.opt.predictive_mask:
        #    outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))# feature 没使用

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses # 计算损失和输出

    def predict_poses(self, inputs, features): # 获得pose所有组合的输出
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in [0,1]}
            else:
                pose_feats = {f_i: inputs["color", f_i, 0] for f_i in [0,1]}


            pose_inputs = [pose_feats[0], pose_feats[1]]

            if self.opt.pose_model_type == "separate_resnet":
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
            elif self.opt.pose_model_type == "posecnn":
                pose_inputs = torch.cat(pose_inputs, 1)

            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, 1)] = axisangle
            outputs[("translation", 0, 1)] = translation

             # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(1 < 0))

        return outputs

    def val(self): # 验证有效性？
        """Validate the model on a single minibatch
        """
        self.set_eval()

        input = self.inputs
        with torch.no_grad():
            outputs, losses = self.process_batch(input)

            print("val loss{}", losses)
            del input, outputs, losses



    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary. 生成图片的预测重投影后的效果图
        """
        for scale in [0]:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            frame_id = 1

            if frame_id == "s":
                    T = inputs["stereo_T"] # 未用
            else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175 # 加载位移和旋转
            if self.opt.pose_model_type == "posecnn": # 暂时没用

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

            cam_points = self.backproject_depth[source_scale]( # 内参矩阵
                    depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

            outputs[("sample", frame_id, scale)] = pix_coords

            outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
            # 获得了多组输出
            #if not self.disable_automasking: #使用
            #        outputs[("color_identity", frame_id, 0)] = \
            #            inputs[("color", frame_id, 0)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images 计算再投影损失
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in [0]:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in [1]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.disable_automasking: # 重投影后在这上遮罩
                identity_reprojection_losses = []
                for frame_id in [1]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses # batch 的损失


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)


    def load_model(self, model_path):
        """Load model(s) from disk
        """
        #self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(model_path), \
            "Cannot find folder {}".format(model_path)
        print("loading model from folder {}".format(model_path))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(model_path, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def preprocess(self, inputs):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)





def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



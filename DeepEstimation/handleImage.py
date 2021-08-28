
import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import datetime

import torch
from torchvision import transforms, datasets

import DeepEstimation.networks as networks
from DeepEstimation.layers import disp_to_depth
from DeepEstimation.utils import download_model_if_doesnt_exist, Normalize


countstep = 0



class DeepEstImage:
    def __init__(self,nameit):
        """Function to predict for a single image or folder of images
        """
        self.indirectpath = os.path.join("B:\\deepEstmationFile", 'images')
        self.indirectpath = os.path.join(self.indirectpath, "2")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available() :#cuda可以做到1秒5张以上，而cpu大概是1秒1张
            self.device = torch.device("cuda")
            print("use cuda")
        else:
            self.device = torch.device("cpu")
            print("use cpu")

        model_name = "drone"

        model_path = os.path.join(os.path.dirname(__file__),"Model")
        model_path = os.path.join(model_path, model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    # FINDING INPUT IMAGES

        if os.path.isdir(self.indirectpath):
            # Searching folder for images
            #paths = glob.glob(os.path.join(indirectpath, '*.{}'.format("jpg")))
            self.output_directory = self.indirectpath
        else:
            raise Exception("Can not find args.image_path: {}".format(self.indirectpath))

    # PREDICTING ON EACH IMAGE IN TURN
        print("图片处理加载完成")
        self.imageIndex = 0



    def images_use_DE(self,image):

        with torch.no_grad():

                starttime = datetime.datetime.now()
                """
                image_path = images[0]
                images.remove(image_path)
                """

                input_image = image
                image_path = self.output_directory



                original_width, original_height = input_image.size #responses[0].height, responses[0].width))
                input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                #input_image.type(torch.DoubleTensor)

#train_label_batch = torch.from_numpy(train_label_batch)
#train_label_batch = train_label_batch.type(torch.FloatTensor)

            # PREDICTION
                input_image = input_image.to(device=self.device, dtype=torch.float)
                features = self.encoder(input_image)
                outputs = self.depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                disp, (64, 64), mode="bilinear", align_corners=False)

            # Saving numpy file
                output_name = str(self.imageIndex)
                self.imageIndex += 1
                #name_dest_npy = os.path.join(self.output_directory, "{}_disp.npy".format(output_name))
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                depth_resized = torch.nn.functional.interpolate(
                depth, (64, 64), mode="bilinear", align_corners=False)
                depth_resized = depth_resized.squeeze().cpu().numpy()

                matrix = Normalize(disp_resized_np)
                #moveit = agent.getRelaDeep2(matrix, 290, 470, picpath)
                #print("moveit {}".format(moveit))
                #move = moveit

                vmax = np.percentile(disp_resized_np, 95) # 锁掉最大的max
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) # 等比例缩放 最小无限 Normlize是用来把数据标准化(归一化)到[0,1]这个期间内,vmin是设置最小值, vmax是设置最大值
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')# mapper? cm? 归一化后配色方案
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                #im = pil.fromarray(colormapped_im)# 转图过程

                #name_dest_im = os.path.join(self.output_directory, "{}_disp.jpeg".format(output_name))
                #im.save(name_dest_im)
                endtime = datetime.datetime.now()
                internaltime = (endtime - starttime).total_seconds()
                #print("  need time {} Processed images - saved prediction to {}".format(internaltime
                #, name_dest_im))
                return  matrix,depth_resized

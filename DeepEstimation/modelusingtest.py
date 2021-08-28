# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


# -*- coding:UTF-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np   # while performing this line order , mistakes are taking place.
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import time
from cv_bridge import CvBridge,CvBridgeError
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


import torch
from torchvision import transforms, datasets
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
import DeepEstimation.networks as networks
from DeepEstimation.layers import disp_to_depth
from DeepEstimation.utils import download_model_if_doesnt_exist
file_dir = os.path.dirname(__file__)  
file_dir = os.path.join(file_dir,"TestDir")

def parse_args(): 
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', default=os.path.join(file_dir,"testkitti"))
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320",
                            "drone"
                        ],default="mono+stereo_no_pt_640x192")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

def callback(data):
     cv_image = np.ndarray(shape=(data.height, data.width, 3), dtype=np.uint8, buffer=data.data)
     input_image = pil.fromarray(cv_image,'RGB')

     original_width, original_height = input_image.size
     input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
     input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
     input_image = input_image.to(device)
     features = encoder(input_image)
     outputs = depth_decoder(features)

     disp = outputs[("disp", 0)]
     disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False) #

        # Saving numpy file
     scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
     print(depth)
        # Saving colormapped depth image
     disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
     vmax = np.percentile(disp_resized_np, 95)
     normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
     mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
     colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
     im = pil.fromarray(colormapped_im) #

     publish_image(im)

def displayImage():
    rospy.init_node('image_d400', anonymous=True)
    rospy.Subscriber('/d400/color/image_raw', Image, callback)
    rospy.spin()



def publish_image(img_data):
    image_deep = Image()
    header = Header(stamp = rospy.Time.now())
    header.frame_id = 'deep'
    image_deep.height = 480
    image_deep.width = 640
    image_deep.encoding = 'rgb8'
    image_deep.data = np.array(img_data).tostring()
    image_deep.header = header
    image_deep.step = 1241 * 3
    img_pub.publish(image_deep)

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"


    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

   
    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("Model", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval() 

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    
#    if os.path.isfile(args.image_path):
#        # Only testing on a single image
#        paths = [args.image_path]
#        output_directory = os.path.dirname(args.image_path)
#    elif os.path.isdir(args.image_path):
#        # Searching folder for images
#        print("# Searching folder for images")
#        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
#        print(paths)
#        output_directory = args.image_path
#    else:
#        raise Exception("Can not find args.image_path: {}".format(args.image_path))

#    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN 
    with torch.no_grad():
        displayImage(feed_height,feed_width)
#        for idx, image_path in enumerate(paths):

#            if image_path.endswith("_disp.jpg"):
#                # don't try to predict disparity for a disparity image!
#                continue

#            # Load image and preprocess
#            input_image = pil.open(image_path).convert('RGB')
#            1print(np.array(input_image))

#            original_width, original_height = input_image.size
#            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
#            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
#            print(input_image)

#            # PREDICTION
#            input_image = input_image.to(device)
#            features = encoder(input_image)
#            outputs = depth_decoder(features) #

#            disp = outputs[("disp", 0)]
#            disp_resized = torch.nn.functional.interpolate(
#                disp, (original_height, original_width), mode="bilinear", align_corners=False) #

#            # 1Saving numpy file
#            output_name = os.path.splitext(os.path.basename(image_path))[0]
#            #1name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
#            scaled_disp, depth = disp_to_depth(disp, 0.1, 100) #
#            #1np.save(name_dest_npy, scaled_disp.cpu().numpy()) #

#            #1if image_path.endswith("0000000117.jpg") or image_path.endswith("0000000005.jpg") or image_path.endswith("0000000469.jpg") or image_path.endswith("0000000049.jpg") or image_path.endswith("0000000033.jpg") or image_path.endswith("0000000067.jpg"):
#            print(image_path[-15:])
#            print(depth)
#            # 1Saving colormapped depth image
#            disp_resized_np = disp_resized.squeeze().cpu().numpy() # torch.squeeze()
#            vmax = np.percentile(disp_resized_np, 95) #
#            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) #
#            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') #
#            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
#            im = pil.fromarray(colormapped_im) #

#            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
#            publish_image(im)
#            im.save(name_dest_im)

#            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
#                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
#    PubImage()
    args = parse_args()
#    rospy.init_node('pub_image_d400', anonymous=True)
    img_pub = rospy.Publisher('/deepEstimation/depth/image_raw', Image, queue_size = 1)
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"


    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("Model", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")


    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)


    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        displayImage()

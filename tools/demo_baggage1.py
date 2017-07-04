# !/usr/bin/env python
#
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
#import dlib
from skimage import io
import errno

CLASSES = ('__background__', # always index 0
			'camera',
			'laptop',
			'gun',
			'gun_component',
			'knife',
			'ceramic_knife')

# CLASSES = ('__background__', # always index 0
# 			'gun')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_faster_rcnn_iter_70000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def mkdir(path):
    directory_name = os.path.basename(path)
    if not os.path.isdir(directory_name):
        # print "    ... creating a mat directory."
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
            pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--prototxt', help='Path to net prototxt')
    parser.add_argument('--caffemodel', help='Path to weights .caffemodel file')
    parser.add_argument('--modelname', help='Name of the model used.')
    parser.add_argument('--imdb', help='Path to images')
    parser.add_argument('--output', help='Path to save detected images.')

    return parser.parse_args()



args = parse_args()

#def vis_detections(im, class_name, dets, thresh=0.5):
def vis_detections(
    im, class_name, dets, thresh=0.5,
    save_image=False, image_name=None, model_name=None, net=None,
    bundle=True, fig=None, ax=None
):
    """Draw detected bounding boxes."""

    im = im[:, :, (2, 1, 0)]

    # Function to save images.
    def save_images(path, filename):
        print '\tSaving', image_name

        # Create output path
        mkdir(args.output)

        # Get image filename
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        filename = os.path.join(path, filename + ".eps")
        plt.savefig(filename, format='eps',  dpi=100)


    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        ax.imshow(im, aspect='equal')
        plt.axis('off')
        if save_image:
            save_images(args.output, image_name)
        return

    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        print(bbox, score, class_name)
    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    if save_image:
        save_images(args.output, image_name)

def demo(model, model_name, net, image_name, save=False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Create a figure to show detected objects on it.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls_val in enumerate(CLASSES[1:]):
        if cls_val not in ['gun_component']:
            continue
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls_val, dets, thresh=CONF_THRESH)
        vis_detections(
	    im=im, class_name=cls_val, dets=dets, thresh=CONF_THRESH,
            save_image=save, image_name=image_name, model_name=model_name, net=model,
            fig=fig, ax=ax
        )



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    print args

    if not os.path.isfile(args.caffemodel):
        raise IOError(('{:s} not found.\n'
                       'Please provide valid path to cafemodel file').format(args.caffemodel))

    if not os.path.isfile(args.prototxt):
        raise IOError(('{:s} not found.\n'
                       'Please provide valid path to prototxt file').format(args.prototxt))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(args.caffemodel)

    if args.imdb.endswith(".txt"):
            print("Reading images from txt file.")
            im_names = [im_name.rstrip('\n') for im_name in open(args.imdb)]

    elif os.path.isdir(args.imdb):
        print('Loading images from directory')
        im_path = args.imdb
        im_names = [os.path.join(im_path, im) for im in os.listdir(im_path) if im.endswith(('.jpg', '.png', '.PNG'))]

    else:
        raise IOError('Unknown input format')

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(args.prototxt, args.modelname, net, im_name, save=True)

    #plt.show()


































# #!/usr/bin/env python
#
# # --------------------------------------------------------
# # Faster R-CNN
# # Copyright (c) 2015 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ross Girshick
# # --------------------------------------------------------
#
# """
# Demo script showing detections in sample images.
#
# See README.md for installation instructions before running.
# """
# ##
# import _init_paths
# from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect
# from fast_rcnn.nms_wrapper import nms
# from utils.timer import Timer
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.io as sio
# from scipy.misc import imsave
# import caffe, os, sys, cv2
# import argparse
# #import dlib
# from skimage import io
# import errno
#
# CLASSES = ('__background__', # always index 0
# 			'camera',
# 			'laptop',
# 			'gun',
# 			'gun_component',
# 			'knife',
# 			'ceramic_knife')
#
# # CLASSES = ('__background__', # always index 0
# # 			'gun')
#
# NETS = {'vgg16': ('VGG16',
#                   'VGG16_faster_rcnn_final.caffemodel'),
#         'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
#                            'vgg_cnn_m_1024_faster_rcnn_iter_70000.caffemodel'),
#         'zf': ('ZF',
#                   'ZF_faster_rcnn_final.caffemodel')}
#
#
# ##
# def mkdir(path):
# 	directory_name = os.path.basename(path)
# 	if not os.path.isdir(directory_name):
# 		# print "    ... creating a mat directory."
# 		try:
# 			os.makedirs(path)
# 		except OSError as e:
# 			if e.errno != errno.EEXIST:
# 				raise e
# 			pass
#
#
# ##
# def crop(img_org):
# 	# Gray Level Image
# 	img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
# 	# Binary Image
# 	_, img_bin = cv2.threshold(img_gry, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# 	# Bounding Rectangle
# 	x, y, w, h = cv2.boundingRect(img_bin)
# 	cv2.rectangle(img_org, (x, y), (x+w, y+h), (0, 0, 0), 1)
# 	# Cropped Image
# 	img_crp = img_org[y: y+h, x: x+w]
# 	return img_crp
#
#
# # def vis_detections(im, class_name, dets, thresh=0.5):
# def vis_detections(
#     im, class_name, dets, thresh=0.5,
#     save_image=False, image_name=None, model_name=None, net=None,
#     bundle=True, fig=None, ax=None
# ):
# 	"""Draw detected bounding boxes."""
#
# 	im = im[:, :, (2, 1, 0)]
#
# 	# font_face = cv2.FONT_HERSHEY_SIMPLEX
# 	# font_scale = 0.75
# 	# thickness = 2
# 	# color = (0, 0, 150)
#
# 	inds = np.where(dets[:, -1] >= thresh)[0]
# 	if len(inds) == 0:
# 		print("\tno target detected!")
# 		ax.imshow(im, aspect='equal')
# 		plt.axis('off')
# 		return
#
# 	ax.imshow(im, aspect='equal')
# 	for i in inds:
# 		bbox = dets[i, :4].astype(int)
# 		score = dets[i, -1]
# 		# text = '{:s} {:.1f}'.format(class_name, score)
# 		#
# 		# text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
# 		# baseline += thickness
# 		#
# 		# cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
# 		# cv2.putText(im, text, (bbox[0], bbox[1]-5), font_face, font_scale, color, thickness)
#
# 		ax.add_patch(
# 		  plt.Rectangle((bbox[0], bbox[1]),
# 		                 bbox[2] - bbox[0],
# 		                 bbox[3] - bbox[1], fill=False,
# 		                 edgecolor='#ff2a2a', linewidth=3.5)
# 		   )
# 		ax.text(bbox[0], bbox[1] - 2,
# 		       '{:s} %{:.1f}'.format(class_name, score*100),
# 		       bbox=dict(facecolor="#ff2a2a", alpha=0.5),
# 		       fontsize=18, color='white')
# 		print(bbox, score, class_name)
#
# 	plt.axis('off')
# 	plt.tight_layout()
# 	plt.draw()
#
# 	# cv2.imshow('image', im)
# 	# cv2.waitKey(0)
#
# 	if save_image is True:
# 		print '\tSaving', image_name
# 		dirname = os.path.join(cfg.ROOT_DIR, 'output', 'faster_rcnn_end2end', 'demo', model_name)
# 		mkdir(dirname)
# 		label = os.path.join(dirname, os.path.basename(image_name)+".eps")
# 		# imsave(label, im[..., ::-1], format='eps')
# 		# print label
# 		plt.savefig(label, format='eps', dpi=100)
#
# 	plt.close("all")
#
#
# ##
# def demo(model, model_name, net, image_name, save=False):
# 	"""Detect object classes in an image using pre-computed object proposals."""
#
# 	##
# 	# Load the demo image
# 	im = cv2.imread(image_name)
# 	im = crop(im)
# 	# im = cv2.resize(im, (min(im.shape[:2]), min(im.shape[:2])), cv2.INTER_AREA)
# 	im = cv2.resize(im, (500, 500), cv2.INTER_AREA)
#
# 	##
# 	# Detect all object classes and regress object bounds
# 	timer = Timer()
# 	timer.tic()
# 	scores, boxes = im_detect(net, im)
#
# 	timer.toc()
# 	print ('Detection took {:.3f}s for '
# 	       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
#
# 	# Create a figure to show detected objects on it.
# 	fig, ax = plt.subplots(figsize=(6, 6))
#
# 	# Visualize detections for each class
# 	CONF_THRESH = 0.8
# 	NMS_THRESH = 0.3
# 	for cls_ind, cls_val in enumerate(CLASSES[1:]):
# 		cls_ind += 1 # because we skipped background
# 		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
# 		cls_scores = scores[:, cls_ind]
# 		dets = np.hstack((cls_boxes,
# 		                  cls_scores[:, np.newaxis])).astype(np.float32)
# 		keep = nms(dets, NMS_THRESH)
# 		dets = dets[keep, :]
# 		#vis_detections(im, cls_val, dets, thresh=CONF_THRESH)
# 		vis_detections(
# 			im=im, class_name=cls_val, dets=dets, thresh=CONF_THRESH,
# 			save_image=save, image_name=image_name, model_name=model_name, net=model,
# 			fig=fig, ax=ax
# 		)
#
#
# ##
# def parse_args():
# 	"""Parse input arguments."""
# 	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
# 	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
# 	                    default=0, type=int)
# 	parser.add_argument('--cpu', dest='cpu_mode',
# 	                    help='Use CPU mode (overrides --gpu)',
# 	                    action='store_true')
# 	parser.add_argument('--prototxt', help='Path to net prototxt')
# 	parser.add_argument('--caffemodel', help='Path to weights .caffemodel file')
# 	parser.add_argument('--modelname', help='Name of the model used.')
# 	parser.add_argument('--imdb', help='Path to images')
#
# 	args = parser.parse_args()
#
# 	return args
#
#
# ##
# class Arguments():
# 	def __init__(self):
# 		self.gpu_id = 0
# 		self.cpu_mode = False
# 		self.prototxt = "/home/sam/lib/py-faster-rcnn/models/baggage6/vgg16/test.prototxt"
# 		self.caffemodel = "/home/sam/lib/py-faster-rcnn/output/faster_rcnn_end2end/baggage6_train/vgg16.caffemodel"
# 		self.modelname = "journal"
# 		self.imdb = "/home/sam/Desktop/t/journal"
#
# ##
# if __name__ == '__main__':
# 	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
#
# 	# args = parse_args()
# 	args = Arguments()
# 	print args
#
# 	if not os.path.isfile(args.caffemodel):
# 		raise IOError(('{:s} not found.\n'
# 	                   'Please provide valid path to cafemodel file').format(args.caffemodel))
#
# 	if not os.path.isfile(args.prototxt):
# 		raise IOError(('{:s} not found.\n'
# 	                   'Please provide valid path to prototxt file').format(args.prototxt))
#
# 	if args.cpu_mode:
# 		caffe.set_mode_cpu()
# 	else:
# 		caffe.set_mode_gpu()
# 		caffe.set_device(args.gpu_id)
# 		cfg.GPU_ID = args.gpu_id
# 	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
#
# 	print '\n\nLoaded network {:s}'.format(args.caffemodel)
#
# 	if args.imdb.endswith(".txt"):
# 		print("Reading images from txt file.")
# 		im_names = [im_name.rstrip('\n') for im_name in open(args.imdb)]
#
# 	if os.path.isdir(args.imdb):
# 		im_path = args.imdb
# 		im_names = [os.path.join(im_path, im) for im in os.listdir(im_path) if im.endswith(('.jpg', '.png', '.PNG'))]
#
# 	for im_name in im_names:
# 		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
# 		print 'Demo for data/demo/{}'.format(im_name)
# 		demo(args.prototxt, args.modelname, net, im_name, save=True)
#
# 	plt.show()




























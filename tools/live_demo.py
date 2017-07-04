#!/usr/bin/env python

##
import _init_paths
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
from utils.timer import Timer
import time
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, sys, cv2
import argparse
import errno

CLASSES = ('__background__',
           'camera',
           'laptop',
           'gun',
           'gun_component',
           'knife',
           'ceramic_knife')
##
#path = '/home/sam/Desktop/tmp/inp'
path = '/mnt/vision-data/research/baggage-data/2D-xray/Durham-Scanner-E246/'
known_files = [i for i in os.listdir(path) if i.endswith('.png')]


##
def detect_file(path):
    files = os.listdir(path)
    for file in files:
        if file not in known_files:
            # Do the detection here. baggage.py
            # detect()
            print("File added: {}".format(file))
            known_files.append(file)


##
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


##
def vis_detections(
    im, class_name, dets, thresh=0.5,
    save_to=None, fig=None, ax=None
):
    """Draw detected bounding boxes."""

    im = im[:, :, (2, 1, 0)]

    # Function to save images.
    def save_image(filename):
        # Create output path
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # Get image filename
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        filename = os.path.join(path, filename + ".eps")
        plt.savefig(filename, format='eps',  dpi=100)

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        ax.imshow(im, aspect='equal')
        plt.axis('off')
        if save_to:
            save_image(save_to)
        return

    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor="#ff2a2a", linewidth=2.0)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor="#ff2a2a", alpha=0.5),
                fontsize=20, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show()

    if save_to:
        save_image(save_to)


##
def demo(net, image_name, save_to=None):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time,
                                           boxes.shape[0])

    # Create a figure to show detected objects on it.
    fig, ax = plt.subplots()

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls_val in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im=im, class_name=cls_val, dets=dets,
                       thresh=CONF_THRESH, save_to=save_to,
                       fig=fig, ax=ax)
    plt.show()
    # plt.close("all")


##
if __name__ == '__main__':
    ##
    # Arguments.
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    gpu_id = 0
    cpu_mode = False
    prototxt = "/home/sam/lib/py-faster-rcnn/models/" \
               "baggage6/vgg16/test.prototxt"
    caffemodel = "/home/sam/lib/py-faster-rcnn/output/" \
               "faster_rcnn_end2end/baggage6_train/vgg16.caffemodel"
    modelname = "journal"
    imdb = "/home/sam/Desktop/tmp/inp"
    save_to = "/home/sam/Desktop/tmp/out"

    ##
    # Load Model.
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n'
                       'Please provide valid path to cafemodel file').format(caffemodel))

    if not os.path.isfile(prototxt):
        raise IOError(('{:s} not found.\n'
                       'Please provide valid path to prototxt file').format(prototxt))

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        cfg.GPU_ID = gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)


    ##
    while True:
        filenames = [f for f in os.listdir(path) if f.endswith('color.png')]
        for filename in filenames:
            if filename not in known_files:
                print("File added: {}".format(filename))
                demo(net, os.path.join(path, filename))
                known_files.append(filename)
        time.sleep(0.5)


    ##

# FRCNN DEMO
MODEL="/home/sam/lib/py-faster-rcnn/models/baggage6/vgg16/test.prototxt"
MODELNAME=journal
WEIGHTS="/home/sam/lib/py-faster-rcnn/output/faster_rcnn_end2end/baggage6_train/vgg16.caffemodel"
PATH_TO_IMAGES="/home/sam/Desktop/t/journal"
PATH_TO_OUT="/home/sam/Desktop/t/journal/out"

./tools/demo_baggage.py			\
	--gpu 0						\
	--prototxt ${MODEL}			\
	--caffemodel ${WEIGHTS}		\
	--modelname ${MODELNAME}	\
	--imdb ${PATH_TO_IMAGES}	\
	--output ${PATH_TO_OUT}

# FRCNN DEMO
MODELNAME=vgg16
DATANAME=baggage6
MODEL="/home/sam/lib/py-faster-rcnn/models/${DATANAME}/${MODELNAME}/test.prototxt"
WEIGHTS="/home/sam/lib/py-faster-rcnn/output/faster_rcnn_end2end/${DATANAME}_train/${MODELNAME}.caffemodel"
#PATH_TO_IMAGES="/home/sam/Desktop/t/atd"
PATH_TO_IMAGES="/home/sam/projects/multi-view/demo"

PATH_TO_DEMO_DIR="/home/sam/projects/atd_bottle_electronics"
PATH_TO_INP=${PATH_TO_DEMO_DIR}/input/laptops
PATH_TO_OUT=${PATH_TO_DEMO_DIR}/output/laptops/FRCN/${MODELNAME}



./tools/atd_bottle_electronics.py		\
	--gpu 0			\
	--prototxt ${MODEL}	\
	--caffemodel ${WEIGHTS}	\
	--modelname ${MODELNAME}\
	--imdb ${PATH_TO_INP}\
	--output ${PATH_TO_OUT}

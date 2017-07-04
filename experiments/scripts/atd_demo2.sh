# FRCNN DEMO
MODELNAME=zf
DATANAME=baggage2
MODEL="/home/sam/lib/py-faster-rcnn/models/${DATANAME}/${MODELNAME}/test.prototxt"
WEIGHTS="/home/sam/lib/py-faster-rcnn/output/faster_rcnn_end2end/${DATANAME}_train/${MODELNAME}.caffemodel"
#PATH_TO_IMAGES="/home/sam/Desktop/t/atd"
PATH_TO_DEMO_DIR="/home/sam/projects/multi-view/demo"
PATH_TO_INP=${PATH_TO_DEMO_DIR}/input/guns
PATH_TO_OUT=${PATH_TO_DEMO_DIR}/output/guns/FRCN/${MODELNAME}

./tools/atd_demo2.py		\
	--gpu 0			\
	--prototxt ${MODEL}	\
	--caffemodel ${WEIGHTS}	\
	--modelname ${MODELNAME}\
	--imdb ${PATH_TO_INP}\
	--output ${PATH_TO_OUT}

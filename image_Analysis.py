import cv2
def analyse_image(img):
     
     component_names=[]
     classNames = []

     with open("coco.names", 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

     configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
     weightpath = 'frozen_inference_graph.pb'

     net = cv2.dnn_DetectionModel(weightpath, configPath)
     net.setInputSize(320 , 230)
     net.setInputScale(1.0 / 127.5)
     net.setInputMean((127.5, 127.5, 127.5))
     net.setInputSwapRB(True)

     classIds, confs, bbox = net.detect(img, confThreshold=0.5)
     for classId, confidence in zip(classIds.flatten(), confs.flatten()):
           component_name = classNames[classId - 1]
           component_names.append(component_name)

     return component_names

    
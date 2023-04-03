#Libraries
import cv2
import numpy as np  

# Load Yolo

net = cv2.dnn.readNet("Requirements\yolov3.weights", "Requirements\yolov3.cfg")

# Import the classes that the model trained on
classes = []
with open('Requirements\coco.names') as f:
    classes = [line.strip() for line in f.read().splitlines()]
    
# Get the Layers of the network and select the output layers
layer_names = net.getLayerNames()

output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("Images/beach_people.jpg")
img = cv2.resize(img, None, fx=0.8, fy=0.8)
height, width, channels = img.shape
np.random.seed(1337)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Pass the image to the net and perform the forward pass

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 3)


#Showing the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

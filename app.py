from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import base64
from io import BytesIO
import cv2
import numpy as np
from data import LABELS_LIST, SIGN_ClASS_COUNT

app = Flask(__name__)
app.run()

import urllib.request

logo = urllib.request.urlopen("https://evcpp.redrobots.xyz/wp-content/uploads/2020/06/TR6.model_.zip").read()
f = open("TR6.model", "wb")
f.write(logo)
f.close()
print("all")

# Возвращает модель для детекции
def get_model_instance_segmentation(num_classes):
    model_out = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model_out.roi_heads.box_predictor.cls_score.in_features
    model_out.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model_out


model = get_model_instance_segmentation(SIGN_ClASS_COUNT)
# на основании тестов оказалось, что тренировка этой эпохи наиболее "удачная"
model.load_state_dict(torch.load("TR6.model"))
model.eval()


@app.route('/', methods=['POST', 'GET'])
def image_work():
    if request.method == 'GET':
        return render_template('imageForm.html')
    image = request.files['image']
    image = Image.open(image)
    # защита от RGBA (срабатывает не всегда, возможна очень странная картинка)
    image_t = transforms.ToTensor()(image)[:3, :, :]
    image_t = image_t.unsqueeze(0)

    image_d = model(image_t)

    image = np.array(image)
    image = image[:, :, ::-1].copy()
    maximum = torch.max(image_d[0]["scores"])

    for i in range(len(image_d[0]["boxes"])):
        if image_d[0]["scores"][i] < (maximum - 0.3):
            continue
        rectangle = image_d[0]["boxes"][i]
        x1 = int(rectangle[0])
        y1 = int(rectangle[1])
        x2 = int(rectangle[2])
        y2 = int(rectangle[3])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        image = cv2.putText(image, LABELS_LIST[image_d[0]["labels"][i]], (x1, y1 - 10), cv2.FONT_ITALIC, 1, (0, 0, 255),
                            2)
    image = image[:, :, ::-1].copy()

    image = Image.fromarray(image.astype('uint8'), 'RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # конвертируем в BASE64 для удобства отображения
    img_str = base64.b64encode(buffered.getvalue())

    return render_template('showImage.html', image=img_str)

import tornado.web
import tornado.ioloop
from tornado.options import define, options
import numpy as np
import torch
import time
import cv2
from PIL import Image
from unet_model import UNet
from io import BytesIO

define('port', default=8080, help='运行端口', type=int)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载网络，图片单通道，分类为1。
net = UNet(n_channels=1, n_classes=1)
net.to(device=device)
net.load_state_dict(torch.load('./model/best_model.pth', map_location=device))
net.eval()


def appPredict(imgPath):
    img = Image.open(BytesIO(imgPath)).convert('RGB')
    img = cv2.cvtColor((np.asarray(img)), cv2.COLOR_RGB2GRAY)
    # 转为batch为1，通道为1，大小为512*512的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    pred = net(img_tensor)
    pred = np.array(pred.data.cpu()[0])[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    return pred


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class unetHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files['files']
        for file in files:
            start = time.time()
            filename = file['filename']
            imgPath = file['body']
            result = appPredict(imgPath)
            end = time.time()
            cv2.imwrite('result.jpg', result)
            self.write('<p>{} is ok ,cost time s : {} </p>'.format(filename, end - start))
        self.flush()


if __name__ == '__main__':
    settings = {
        'template_path': 'templates',
    }
    app = tornado.web.Application(
        [
            (r'/', MainHandler),
            (r'/unet', unetHandler),
        ], debug=True, **settings)
    app.listen(options.port)
    print('http://localhost:{}/'.format(options.port))
    tornado.ioloop.IOLoop.current().start()
import tornado.web
import tornado.ioloop
from tornado.options import define, options
import numpy as np
import torch
import os
import time
import cv2
from PIL import Image
from unet_model import UNet
from io import BytesIO


define('port', default=8080, help='运行端口', type=int)
define('address', default="127.0.0.1", help='运行端口', type=int)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载网络，图片单通道，分类为1。
net = UNet(n_channels=1, n_classes=1)
net.to(device=device)
net.load_state_dict(torch.load('./model/best_model.pth', map_location=device))
net.eval()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class unetHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files['files']
        for file in files:
            result = {}
            start = time.time()
            try:
                filename = file['filename']
                imgInfo = file['body']
                img = Image.open(BytesIO(imgInfo)).convert('RGB')
                img = cv2.cvtColor((np.asarray(img)), cv2.COLOR_RGB2GRAY)
                # 转为batch为1，通道为1，大小为512*512的数组
                img = img.reshape(1, 1, img.shape[0], img.shape[1])
                img_tensor = torch.from_numpy(img)
                img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                pred = net(img_tensor)
                pred = np.array(pred.data.cpu()[0])[0]
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                cv2.imwrite('./static/' + 'res_' + filename, pred)
                cost = time.time() - start
                cost = int(cost * 1000) / 1000
                result['imgPath'] = 'http://127.0.0.1:8000/' + 'res_' + filename
                result['costTime'] = cost
                result['info'] = 'successfully!!!'
                self.write(result)
            except Exception as e:
                cost = time.time() - start
                cost = int(cost * 1000) / 1000
                result['imgPath'] = 'http://127.0.0.1:8000/' + 'res_' + filename
                result['costTime'] = cost
                result['info'] = 'occur error!please check img'
                self.write(result)
                print(e)
        self.flush()


if __name__ == '__main__':
    settings = {
        'template_path': 'templates',
        'static_path': os.path.join(os.path.dirname(__file__), "static"),
    }
    app = tornado.web.Application(
        [
            (r'/', MainHandler),
            (r'/unet', unetHandler),
        ], **settings)
    app.listen(options.port)
    print('http://{}:{}/'.format(options.address, options.port))
    tornado.ioloop.IOLoop.current().start()



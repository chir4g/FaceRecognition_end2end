import requests
import json
import cv2

local = "http://0.0.0.0:5000"
globalhost = "http://103.212.144.244:5000"
addr = 'https://webandroidtest.herokuapp.com'
test_url = addr + '/imageprocess'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('chirag.jpeg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))

import requests
import matplotlib.pyplot as plt
import pybase64
import numpy as np
import json

# url = 'https://objects-detection-app.herokuapp.com/test/'
url = 'http://localhost:5000/test/'
data = 'https://i.dailymail.co.uk/1s/2018/11/15/22/6246344-6395507-Crickhowell_was_named_winner_of_the_Great_British_High_Street_Aw-a-106_1542321447042.jpg'

j_data = {'name':data}
r = requests.get(url = url, params = j_data)
coded_string = r.text

if coded_string is str:
    print(coded_string)
else:
    print(type(coded_string))
    print(coded_string[:100])
    # img = pybase64.b64decode(coded_string)
    # plt.figure(figsize=(10,8))
    # imgplot = plt.imshow(img)
    # plt.axis("off")
    # plt.show()
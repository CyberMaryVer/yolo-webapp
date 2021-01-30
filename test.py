import requests
import matplotlib.pyplot as plt
import numpy as np
import json

# url = 'https://objects-detection-app.herokuapp.com/test/'
url = 'http://localhost:5000/test/'
data = 'https://ericinlithuania.files.wordpress.com/2011/12/old-town-street-21.jpg'

j_data = {'name':data}
r = requests.get(url = url, params = j_data)
coded_string = r.text

if len(coded_string) < 200:
    print(json.loads(coded_string)['MESSAGE'])
else:
    print(type(coded_string))
    jsondict = json.loads(coded_string)
    print(jsondict.keys())
    img = np.array(jsondict["image"])
    img_shape = jsondict["shape"]
    print(img_shape)

    img = img.reshape(*img_shape)
    plt.figure(figsize=(10,8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
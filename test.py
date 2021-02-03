import requests
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

url0 = 'https://ericinlithuania.files.wordpress.com/2011/12/old-town-street-21.jpg'


def test_api(img_url, heroku=False):
    url_h = 'https://objects-detection-app.herokuapp.com/test/'
    url = 'http://localhost:5000/test/'

    if heroku:
        url = url_h

    j_data = {'name': url}
    r = requests.get(url=url, params=j_data)
    coded_string = r.text
    jsondict = json.loads(coded_string)

    if 'MESSAGE' in jsondict.keys():
        print(json.loads(coded_string)['MESSAGE'])
    else:
        print(jsondict.keys())
        img = np.array(jsondict["image"])
        img_shape = jsondict["shape"]
        print(img_shape)

        img = img.reshape(*img_shape)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


# test_api(url0)

def png2rgb(png, background=(255, 255, 255)):
    """Image converting in case if we get a link"""
    image_np = png
    row, col, ch = image_np.shape

    if ch == 3:
        return image_np

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2], image_np[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [overlay, np.ones((overlay.shape[0], overlay.shape[1], 1),
                              dtype=overlay.dtype) * 255], axis=2, )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def draw_m(img, center_coords=None, d0=None, thickness=4):
    h, w = img.shape[:-1]

    if d0 is None:
        d0 = int(0.04 * w)
    if center_coords is None:
        center_coords = (0 + w // 2, 0 + h // 2)

    d1 = int(2 * d0)
    d2 = int(2.8 * d0)

    pt1 = (center_coords[0] - d2, center_coords[1])
    pt2 = (center_coords[0] + d2, center_coords[1])
    pt3 = (center_coords[0], center_coords[1] - d2)
    pt4 = (center_coords[0], center_coords[1] + d2)
    color = (0, 255, 0)

    # Draw green circle
    img = cv2.circle(img, center_coords, d1, (0, 255, 0), thickness)
    img = cv2.circle(img, center_coords, d0, (0, 255, 0), thickness)
    img = cv2.line(img, pt1, pt2, (0, 255, 0), thickness)
    img = cv2.line(img, pt3, pt4, (0, 255, 0), thickness)

    return img


img = cv2.imread("templates/chess.jpg", cv2.IMREAD_UNCHANGED)
icon = cv2.imread("templates/icon-person.png", cv2.IMREAD_UNCHANGED)
icon = png2rgb(icon)
print(img.shape, icon.shape)

img = cv2.resize(img, dsize=None, fx=2, fy=2)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#######################################################
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,5)

# img = overlay_transparent(img, icon, 200, 200)
# img = draw_m(img, center_coords=(300, 500), thickness=6)
# img = draw_m(img, thickness=10)

plt.imshow(img, cmap='gray')
plt.savefig('test.jpg')
plt.show()

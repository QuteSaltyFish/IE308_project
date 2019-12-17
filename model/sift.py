import cv2
import numpy as np
from matplotlib import pyplot as plt
 
MIN_MATCH_COUNT = 10
img1 = cv2.imread('data/train_origin/BR 176 YUAN XIA F39Y_20160113_133403_image.jpg', 0)  # 查询图片
img2 = cv2.imread('data/train_noise/BR 176 YUAN XIA F39Y_20160113_133403_image.jpg', 0)  # 训练图片
 
# 初始化SIFT探测器
sift = cv2.xfeatures2d.SIFT_create()
 
# 用SIFT找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
 
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
 
'''
现在我们设置一个条件，即至少10个匹配（由MIN_MATCH_COUNT定义）将在那里以找到该对象。
 否则，只需显示一条消息，说明没有足够的匹配。
如果找到足够的匹配，我们将提取两个图像中匹配关键点的位置。
他们通过寻找这种转变。 一旦我们得到这个3x3转换矩阵，
我们就用它来将queryImage的角点转换成trainImage中相应的点。 然后我们绘制它。
'''
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
 
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
 
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
 
else:
    print("Not enough matches are found", (len(good), MIN_MATCH_COUNT))
    matchesMask = None
 
# 最后绘制内点(如果成功找到对象)或匹配关键点(如果失败)
 
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
 
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
 
plt.imshow(img3, 'gray'), plt.imsave('tmp.jpg')
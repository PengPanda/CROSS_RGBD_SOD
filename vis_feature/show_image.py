import cv2

img = cv2.imread('./vis_feature/1_fea_1.png',cv2.IMREAD_GRAYSCALE)
# heatmapshow = None
# img = cv2.normalize(img,heatmapshow,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

fea_img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
cv2.imwrite('./vis_feature/test.png',fea_img)
print('----done----')
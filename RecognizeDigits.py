import cv2
from process_image import image_refiner,predict_digit,put_label

path = "5798.jpg"
imgLoad_org = cv2.imread(path)
imgLoad = cv2.cvtColor(imgLoad_org, cv2.COLOR_BGR2GRAY)
r = cv2.selectROI(imgLoad)

imgNotRes = imgLoad[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
imgNotRes_org = imgLoad_org[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

scale = round(100 / r[3])
img = cv2.resize(imgNotRes, (r[2] * scale, r[3] * scale), interpolation=cv2.INTER_LINEAR)
img_org = cv2.resize(imgNotRes_org, (r[2] * scale, r[3] * scale), interpolation=cv2.INTER_LINEAR)

print("Selected roi: w=" + str(r[2]) + ", h=" + str(r[3]))
cv2.imshow('image',img)

ret,thresh = cv2.threshold(img,90,255,0)
cv2.imshow('thresh',thresh)

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for j, cnt in enumerate(contours):
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    hull = cv2.convexHull(cnt)
    k = cv2.isContourConvex(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    if (hierarchy[0][j][3] != -1 and w > 10 and h > 10):
        # putting boundary on each digit
        cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cropping each image and process
        roi = thresh[y:y + h, x:x + w]
        roiResized = cv2.resize(roi, (w * 3, h * 2), interpolation=cv2.INTER_LINEAR)
        roiResized = cv2.bitwise_not(roi)
        roiResized = image_refiner(roiResized)
        th, fnl = cv2.threshold(roiResized, 127, 255, cv2.THRESH_BINARY)

        # getting prediction of cropped image
        pred = predict_digit(roiResized)
        print(pred)

        # placing label on each digit
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        img_org = put_label(img_org, pred, x, y)

        cv2.imshow("contour" + str(j), roiResized)

cv2.imshow('img_detected',img_org)
cv2.waitKey(0)
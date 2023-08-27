import cv2
import numpy as np
import imutils
import easyocr
import pytesseract

chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def get_croped_image(img):
    # Grayscale and Blur 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filter and find edges for localization
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    # Find Contours and Apply Mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    return cropped_image, approx


def readtext_version1(cropped_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    return result[0][-2]


def readtext_version2(cropped_image):
    text = pytesseract.image_to_string(cropped_image, config=f'-c tessedit_char_whitelist={chars} --psm 8 --oem 3')
    return text


def readtext_version3(cropped_image):
    gray = cropped_image
    gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray, 3)

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # apply dilation 
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    # find contours
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # create copy of image
    im2 = gray.copy()

    plate_num = ""
    height, width = im2.shape
    # loop through contours and find letters in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        
        # if height of box is not a quarter of total height then skip
        if  height / float(h) > 10: continue
        ratio = h / float(w)

        # if height to width ratio is less than 1.5 skip
        if ratio < 1.0: continue

        area = h * w

        # if width is not more than 25 pixels skip
        if width / float(w) > 25: continue

        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        try:
            rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            if text[0] in chars:
                plate_num += text[0]
        except:
            continue

    return plate_num


def render_result(img, text, approx):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # add plate number to the image
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 70), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    
    return res


def get_plate_number(image_path, function='readtext_version2'):
    img = cv2.imread(image_path)
    cropped_image, approx = get_croped_image(img)

    # use function to read the text
    text = eval(function)(cropped_image)

    # get only the characters that are in the list of characters
    text = ''.join([letter for letter in text if letter in chars])

    # Render Result
    res = render_result(img, text, approx)
    
    # save the image
    cv2.imwrite('plate_image_result.jpg', res)

    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    # plt.show()
    print(text)
    return text

if __name__ == "__main__":
    get_plate_number('1.jpg')
import numpy
import easyocr
import cv2
import imutils
from matplotlib import pyplot as pl


panel_color = (96,96,96)

workField = 0
#point dict = []
def getWorkFieldValue(image) -> int:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filter = cv2.bilateralFilter(image, 11, 15, 15)
    edges = cv2.Canny(img_filter, 30, 200, )
    pl.imshow(edges)
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image)
    print(result)
    for i in range(len(result)):
        string = result[i][1]
        str_to_find = "View field: "
        if string.rfind(str_to_find) != -1:
            ret_string = ""
            for key in range(len(str_to_find), len(string)):
                if string[key] == ' ':
                    return int(ret_string);
                ret_string+=string[key]
    return -1;


def computeImage(imagepath):
    image = cv2.imread(imagepath)
    arr = cutPicture(image, panel_color)
    image_dust = arr[0]
    image_panel = arr[1]
    workField = getWorkFieldValue(image_panel)
    # Микрометров на один пиксель:
    um_per_pixel = getPixelsinNM(image_dust.shape[0], workField)
    print(workField)
    print(um_per_pixel)

    #распознавание частиц(Не то)

    gray = cv2.cvtColor(image_dust, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) #размыть
    cv2.imwrite("test.png", gray)

    edges = cv2.Canny(gray, 10, 250) #Контуры
    cv2.imwrite("edges.png", edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("closed.jpg", closed)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print(cnts)
    for c in cnts:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        #print(approx)
        if len(approx) == 4:
            cv2.drawContours(image_dust, [approx], -1, (0,255,0), 4)
            cv2.imwrite("Line.png", image_dust)


def cutPicture(image, cut_point) -> numpy.ndarray[2]:
    dimensions = image.shape
    height_cutoff = findCutLine(image, cut_point)
    s1 = image[:height_cutoff, :]
    s2 = image[height_cutoff:, :]
    arr = (s1, s2)
    return arr
    cv2.imwrite("p1.png", s1)
    cv2.imwrite("p2.png", s2)

#найти разделитель изображения
def findCutLine(photo, color_rgb) -> int:
    height = photo.shape[0]
    print(height)
    for i in range (height):
        if numpy.all(photo[i][0] == [96, 96,96]):
            return i


#Перевести в микрометры
def convertToUM(value_in_pix, um_per_pixel) -> float:
    return value_in_pix * um_per_pixel

#def createTable(headers,values):
def getPixelsinNM(width_pixels, width_nanom)-> float:
    return width_pixels/width_nanom

table_data = [ 1 , 300] #потом переложу
def fixPoint(picture, coords, size_in_pixels, table):
    cv2.putText(picture, coords, cv2.FONT_HERSHEY_COMPLEX, 1, color=(0,255,0), thickness = 2)
    table_data.insert([len(table_data), convertToUM(size_in_pixels)])

def addToTable(table):
    table;


computeImage("photos/x1000.png")

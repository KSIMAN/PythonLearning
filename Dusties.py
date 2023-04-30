import matplotlib.pyplot as plt
import numpy
import easyocr
import cv2
import imutils
from matplotlib import pyplot as pl


panel_color = (96,96,96)

table_data = [[ 1 , 300]] #потом переложу
workField = 0


#point dict = []
def getWorkFieldValue(image) -> int:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filter = cv2.bilateralFilter(image, 11, 15, 15)
    edges = cv2.Canny(img_filter, 30, 200)
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
    fixParticle(image, (30, 20), 50,um_per_pixel, 10)
    fixParticle(image, (10, 40), 10, um_per_pixel, 10)
    cv2.imwrite('image_2.jpg', image)
    createTable("image_1.jpg")

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

def fixParticle(picture, coords, size_in_pixels, um_per_pix, table):
    font = cv2.FONT_HERSHEY_COMPLEX
    size_in_um = convertToUM(size_in_pixels, um_per_pix)
    cv2.putText(picture, str(size_in_um), coords, font , 1, color=(0,255,0), thickness = 2)
    table_data.append([str(len(table_data) + 1), size_in_um])
    print(table_data)

def createTable(table_path):
    fig, ax = plt.subplots()
    table = ax.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1, 4)
    ax.axis('off')
    # display table
    plt.show()
    cv2.imwrite(table_path, plt)


computeImage("photos/x1000.png")

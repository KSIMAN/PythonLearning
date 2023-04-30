import matplotlib.pyplot as plt
import numpy
import easyocr
import os
import pandas as pd
import cv2
import imutils
from matplotlib import pyplot as pl

table_dir = "output/tables"
image_dir = "output/images"
photos_dir = ""
panel_color = (96,96,96)
panel_line = (164,164,164)
table_data = []
workField = 0

def getWorkFieldValue(image) -> int:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

# Kmeans
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = numpy.zeros([h*w,3], dtype=numpy.float32)
    count = 0
    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001),
            rounds,
            cv2.KMEANS_RANDOM_CENTERS)

    centers = numpy.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


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

    original = image_dust.copy()

    # Perform kmeans color segmentation, grayscale, Otsu's threshold
    kmeans = kmeans_color_quantization(image_dust, clusters=2)
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours, remove tiny specs using contour area filtering, gather points
    points_list = []
    size_list = []
    cnts, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    AREA_THRESHOLD = 2
    for c in cnts:
        area = cv2.contourArea(c)

        if area < AREA_THRESHOLD:
            # cv2.drawContours(thresh, [c], -1, 0, -1)
            # cv2.drawContours(thresh, [c], -1, 0, 1)
            cv2.drawContours(thresh, cnts, -1, (255, 0, 0), 3, cv2.LINE_AA, h, 1)
        else:
            (x, y), radius = cv2.minEnclosingCircle(c)

            points_list.append((int(x), int(y)))
            size_list.append(area)
            fixParticle(image_dust, (int(x), int(y)), area, um_per_pixel)

    # Apply mask onto original image
    result = cv2.bitwise_and(original, original, mask=thresh)
    result[thresh == 255] = (36, 255, 12)

    # Overlay on original
    original[thresh == 255] = (36, 255, 12)

    print("Number of particles: {}".format(len(points_list)))
    print("Average particle size: {:.3f}".format(sum(size_list) / len(size_list)))

    # Display
    cv2.imwrite('kmeans.png', kmeans)
    cv2.imwrite('original.png', original)
    cv2.imwrite('thresh.png', thresh)
    cv2.imwrite('result.png', result)
    file_name = os.path.basename(imagepath)
    image_name = os.path.splitext(file_name)[0]
    print(image_name)
    cv2.imwrite(image_dir + '/' + image_name + ".jpg", image_dust)
    createTable(image_name)

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
    for i in range ( height - 1, 0, -1):
        if numpy.all(photo[i][0] != panel_line) and numpy.all(photo[i][0] != panel_color) :
            return i


#Перевести в микрометры
def convertToUM(value_in_pix, um_per_pixel) -> float:
    return value_in_pix * um_per_pixel

#def createTable(headers,values):
def getPixelsinNM(width_pixels, width_nanom)-> float:
    return width_nanom/width_pixels

def fixParticle(picture, coords, size_in_pixels, um_per_pix):
    font = cv2.FONT_HERSHEY_COMPLEX
    size_in_um = convertToUM(size_in_pixels, um_per_pix)
    cv2.putText(picture, str(size_in_um), coords, font , 1, color=(0,255,0), thickness = 2)
    table_data.append([ size_in_um, coords])

def createTable(table_name):
    # make this example reproducible
    numpy.random.seed(0)
    # define figure and axes
    fig, ax = plt.subplots()
    # hide the axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    # create data
    df = pd.DataFrame(table_data, columns=['Размер в микрометрах', 'Координаты'])
    df.to_excel( table_dir + '/' + table_name + '.xlsx')


if not os.path.isdir("output"):
    os.mkdir("output")

if not os.path.isdir(table_dir):
    os.mkdir(table_dir)
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

print("Введите путь до папки с фотографиями: ")
file_path = input()
photos_dir = file_path
# if(file_path) если на конце палка, то убрать
with os.scandir(file_path) as files:
    for file in files:
        if file.name.find(' ') != -1:
            new_name = file.name
            new_name = new_name.replace(" ", "\\ ")
            print(new_name)
            computeImage(file_path + "/" + new_name)
            break
        computeImage(file_path + "/" + file.name)


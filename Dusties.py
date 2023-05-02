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
stat_dir = "output/stats"
photos_dir = ""
workField = 0

#разделить изображение на две части
def cutPicture(image) -> numpy.ndarray[2]:
    dimensions = image.shape
    height_cutoff = findCutLine(image)
    s1 = image[:height_cutoff, :]
    s2 = image[height_cutoff:, :]
    arr = (s1, s2)
    #cv2.imwrite("p1.png", s1)
    #cv2.imwrite("p2.png", s2)
    return arr

#найти разделяющую линию изображения
def findCutLine(photo) -> int:
    height = photo.shape[0]
    width = photo.shape[1]
    panel_color = photo[height-1][width-1]
    for i in range (height-1, 0, -1):
        if numpy.all(photo[i][width-1] == [96, 96,96]):
            continue
        return i+1

def getWorkFieldValue(image) -> float:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image)
    for i in range(len(result)):
        string = result[i][1]
        str_to_find = "View field: "
        if string.rfind(str_to_find) != -1:
            ret_string = ""
            for key in range(len(str_to_find), len(string)):
                if string[key] == ' ':
                    return float(ret_string)
                ret_string+=string[key]
    return -1

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
    image, panel = cutPicture(image)
    if numpy.all(image) == None:
        print("Error of opening file")
        return
    workField = getWorkFieldValue(panel)
    # Микрометров на один пиксель:
    um_per_pixel = getPixelsinNM(image.shape[0], workField)
    original = image.copy()

    # Perform kmeans color segmentation, grayscale, Otsu's threshold
    kmeans = kmeans_color_quantization(image, clusters=2)
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3))
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours, remove tiny specs using contour area filtering, gather points
    points_list = []
    size_list = []
    table_data = []
    cnts, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    AREA_THRESHOLD = 2

    for c in cnts:
        area = cv2.contourArea(c)

        if area < AREA_THRESHOLD:
            cv2.drawContours(thresh, cnts, -1, (255, 0, 0), 3, cv2.LINE_AA, h, 1)
        else:
            (x, y), radius = cv2.minEnclosingCircle(c)

            points_list.append((int(x), int(y)))
            size_list.append(area * um_per_pixel * um_per_pixel)
            fixParticle(image, (int(x), int(y)), area, um_per_pixel, table_data)

    # Apply mask onto original image
    result = cv2.bitwise_and(original, original, mask=thresh)
    result[thresh == 255] = (36, 255, 12)

    # Overlay on original
    original[thresh == 255] = (36, 255, 12)

    #cv2.imwrite('original.png', original)

    print("Number of particles: {}".format(len(points_list)))
    print("Average particle size: {:.3f}".format(sum(size_list) / len(size_list)))

    # Display
    #cv2.imwrite('kmeans.png', kmeans)
    #cv2.imwrite('original.png', original)
    #cv2.imwrite('thresh.png', thresh)
    #cv2.imwrite('result.png', result)

    file_name = os.path.basename(imagepath)
    image_name = os.path.splitext(file_name)[0]
    cv2.imwrite(image_dir + '/' + image_name + ".jpg", image)
    writeImageStat(image_name, points_list, size_list)
    createTable(image_name, table_data)
    
    '''
        fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(size_list, bins = 100)
    ax.grid()
    plt.show()
    '''

#Перевести в микрометры
def convertToUM(value_in_pix, um_per_pixel) -> float:
    ret = format(value_in_pix * um_per_pixel * um_per_pixel, '.3f')
    return float(ret)

#def createTable(headers,values):
def getPixelsinNM(width_pixels, width_nanom)-> float:
    return width_nanom/width_pixels

def fixParticle(picture, coords, size_in_pixels, um_per_pix, table_data):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size_in_um = convertToUM(size_in_pixels, um_per_pix)
    #cv2.putText(picture, str(size_in_um), coords, font , 1, color=(0, 255, 0), thickness = 1)
    cv2.circle(picture, 
               coords, 
               1, 
               (0, 0, 255),
            thickness=5,
            lineType=cv2.LINE_AA)
    table_data.append([ size_in_um, coords])
    #dust_size.append(size_in_um)

def createTable(table_name, table_data):
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

def writeImageStat(file_name, points_list, size_list):
    f = open(stat_dir + "/" + file_name + '.txt', '+w')
    f.write('NUMBER OF PARTICLES: ' + str(len(points_list)) + '\n')
    f.write('MAX_AREA: ' +  str(max(size_list)) + '\n')
    f.write('MIN_AREA: ' +  str(min(size_list)) + '\n')
    f.write('AVEARAGE: ' + str(sum(size_list)/len(size_list)) + '\n')
    f.write('VARIANCE: ' + str(variance_area(size_list)) + '\n')
    f.close()

def variance_area(size_list):
    mean_area = sum(size_list)/len(size_list)
    deviation_squared = []
    for i in range(len(size_list)):
        deviation_squared.append((size_list[i] - mean_area)**2)
    return sum(deviation_squared)/len(deviation_squared)


if not os.path.isdir("output"):
    os.mkdir("output")
if not os.path.isdir(table_dir):
    os.mkdir(table_dir)
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
if not os.path.isdir(stat_dir):
    os.mkdir(stat_dir)

print("Введите путь до папки с фотографиями: ")
#file_path = input()
file_path = 'photos'
photos_dir = file_path

computeImage('photos/0-5_1_5.png')
print('done')
computeImage('photos/0-5_450.png')

'''
with os.scandir(file_path) as files:
    for file in files:
        if file.name.find(' ') != -1:
            new_name = file.name
            new_name = new_name.replace(" ", "\\ ")
            computeImage(file_path + "/" + new_name)
            break
        computeImage(file_path + "/" + file.name)
        #break
'''
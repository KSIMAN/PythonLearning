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
    cv2.imwrite('image_2.jpg', image_dust)
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
    for i in range ( height - 1, 0, -1):
        if numpy.all(photo[i][0] != [96, 96,96]):
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

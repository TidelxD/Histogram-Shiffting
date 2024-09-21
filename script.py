import json
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QFileDialog, QHBoxLayout
, QDesktopWidget, QVBoxLayout, QSizePolicy, QLineEdit)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image, PngImagePlugin
import numpy as np
import math
import os

if not os.path.exists('./temp'):
    os.mkdir('./temp')


def Psnr(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calculate_ber(img1, img2):
    """Calculate Bit Error Rate (BER)"""
    original = cv2.imread(img1)
    watermarked = cv2.imread(img2)
    total_pixels = original.size
    bit_error = np.sum(original != watermarked)
    return bit_error / total_pixels


def calculate_nc(original_img, watermarked_img):
    """Calculate Normalized Correlation (NC) with overflow prevention"""
    try:
        original = cv2.imread(original_img).astype(np.float64)
        watermarked = cv2.imread(watermarked_img).astype(np.float64)
        result = np.sum(original * watermarked) / np.sqrt(np.sum(original ** 2) * np.sum(watermarked ** 2))
        return result  # /412.214
        # Rest of the code...

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


def calculate_ssim(original_img, watermarked_img):
    # Example images
    image1 = cv2.imread(original_img)
    image2 = cv2.imread(watermarked_img)

    # Compute SSIM with win_size and channel_axis if needed
    ssim_index, _ = ssim(image1, image2, win_size=3, channel_axis=None, full=True)
    print("SSIM Index:", ssim_index)
    return ssim_index


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot()

    def plot(self, x, y, color='', title='plot'):
        ax = self.figure.add_subplot(111)
        ax.bar(x, y, color=color)
        ax.set_title(title)
        self.draw()


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Histogram Shiffting UTMB'
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.left = 45
        self.top = 100
        self.size = 128
        self.maxp = 128
        self.maxp2 = 127
        self.flag = 0
        self.width = sizeObject.width() - 800
        self.height = sizeObject.height() - 400
        self.loc1 = ''
        self.extractAdress = ''
        self.initUI()

    # image convert
    def black_and_white(self, input_image_path, output_image_path='./temp/in_wm.png'):

        dim = (self.size, self.size)
        img = cv2.imread(input_image_path)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output_image_path, img)
        self.in_wm = np.array(img)
        # self.wm1.setPixmap(QPixmap('./temp/in_wm.png'))
        self.in_img(address='1')
        self.image3.setPixmap(
            QPixmap('./temp/embeded.png').scaled(280, 180, Qt.IgnoreAspectRatio, Qt.FastTransformation))

    def save_embedding_params(self ,maxp, maxp2, flag, output_path='./temp/embedding_params.json'):
        params = {
            "maxp": maxp,
            "maxp2": maxp2,
            "flag": flag
        }
        with open(output_path, 'w') as file:
            json.dump(params, file)

    def load_embedding_params(self ,param_path='./temp/embedding_params.json'):
        try:
            with open(param_path, 'r') as file:
                params = json.load(file)
            return params["maxp"], params["maxp2"], params["flag"]
        except FileNotFoundError:
            print("Error: Embedding parameters not found.")
            return None, None, None

    # image input ( Histogram shiftting if adress is empty else use embedding DAta )
    def in_img(self, address=''):
        if (address == ''):
            im = Image.open(self.loc1)
            self.image1.setPixmap(QPixmap(self.loc1).scaled(280, 180, Qt.IgnoreAspectRatio, Qt.FastTransformation))
            pix = im.load()
            height, width = im.size  # Get the width and hight of the image for iterating over
            temp = np.zeros((width, height))
            # calculating frequency of grayscale value in the image and ploting bar graph
            frequency = list()
            col = list()
            x = list()
            for i in range(256):
                frequency.append(0)
                x.append(i)
                col.append('lime')
            for i in range(height):
                for j in range(width):
                    frequency[pix[i, j]] += 1

            # finding peak point and minimum point
            maxp = 0
            for i in range(256):
                if (frequency[i] > frequency[maxp]):
                    maxp = i
            if (maxp == 0):
                maxp2 = 1
            else:
                maxp2 = 0
            for i in range(256):
                if (frequency[i] > frequency[maxp2] and frequency[i] < frequency[maxp]):
                    maxp2 = i
            print(f"maxpp2 {maxp} , {maxp2}")
            if (maxp > maxp2):
                for i in range(maxp2, maxp + 1):
                    col[i] = 'cyan'
            else:
                for i in range(maxp, maxp2 + 1):
                    col[i] = 'cyan'
            self.g1.plot(x=x, y=frequency, title='initial img', color=col)
            # finding and storing coordinates of minimum value
            self.cor = list()
            for i in range(height):
                for j in range(width):
                    if (pix[i, j] == maxp2):
                        self.cor.append([i, j])

            # histogram shifting
            self.flag = 0
            if (maxp - maxp2 == 1):
                self.flag = 1
                maxp2 -= 1
            elif (maxp - maxp2 == -1):
                maxp2 += 1
                self.flag = -1

            if (maxp < maxp2):
                for i in range(height):
                    for j in range(width):
                        if (pix[i, j] < maxp2 and pix[i, j] > maxp):
                            temp[j][i] = 255
                            pix[i, j] += 1
            else:
                for i in range(height):
                    for j in range(width):
                        if (pix[i, j] > maxp2 and pix[i, j] < maxp):
                            temp[j][i] = 255
                            pix[i, j] -= 1
            maxp2 += self.flag

            # recalculating frequency
            for i in range(256):
                frequency[i] = 0
            for i in range(height):
                for j in range(width):
                    frequency[pix[i, j]] += 1

            self.maxp = maxp
            self.maxp2 = maxp2

            # saving the image
            cv2.imwrite('./temp/shift.png', temp)
            self.g2.plot(x=x, y=frequency, color='lime', title=(
                    'after histogram shifting\n' + 'psnr=' + str(round(Psnr(self.loc1, './temp/shift.png'), 4))))
            self.image2.setPixmap(
                QPixmap('./temp/shift.png').scaled(280, 180, Qt.IgnoreAspectRatio, Qt.FastTransformation))
        else:
            embedded_img  = Image.open(self.loc1)
            pix = embedded_img.load()
            height, width = embedded_img.size  # Get the width and hight of the image for iterating over

            maxp = self.maxp
            maxp2 = self.maxp2
            # Get text from QLineEdit and convert to binary
            text = self.text_input.text()
            bstring = ''.join(format(ord(char), '08b') for char in text)
            print("bstring : " + bstring)
            #  bstring=''
            #    for i in range(self.size):
            #      for j in range(self.size):
            #          if(self.in_wm[i][j]):
            #              bstring+='1'
            #          else:
            #            bstring+='0'

            # histogram shifting
            maxp2 -= self.flag
            if (maxp < maxp2):
                for i in range(height):
                    for j in range(width):
                        if (pix[i, j] < maxp2 and pix[i, j] > maxp):
                            pix[i, j] += 1
            else:
                for i in range(height):
                    for j in range(width):
                        if (pix[i, j] > maxp2 and pix[i, j] < maxp):
                            pix[i, j] -= 1
            maxp2 += self.flag

            # writing binary data into image
            k = 0
            for i in range(height):
                for j in range(width):
                    if (pix[i, j] == maxp and k < len(bstring)):
                        if (bstring[k] == '1' and maxp2 > maxp):
                            pix[i, j] += 1
                        elif (bstring[k] == '1' and maxp2 < maxp):
                            pix[i, j] -= 1
                        k += 1

            # recalculating frequency
            frequency = list()
            x = list()
            for i in range(256):
                frequency.append(0)
                x.append(i)
            for i in range(height):
                for j in range(width):
                    frequency[pix[i, j]] += 1

            self.maxp = maxp
            self.maxp2 = maxp2
            # Inside the in_img function, after embedding the watermark, call this function:
          #  self.save_embedding_params(self.maxp, self.maxp2, self.flag)
            # saving the image
            # Set metadata for the image
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("maxp", str(self.maxp))
            metadata.add_text("maxp2", str(self.maxp2))
            metadata.add_text("flag", str(self.flag))
            # Save the image with metadata
            embedded_img.save('./temp/embeded.png', "PNG", pnginfo=metadata)
           # embedded_img.save('./temp/embeded.png', pnginfo=metadata)

            # Calculate PSNR, NC, BER, and SSIM
            psnr_value = str(round(Psnr(self.loc1, './temp/embeded.png'), 4))
            nc_value = str(round(calculate_nc(self.loc1, './temp/embeded.png'), 4))
            ber_value = str(round(calculate_ber(self.loc1, './temp/embeded.png'), 4))
            ssim_value = str(round(calculate_ssim(self.loc1, './temp/embeded.png'), 4))

            self.g3.plot(x=x, y=frequency, color='lime',
                         title=f'after writing data PSNR={psnr_value}, NC={nc_value}, \n BER={ber_value}, SSIM={ssim_value}')

    # image output ( extractin of of data and restore the originall image )

    def binary_to_text(seld,binary_data=''):
        # Split the binary data into chunks of 8 bits (1 byte)
        bytes_list = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]

        # Convert each byte to its corresponding ASCII character
        text = ''.join([chr(int(byte, 2)) for byte in bytes_list])

        return text

    def extraction_img(self):
        try:
            im = Image.open(self.extractAdress)
            pix = im.load()
            height, width = im.size  # Get the width and hight of the image for iterating over

            # extracting data
            image = np.zeros((self.size, self.size))
            # Inside the extraction_img function, before processing the image:
           # self.maxp, self.maxp2, self.flag = self.load_embedding_params()
            # Retrieve the metadata
            metadata = im.info
            self.maxp = int(metadata.get("maxp", 0))
            self.maxp2 = int(metadata.get("maxp2", 0))
            self.flag = int(metadata.get("flag", 0))
            k1 = 0
            k2 = 0
            binary_data = ''
            flag = False
            for i in range(height):
                for j in range(width):
                    if (k2 == self.size):
                        k1 += 1
                        k2 = 0
                    if (k1 == self.size):
                        flag = True
                        break
                    if (pix[i, j] == self.maxp):
                        k2 += 1
                        binary_data += '0'
                    elif (pix[i, j] == self.maxp - 1 and self.maxp2 < self.maxp):
                        image[k1, k2] = 255
                        k2 += 1
                        binary_data += '1'
                    elif (pix[i, j] == self.maxp + 1 and self.maxp2 > self.maxp):
                        image[k1, k2] = 255
                        k2 += 1
                        binary_data += '1'
                if (flag):
                    break
            cv2.imwrite('./temp/ex_wm.png', image)
            # Convert binary data to text
            extracted_text = self.binary_to_text(binary_data)
            # Display the extracted text or use it as needed
            print("Extracted Text:", extracted_text)
            self.title3_2.setText(f"Extracted Text: {extracted_text}")
            # histogram shifting
            self.maxp2 -= self.flag
            if (self.maxp < self.maxp2):
                for i in range(height):
                    for j in range(width):
                        if (pix[i, j] <= self.maxp2 and pix[i, j] > self.maxp):
                            pix[i, j] -= 1
            else:
                for i in range(height):
                    for j in range(width):
                        if (pix[i, j] >= self.maxp2 and pix[i, j] < self.maxp):
                            pix[i, j] += 1
            i = 0

            # finding and storing coordinates of minimum value
            self.cor = list()
            for i in range(height):
                for j in range(width):
                    if (pix[i, j] == self.maxp2):
                        self.cor.append([i, j])
            self.maxp2 += self.flag

            for i in self.cor:
                pix[i[0], i[1]] = self.maxp2

            frequency = list()
            x = list()
            for i in range(256):
                frequency.append(0)
                x.append(i)
            for i in range(height):
                for j in range(width):
                    frequency[pix[i, j]] += 1
            im.save('./temp/restored.png')
            self.image4.setPixmap(
                QPixmap('./temp/restored.png').scaled(280, 180, Qt.IgnoreAspectRatio, Qt.FastTransformation))
            self.g4.plot(x=x, y=frequency, color='lime',
                         title=('after restoration    ' + 'psnr=' + str(
                             round(Psnr(self.extractAdress, './temp/restored.png'), 4))))
            # saving the image
        except Exception as e:
             print(f"An unexpected error occurred: {e}")


    # event handeling
    #
    def on_click1(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')

        if fname[0]:
            self.loc1 = fname[0]
            img = cv2.imread(self.loc1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.loc1 = './temp/gray_img.png'
            cv2.imwrite(self.loc1, img)
            self.in_img()

    def on_click2(self):
        if self.loc1 != '':
            fname = QFileDialog.getOpenFileName(self, 'Open file', './')

            if fname[0]:
                # processing data
                self.black_and_white(input_image_path=fname[0])
            #  self.extraction_img()

            # displaying data
            #  self.wm2.setPixmap(QPixmap('./temp/ex_wm.png'))

    # CALCULATE NC , BEER , SSIM
    def extractionClickListener(self):
        fname = QFileDialog.getOpenFileName(self, 'extraction', './')
        if fname[0]:
            self.extractAdress = fname[0]
            self.extraction_img()

    # driver function
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # layout
        mbox = QHBoxLayout()
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()
        vbox4 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        vbox31 = QVBoxLayout()
        #  vbox32 = QVBoxLayout()

        # for vbox1
        # Open File Button
        open_file1 = QPushButton('open file', self)
        # Image one label
        self.image1 = QLabel('image1')
        open_file1.clicked.connect(self.on_click1)
        self.g1 = PlotCanvas(self)

        # for vbox2
        title2 = QLabel('\nimage difference')
        self.image2 = QLabel('image1')
        self.g2 = PlotCanvas(self)

        # for vbox3
        open_file2 = QPushButton('embedding', self)
        # Add QLineEdit for text input
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText('Enter text to embed')
        title3_1 = QLabel('embedding watermark')
        self.title3_2 = QLabel('\nextracting watermark')
        self.wm1 = QLabel('')
        self.image3 = QLabel('image3')
        self.wm2 = QLabel('')
        self.image4 = QLabel('image4')
        open_file2.clicked.connect(self.on_click2)

        self.g3 = PlotCanvas(self)
        self.g4 = PlotCanvas(self)

        # layout management
        # for vbox1 FOR ORIGINAL HOSTOGRAM
        vbox1.addWidget(open_file1)
        vbox1.addWidget(self.image1)
        vbox1.addWidget(self.g1)
        # for vbox2 FOR SHIFFTING HISTOGRAM
        vbox2.addWidget(title2)
        vbox2.addWidget(self.image2)
        vbox2.addWidget(self.g2)
        # for vbox3 FOR EMBEDDING
        vbox31.addWidget(open_file2)
        vbox31.addWidget(title3_1)
        vbox31.addWidget(self.wm1)
        vbox31.addWidget(self.image3)
        #  vbox32.addWidget(title3_2)
        #   vbox32.addWidget(self.wm2)
        #  vbox32.addWidget(self.image4)
        hbox1.addLayout(vbox31)
        #   hbox1.addLayout(vbox32)
        vbox3.addLayout(hbox1)
        vbox3.addWidget(self.g3)

        # for vbox4 FOR EXTRACTION
        extractionButton = QPushButton('extraction', self)
        extractionButton.clicked.connect(self.extractionClickListener)
        vbox4.addWidget(extractionButton)
        vbox4.addWidget(self.title3_2)
        vbox4.addWidget(self.wm2)
        vbox4.addWidget(self.image4)
        vbox4.addWidget(self.g4)
        # Add the text input to your layout (for example, in vbox3 layout)
        vbox31.addWidget(self.text_input)

        mbox.addLayout(vbox1)
        mbox.addLayout(vbox2)
        mbox.addLayout(vbox3)
        mbox.addLayout(vbox4)
        self.setLayout(mbox)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

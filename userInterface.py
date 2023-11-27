# This Python file uses the following encoding: utf-8
#import sys

import numpy as np

from PySide6.QtWidgets import QApplication, QWidget

import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

import noiseMethods
from time import perf_counter


# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.ui.calculateButton.clicked.connect(self.calculateCalled)

        self.ui.xCountSlider.sliderMoved.connect(self.autoCalculate)
        self.ui.xCountSlider.sliderMoved.connect(self.xLabelUpdate)

        self.ui.yCountSlider.sliderMoved.connect(self.autoCalculate)
        self.ui.yCountSlider.sliderMoved.connect(self.yLabelUpdate)

        self.ui.gridRandomisation.sliderMoved.connect(self.autoCalculate)
        self.ui.gridRandomisation.sliderMoved.connect(self.randomisationLabelUpdate)

        self.ui.seedSelector.valueChanged.connect(self.autoCalculate)
        self.ui.resolutionBox.valueChanged.connect(self.autoCalculate)
        self.ui.noiseSelection.currentTextChanged.connect(self.autoCalculate)

        self.ui.noiseSelection.currentTextChanged.connect(self.controlsUpdate)

        self.ui.autoCalculateTickbox.stateChanged.connect(self.calculateChanged)

        self.ui.browseButton.clicked.connect(self.fileBrowse)
        self.ui.saveButton.clicked.connect(self.saveRoutine)

        self.ui.interpolationTickBox.stateChanged.connect(self.interpolationUpdate)

        self.ui.smoothingSlider.sliderMoved.connect(self.smoothingLabelUpdate)

        self.controlsUpdate()
        self.xLabelUpdate()
        self.yLabelUpdate()
        self.randomisationLabelUpdate()
        self.interpolationUpdate()
        self.smoothingLabelUpdate()

    def interpolationUpdate(self):
        if self.ui.interpolationTickBox.isChecked():
            self.ui.iResolutionLabel.setVisible(True)
            self.ui.iResolutionSpinBox.setVisible(True)
            self.ui.smoothingLabel.setVisible(True)
            self.ui.smoothingSlider.setVisible(True)
            self.ui.smoothingSliderLabel.setVisible(True)
        else:
            self.ui.iResolutionLabel.setVisible(False)
            self.ui.iResolutionSpinBox.setVisible(False)
            self.ui.smoothingLabel.setVisible(False)
            self.ui.smoothingSlider.setVisible(False)
            self.ui.smoothingSliderLabel.setVisible(False)



    def autoCalculate(self):
        if self.ui.autoCalculateTickbox.isChecked():
            print("update forwarded")
            print(self.ui.autoCalculateTickbox.isChecked())
            self.calculateCalled()
        else:
            print("updated")

    def calculateCalled(self):
        start = perf_counter()
        print("calculate signal recived")

        QtGui.QGuiApplication.setOverrideCursor(Qt.WaitCursor)

        useGPU = self.ui.GPUTickbox.isChecked()
        xCount = int(self.ui.xCountSlider.value())
        yCount = int(self.ui.yCountSlider.value())
        seed = int(self.ui.seedSelector.value())
        resolution = int(self.ui.resolutionBox.value())
        gridRandomisation = self.ui.gridRandomisation.value()/100
        noiseType = self.ui.noiseSelection.currentText()
        print("type:", noiseType)
        print(f"xCount: {xCount}\n yCount:{yCount}\n seed: {seed} \n resolution: {resolution}")

        #self.ui.progressBar.setMinimum(0)
        #self.ui.progressBar.setMaximum(0)

        if noiseType == "Perlin Noise": #perlin noise
            self.bitmap = noiseMethods.perlinNoise(xCount, yCount, seed, resolution, useGPU, self.ui.progressBar)
        elif noiseType == "White Noise": #white noise
            self.bitmap = noiseMethods.valueGeneration(resolution, resolution, seed)
        elif noiseType == "Voronoi Noise": #voronoi noise
            self.bitmap = noiseMethods.cellNoise(xCount, yCount, seed, resolution, gridRandomisation, 0, self.ui.progressBar)
        elif noiseType == "Worley Noise": #worley noise
            self.bitmap = noiseMethods.cellNoise(xCount, yCount, seed, resolution, gridRandomisation, 1, self.ui.progressBar)
        else:
            print(noiseType, " not recognised")



        if self.ui.interpolationTickBox.isChecked():
            #upscale
            interpolatedResolution = self.ui.iResolutionSpinBox.value()
            smoothingFactor = self.ui.smoothingSlider.value()/100

            self.interpolatedBitmap = noiseMethods.interpolation(self.bitmap, resolution, interpolatedResolution, smoothingFactor)
            rgbMap = np.uint8(self.interpolatedBitmap * 255)
            qImg = QImage(rgbMap.data, interpolatedResolution, interpolatedResolution, interpolatedResolution, QImage.Format_Grayscale8)


        else:
            rgbMap = np.uint8(self.bitmap * 255)
            qImg = QImage(rgbMap.data, resolution, resolution, resolution, QImage.Format_Grayscale8)

        #Format_RGB32
        #Grayscale16

        qPix = QtGui.QPixmap.fromImage(qImg)

        self.ui.previewWindow.setPixmap(qPix)

        QtGui.QGuiApplication.setOverrideCursor(Qt.ArrowCursor)

        print("done")
        end = perf_counter()

        timeTaken = round((end - start), 3)
        if timeTaken < 1:
            timeTaken *= 1000
            self.ui.processTimeLabel.setText(f"Processed in: {timeTaken}ms")
        else:
            self.ui.processTimeLabel.setText(f"Processed in: {timeTaken}s")
        #print("Took ", round((end - start), 3), "s")




    def controlsUpdate (self):
        selection = self.ui.noiseSelection.currentText()

        if selection == "Perlin Noise": #perlin noise
            #controls needed: xCount, yCount, seed, useGPU
            self.ui.xCountSlider.setVisible(True)
            self.ui.xCountLabel.setVisible(True)
            self.ui.xSliderCounter.setVisible(True)

            self.ui.yCountSlider.setVisible(True)
            self.ui.yCountLabel.setVisible(True)
            self.ui.ySliderCounter.setVisible(True)

            self.ui.seedSelector.setVisible(True)
            self.ui.seedLabel.setVisible(True)
            self.ui.GPUTickbox.setVisible(True)
            self.ui.GPUTickbox.setEnabled(True)

            self.ui.gridRandomisation.setVisible(False)
            self.ui.gridRandomisationLabel.setVisible(False)
            self.ui.randomisationCounter.setVisible(False)


        elif selection == "White Noise": #white noise
            #controls needed: seed
            self.ui.seedSelector.setVisible(True)
            self.ui.seedLabel.setVisible(True)

            self.ui.xCountSlider.setVisible(False)
            self.ui.xCountLabel.setVisible(False)
            self.ui.xSliderCounter.setVisible(False)
            self.ui.yCountSlider.setVisible(False)
            self.ui.yCountLabel.setVisible(False)
            self.ui.ySliderCounter.setVisible(False)
            self.ui.xSliderCounter.setVisible(False)
            self.ui.GPUTickbox.setVisible(False)
            self.ui.gridRandomisation.setVisible(False)
            self.ui.gridRandomisationLabel.setVisible(False)
            self.ui.randomisationCounter.setVisible(False)

        elif selection == "Voronoi Noise": #voronoi noise
            #controls needed: xCount, yCount, seed, gridRandomisation, useGPU
            self.ui.xCountSlider.setVisible(True)
            self.ui.xCountLabel.setVisible(True)
            self.ui.xSliderCounter.setVisible(True)
            self.ui.yCountSlider.setVisible(True)
            self.ui.yCountLabel.setVisible(True)
            self.ui.ySliderCounter.setVisible(True)
            self.ui.seedSelector.setVisible(True)
            self.ui.seedLabel.setVisible(True)
            self.ui.gridRandomisation.setVisible(True)
            self.ui.gridRandomisationLabel.setVisible(True)
            self.ui.randomisationCounter.setVisible(True)

            self.ui.GPUTickbox.setVisible(True)
            self.ui.GPUTickbox.setEnabled(False)#todo remove when gpu works
            self.ui.GPUTickbox.setChecked(False)

        elif selection == "Worley Noise": #worley noise
            #controls needed: xCount, yCount, seed, gridRandomisation, useGPU
            self.ui.xCountSlider.setVisible(True)
            self.ui.xCountLabel.setVisible(True)
            self.ui.yCountSlider.setVisible(True)
            self.ui.yCountLabel.setVisible(True)
            self.ui.seedSelector.setVisible(True)
            self.ui.seedLabel.setVisible(True)
            self.ui.gridRandomisation.setVisible(True)
            self.ui.gridRandomisationLabel.setVisible(True)
            self.ui.randomisationCounter.setVisible(True)

            self.ui.GPUTickbox.setVisible(True)
            self.ui.GPUTickbox.setEnabled(False)#todo remove when gpu works
            self.ui.GPUTickbox.setChecked(False)

    def calculateChanged(self):
        if self.ui.autoCalculateTickbox.isChecked():
            self.ui.calculateButton.setEnabled(False)
        else:
            self.ui.calculateButton.setEnabled(True)

    def fileBrowse(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, "Open File", "/path/to/default/directory", "PNG Files (*.png)")[0]
        if fileName != "":
            self.ui.filePath.setText(fileName)

    def saveRoutine(self):
        bitDepth = self.ui.bitDepthBox.currentText()


        if self.ui.interpolationTickBox.isChecked():
            resolution = self.ui.iResolutionSpinBox.value()

            if bitDepth == "8 Bit":
                rgbMap = np.uint8(self.interpolatedBitmap * 255)

                qImg = QImage(rgbMap.data, resolution, resolution, resolution, QImage.Format_Grayscale8)
            elif bitDepth == "16 Bit":
                print("16 Bit")
                rgbMap = np.uint16(self.interpolatedBitmap * 65535)

                qImg = QImage(rgbMap.data, resolution, resolution, resolution * 2, QImage.Format_Grayscale16)
        else:
            resolution = self.ui.resolutionBox.value()

            if bitDepth == "8 Bit":
                rgbMap = np.uint8(self.bitmap * 255)

                qImg = QImage(rgbMap.data, resolution, resolution, resolution, QImage.Format_Grayscale8)
            elif bitDepth == "16 Bit":
                print("16 Bit")
                rgbMap = np.uint16(self.bitmap * 65535)

                qImg = QImage(rgbMap.data, resolution, resolution, resolution * 2, QImage.Format_Grayscale16)

        saveLocation = self.ui.filePath.text()
        returnValue = qImg.save(saveLocation, "PNG")
        print(returnValue)

    def xLabelUpdate(self):
        self.ui.xSliderCounter.setText(str(self.ui.xCountSlider.value()))

    def yLabelUpdate(self):
        self.ui.ySliderCounter.setText(str(self.ui.yCountSlider.value()))

    def randomisationLabelUpdate(self):
        value = self.ui.gridRandomisation.value()
        self.ui.randomisationCounter.setText(f"{value}%")

    def smoothingLabelUpdate(self):
        value = self.ui.smoothingSlider.value()
        self.ui.smoothingSliderLabel.setText(f"{value}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())

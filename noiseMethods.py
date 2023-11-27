import numpy as np
import matplotlib.pyplot as plt
import math
from time import perf_counter
from numba import cuda
from PySide6.QtWidgets import QApplication, QWidget
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage


import os
os.environ['NUMBAPRO_LIBDEVICE'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\libdevice"
os.environ['NUMBAPRO_NVVM'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\bin\nvvm64_40_0.dll"


def lerpCPU(a : float, b : float, c : float):
    #assert c >= 0 and c <= 1
    return float((b - a) * c + a)

@cuda.jit
def lerpGPU(a : float, b : float, c : float):
    #assert c >= 0 and c <= 1
    return float((b - a) * c + a)

def smoothCPU(v):
    #assert v >= 0 and v <= 1
    outputValue = v * v * v * (6 * v * v - 15 * v + 10 )
    #assert outputValue <= 1 and outputValue >= 0
    return outputValue

@cuda.jit
def smoothGPU(v):
    #assert v >= 0 and v <= 1
    outputValue = v * v * v * (6 * v * v - 15 * v + 10 )
    #assert outputValue <= 1 and outputValue >= 0
    return outputValue

def rescale(bitmap):
    maxValue = bitmap.max()
    minValue = bitmap.min()
    #Find the lowest and highest pixel values

    valueRange = maxValue - minValue
    rescaledBitmap = (bitmap - minValue) * (1/valueRange)
    #compress or expand it so every pixel lies between 0 and 1

    return rescaledBitmap

def vectorGeneration(xCount : int, yCount : int, seed : int):
    pointVectors = np.zeros((xCount,yCount,2), dtype=float) 
    #initialise the list

    rng = np.random.default_rng(seed)
    #seed the random function

    for x in range(xCount):
        for y in range(yCount):
            angle = rng.random() * 2 * math.pi 
            #generate a value between 0 and 2pi radians

            pointVectors[x,y] = [math.cos(angle), math.sin(angle)]

            #testing functions, remove when done
            #pointVectors[x,y] = [math.sqrt(2)/2, math.sqrt(2)/2]
            #pointVectors[x,y] = [1, 0]
            #if x == 1 and y == 1:
                #pass
                #pointVectors[x,y] = [-1,0]
            #convert to cartesian and assign

    #return list of cartesian magnitudes [x magnitude, y magnitude]
    return pointVectors

def valueGeneration(xCount : int, yCount : int, seed : int):
    rng = np.random.default_rng(seed)
    pointValues = rng.random((xCount,yCount))
    return pointValues

def pointGeneration(gridRandomisation, xCount : int, yCount : int, seed : int):

    spacingX = 1 / xCount
    spacingY = 1 / yCount

    pointCoords = np.zeros((xCount,yCount,2), dtype=float) 
    #initialise the list

    for x in range(xCount):
        for y in range(yCount):
            pointCoords[x,y] = [((x+0.5) * spacingX), ((y+0.5) * spacingY)]
    #generate X points and duplicate Y times
    
    if gridRandomisation == 0:
        return pointCoords

    rng = np.random.default_rng(seed) # seed the random fucntion
    
    #randomise points
    for x in range(xCount):
        for y in range(yCount):
            pointCoords[x,y,0] += ((rng.random() * 2) - 1) * gridRandomisation * spacingX 
            pointCoords[x,y,1] += ((rng.random() * 2) - 1) * gridRandomisation * spacingY 
            #multiplying by 2 and subtracting 1 increases the span of the float generator from 0-1 to -1 - 1 

    #iterate through every coordinate and randomise XY

    return pointCoords

def NewCPUNearestPoint(xPixel : int, yPixel : int, pointCoords, xCount : int, yCount : int, resolution, pointValues):

    #print(pointValues)
    #nearestPoints uses the 0-1 coordinate system
    xPosition = xPixel/ resolution
    yPosition = yPixel / resolution
    #divide to turn pixel coordinate to grid coordinate

    spacingX = 1 / xCount
    spacingY = 1 / yCount

    gridX = int(xPosition // spacingX)
    gridY = int(yPosition // spacingY)
    #find which box the pixel is in

    searchRadius = math.sqrt((spacingX * spacingX) + (spacingY * spacingY))
    searchX = math.ceil(searchRadius/spacingX)
    searchY = math.ceil(searchRadius/spacingY)

    #print(searchX)
    #print(searchY)
    
    pointDistance = []

    for x in range(-searchX , searchX + 1):
        for y in range(-searchY , searchY + 1):
            
            testX = int(gridX + x)
            testY = int(gridY + y)
            #which grid box is being analysed

            if (testX >= 0) and (testX < xCount) and (testY >= 0) and (testY < yCount):
                point = pointCoords[testX][testY]
                #find the x and y value of the point within the box
                value = pointValues[testX][testY]
                #find the value assigned to it

                xDistance = (point[0] - xPosition)
                yDistance = (point[1] - yPosition)

                totalDistance = (xDistance * xDistance) + (yDistance * yDistance )#no need to square root it as the distance is comparative

                pointDistance.append([totalDistance, value])    
    
    #pointDistance.sort()
    sortedDistances = min(pointDistance, key = lambda x : x[0])
    #print(sortedDistances)

    #assert sortedDistances[0][0] <= sortedDistances[1][0]

    return sortedDistances #returns distance and value of the closest point


def CPUPerlinPixel(xPixel : int, yPixel : int, xCount : int, yCount : int, resolution, pointVectors):

    xPosition = (xPixel + 0.5)/ resolution
    yPosition = (yPixel + 0.5)/ resolution
    #divide to turn pixel coordinate to grid coordinate

    #assert xPosition <= 1 and xPosition >= 0
    #assert yPosition <= 1 and yPosition >= 0

    spacingX = 1 / (xCount - 1)
    spacingY = 1 / (yCount - 1)

    gridX = int(xPosition // spacingX)
    gridY = int(yPosition // spacingY)
    #find which box the pixel is in

    distanceToLeft = (xPosition / spacingX) % 1
    distanceToBottom  = (yPosition / spacingY) % 1
    #find where within the box the pixel is

    distanceToRight = distanceToLeft - 1.0
    distanceToTop = distanceToBottom - 1.0
    #derive other 2 distances

    #assert gridX <= xCount
    #assert gridY <= yCount

    #perlinpixel uses the pixel based coordinate system
    #structure of pointPositions: [x position, y position],[x magnitude, y magnitude]
    #structure of pixelPosition [x position within square, y position within square]

    #BottomLeft
    pointVectorX = pointVectors[gridX][gridY][0]
    pointVectorY = pointVectors[gridX][gridY][1]
    secondaryVectorX = distanceToLeft
    secondaryVectorY = distanceToBottom
    bottomLeftDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))

    #topleft
    pointVectorX = pointVectors[gridX][gridY + 1][0]
    pointVectorY = pointVectors[gridX][gridY + 1][1]
    secondaryVectorX = distanceToLeft
    secondaryVectorY = distanceToTop
    topLeftDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))
    
    #bottomRight
    pointVectorX = pointVectors[gridX + 1][gridY][0]
    pointVectorY = pointVectors[gridX + 1][gridY][1]
    secondaryVectorX = distanceToRight
    secondaryVectorY = distanceToBottom
    bottomRightDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))

    #topRight
    pointVectorX = pointVectors[gridX + 1][gridY + 1][0]
    pointVectorY = pointVectors[gridX + 1][gridY + 1][1]
    secondaryVectorX = distanceToRight
    secondaryVectorY = distanceToTop
    topRightDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))
    
    lerpBottom = lerpCPU(bottomLeftDotProduct, bottomRightDotProduct, smoothCPU(distanceToLeft))
    lerpTop = lerpCPU(topLeftDotProduct, topRightDotProduct, smoothCPU(distanceToLeft))
    finalValue = lerpCPU(lerpBottom, lerpTop , smoothCPU(distanceToBottom)) 
    #bilinear interpolation within the grid cell

    return finalValue

@cuda.jit
def GPUPerlinPixel(xCount, yCount, resolution, pointVectors, bitmap):
    xPixel, yPixel = cuda.grid(2)
    #returns the x and y 
    if xPixel < resolution and yPixel < resolution:
        xPosition = (xPixel + 0.5)/ resolution
        yPosition = (yPixel + 0.5)/ resolution
        #divide to turn pixel coordinate to grid coordinate

        #assert xPosition <= 1 and xPosition >= 0
        #assert yPosition <= 1 and yPosition >= 0

        spacingX = 1 / (xCount - 1)
        spacingY = 1 / (yCount - 1)

        gridX = int(xPosition // spacingX)
        gridY = int(yPosition // spacingY)
        #find which box the pixel is in

        distanceToLeft = (xPosition / spacingX) % 1
        distanceToBottom  = (yPosition / spacingY) % 1
        #find where within the box the pixel is

        distanceToRight = distanceToLeft - 1.0
        distanceToTop = distanceToBottom - 1.0
        #derive other 2 distances

        #assert gridX <= xCount
        #assert gridY <= yCount

        #perlinpixel uses the pixel based coordinate system
        #structure of pointPositions: [x position, y position],[x magnitude, y magnitude]
        #structure of pixelPosition [x position within square, y position within square]

        #BottomLeft
        pointVectorX = pointVectors[gridX][gridY][0]
        pointVectorY = pointVectors[gridX][gridY][1]
        secondaryVectorX = distanceToLeft
        secondaryVectorY = distanceToBottom
        bottomLeftDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))

        #topleft
        pointVectorX = pointVectors[gridX][gridY + 1][0]
        pointVectorY = pointVectors[gridX][gridY + 1][1]
        secondaryVectorX = distanceToLeft
        secondaryVectorY = distanceToTop
        topLeftDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))
        
        #bottomRight
        pointVectorX = pointVectors[gridX + 1][gridY][0]
        pointVectorY = pointVectors[gridX + 1][gridY][1]
        secondaryVectorX = distanceToRight
        secondaryVectorY = distanceToBottom
        bottomRightDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))

        #topRight
        pointVectorX = pointVectors[gridX + 1][gridY + 1][0]
        pointVectorY = pointVectors[gridX + 1][gridY + 1][1]
        secondaryVectorX = distanceToRight
        secondaryVectorY = distanceToTop
        topRightDotProduct = ((pointVectorX * secondaryVectorX) + (pointVectorY * secondaryVectorY))
        
        lerpBottom = lerpGPU(bottomLeftDotProduct, bottomRightDotProduct, smoothGPU(distanceToLeft))
        lerpTop = lerpGPU(topLeftDotProduct, topRightDotProduct, smoothGPU(distanceToLeft))
        finalValue = lerpGPU(lerpBottom, lerpTop , smoothGPU(distanceToBottom)) 
        #bilinear interpolation within the grid cell

        bitmap[xPixel, yPixel] = finalValue
       
def perlinNoise(xCount : int, yCount : int, seed : int, resolution : int, useGPU : bool, progressBar):
    #structure [x position, y position]
    pointVectors = vectorGeneration(xCount, yCount, seed)
    #structure [x magnitude, y magnitude]
    #print("done 1!")
    bitmap = np.zeros((resolution,resolution), dtype = float)

    

    if useGPU == False:
        #print(f"Starting perlin on CPU\nxCount:{xCount}\nyCount:{yCount}\nresolution:{resolution}")
        percentageCompleted = 0
        pixelsCompleted = 0
        progressBar.setValue(0)

        for xPixel in range(resolution):
            
            for yPixel in range(resolution):

                if ((pixelsCompleted / (resolution * resolution))*100) >= (percentageCompleted + 1):
                    percentageCompleted += 1
                    progressBar.setValue(percentageCompleted)
                    QtGui.QGuiApplication.processEvents()
                    #print(percentageCompleted)

                #print(f"Pixel: ({xPixel},{yPixel})") #debug print

                value = CPUPerlinPixel(xPixel, yPixel, xCount, yCount, resolution, pointVectors)

                #run perlin noise
                #print(f"Value of pixel: {value}") #debug print

                bitmap[xPixel, yPixel] = value

                pixelsCompleted += 1 
        progressBar.setValue(100)
    else:
        print("GPU")
        
        devicePointVectors = cuda.to_device(pointVectors)
        deviceBitmap = cuda.to_device(bitmap)
        blockWidth = 16
        blockCount = int(resolution/blockWidth) + 1

        GPUPerlinPixel[(blockCount, blockCount), (blockWidth, blockWidth)](xCount, yCount, resolution, devicePointVectors, deviceBitmap)
        cuda.synchronize()
        #wait until the previous function has finished

        bitmap = deviceBitmap.copy_to_host()
        #retrieve the gata from the device
        
    #print(percentageCompleted)
    #print(f"Max: {bitmap.max()}\nMin: {bitmap.min()}")
    print("Done calculating")
    rescaledBitmap = rescale(bitmap)
    print("Done rescaling")

    return rescaledBitmap

def cellNoise(xCount : int, yCount : int, seed : int, resolution : int, gridRandomisation : float, noiseType, progressBar):
    pointCoords = pointGeneration(gridRandomisation, xCount, yCount, seed)
    oldPointCoords = pointGeneration(0, xCount, yCount, seed)
    #structure [x position, y position]
    pointValues = valueGeneration(xCount, yCount, seed)
    #structure [value]
    
    bitmap = np.zeros((resolution,resolution), dtype = float)

    percentageCompleted = 0
    pixelsCompleted = 0
    progressBar.setValue(0)

    for xPixel in range(resolution):
        for yPixel in range(resolution):
            if ((pixelsCompleted / (resolution * resolution))*100) >= (percentageCompleted + 1):
                percentageCompleted += 1
                progressBar.setValue(percentageCompleted)
                QtGui.QGuiApplication.processEvents()
            #print(f"Pixel: ({xPixel},{yPixel})") #debug print

            closestPoint = NewCPUNearestPoint(xPixel, yPixel, pointCoords, xCount, yCount, resolution, pointValues)
            
            if noiseType == 0: # voronoi
                value = closestPoint[1]
            else: #worley
                value = closestPoint[0]

            

            bitmap[xPixel, yPixel] = value

            pixelsCompleted += 1
    progressBar.setValue(100)

    rescaledBitmap = rescale(bitmap)

    return rescaledBitmap

def whiteNoise(seed, resolution : int):
    bitmap = np.random.rand(resolution,resolution)
    return bitmap

def interpolation(bitmap, resolution : int, interpolatedResolution : int, smoothingFactor : float):
    print("interpolation started")
    interpolatedBitmap = np.zeros((interpolatedResolution,interpolatedResolution), dtype = float)

    for x in range(interpolatedResolution):
        for y in range(interpolatedResolution):
            xPos = x/interpolatedResolution 
            yPos = y/interpolatedResolution 
            #find the position on the canvas between 0 and 1

            xPosBitmap = xPos * (resolution - 1) 
            yPosBitmap = yPos * (resolution - 1) 
            #find position as float on original bitmap

            topLeftPixel = bitmap[math.floor(xPosBitmap),math.ceil(yPosBitmap)]
            bottomLeftPixel = bitmap[math.floor(xPosBitmap),math.floor(yPosBitmap)]
            topRightPixel = bitmap[math.ceil(xPosBitmap),math.ceil(yPosBitmap)]
            bottomRightPixel = bitmap[math.ceil(xPosBitmap),math.floor(yPosBitmap)]

            horizontalPosition = xPosBitmap % 1
            verticalPosition = yPosBitmap % 1

            if smoothingFactor != 0:
                horizontalSmoothed = smoothCPU(horizontalPosition)
                verticalSmoothed = smoothCPU(verticalPosition)

                horizontalFactor = lerpCPU(horizontalPosition, horizontalSmoothed, smoothingFactor)
                verticalFactor = lerpCPU(verticalPosition, verticalSmoothed, smoothingFactor)
            else:
                horizontalFactor = horizontalPosition
                verticalFactor = verticalPosition

            lerpBottom = lerpCPU(bottomLeftPixel, bottomRightPixel, horizontalFactor)
            lerpTop = lerpCPU(topLeftPixel, topRightPixel, horizontalFactor)
            finalValue = lerpCPU(lerpBottom, lerpTop , verticalFactor) 

            interpolatedBitmap[x][y]= finalValue
    
    return interpolatedBitmap


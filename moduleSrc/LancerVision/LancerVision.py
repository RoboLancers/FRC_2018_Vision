import libjevois as jevois
import time
import re
import numpy as np
import cv2
from datetime import datetime


####################################################################################################
# Lancer Vision - Vision Processing for the FIRST Powerup
####################################################################################################
class LancerVision:
    ####################################################################################################
    # Constructor
    def __init__(self):
        jevois.LINFO("LancerVision Constructor...")

        # Frame Index
        self.frame = 0

        # USB send frame decimation
        # Reduces send rate by this factor to limit USB bandwidth at high process rates
        self.frame_dec_factor = 6  # At 60FPS, this still delivers 10FPS to the driver

        # Tuning constants
        self.hsv_thresh_lower = np.array([0, 0, 220])
        self.hsv_thresh_upper = np.array([255, 255, 255])

        # Target Information
        self.tgtAngle = "0.0"
        self.tgtRange = "0.0"
        self.tgtAvailable = "f"

        # Timer and Variables to track statistics we care about
        self.timer = jevois.Timer("LancerVisionStat", 25, jevois.LOG_DEBUG)

        # Regex for parsing info out of the status returned from the jevois Timer class
        self.pattern = re.compile(
            '([0-9]*\.[0-9]+|[0-9]+) fps, ([0-9]*\.[0-9]+|[0-9]+)% CPU, ([0-9]*\.[0-9]+|[0-9]+)C,')

        # Tracked stats
        self.frame_rate_fps = "0"
        self.CPULoad_pct = "0"
        self.CPUTemp_C = "0"
        self.pipelineDelay_us = "0"

        # Data structure object to hold info about the present data processed from the image fram
        self.curTargets = []
        self.sortedArray = []

        jevois.LINFO("LancerVision construction Finished")

    # Process image with no image output
    def processNoUSB(self, inframe):
        # Start measuring image processing time:
        self.timer.start()

        # No targets found yet
        self.tgtAvailable = False
        self.curTargets = []

        # Capture image from camera
        inimg = inframe.getCvBGR()
        self.frame += 1

        # Mark start of pipeline time
        pipeline_start_time = datetime.now()

        ###############################################
        # Start Image Processing Pipeline
        ###############################################
        # Move the image to HSV color space
        hsv = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)

        # Create a mask of only pixels which match the HSV color space thresholds we've determined
        hsv_mask = cv2.inRange(hsv, self.hsv_thresh_lower, self.hsv_thresh_upper)

        # Erode image to remove noise if necessary.
        hsv_mask = cv2.erode(hsv_mask, None, iterations=3)
        # Dilate image to fill in gaps
        hsv_mask = cv2.dilate(hsv_mask, None, iterations=3)

        # Find all contours of the outline of shapes in that mask
        contours = cv2.findContours(hsv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[-2]

        # Filter out contours with smaller perimeters
        contours = [x for x in contours if not cv2.arcLength(x, True) < 50]

        # If we have more than 2 contours, figure out which two is the target.
        if len(contours) > 1:

            # Hold the areas of all contours
            area_array = []

            # Loop through contours and add to area array
            for i, j in enumerate(contours):
                area = cv2.contourArea(j)
                area_array.append(area)

            # Sort the area array based on area
            self.sortedArray = sorted(zip(area_array, contours), key=lambda x: x[0], reverse=True)

            # Find the nth largest contour [n-1][n] only if there is more than one contour
            first_target = self.sortedArray[0][1]
            second_target = self.sortedArray[1][1]

            # Find information on first target
            x, y, w, h = cv2.boundingRect(first_target)
            moment = cv2.moments(first_target)
            if moment["m00"] != 0:
                center_x = int(moment["m10"] / moment["m00"])
                center_y = int(moment["m01"] / moment["m00"])
                self.curTargets.append(Target(center_x, center_y, w, h))

            # Find information on second target
            x1, y1, w1, h1 = cv2.boundingRect(second_target)
            moment = cv2.moments(second_target)
            if moment["m00"] != 0:
                center_x = int(moment["m10"] / moment["m00"])
                center_y = int(moment["m01"] / moment["m00"])
                self.curTargets.append(Target(center_x, center_y, w1, h1))

            # Check aspect ratio of our target to make sure they are correct ratio
            if (1.5 / 15 <= w / h <= 3.5 / 17 and 1.5 / 15 <= w1 / h1 <= 3.5 / 17) and len(self.curTargets) > 1:
                self.tgtAvailable = True
                self.tgtAngle = (((self.curTargets[0].X + self.curTargets[1].X) / 2) / 352 * 65) - 65 / 2

        ###############################################
        # End Image Processing Pipeline
        ###############################################

        # Mark end of pipeline
        # For accuracy, Must be done as close to sending the serial data as possible
        pipeline_end_time = datetime.now() - pipeline_start_time
        self.pipelineDelay_us = pipeline_end_time.microseconds

        # Send processed data about target location and current status
        # Note the order and number of params here must match with the roboRIO code.
        # jevois.sendSerial("{{{},{},{},{},{},{},{},{}}}\n".format(self.frame,("T" if self.tgtAvailable else "F"),self.tgtAngle, self.tgtRange,self.frame_rate_fps,self.CPULoad_pct,self.CPUTemp_C,self.pipelineDelay_us))

        # Track Processor Statistics
        results = self.pattern.match(self.timer.stop())
        if results is not None:
            self.frame_rate_fps = results.group(1)
            self.CPULoad_pct = results.group(2)
            self.CPUTemp_C = results.group(3)

    # Process function with USB output
    def process(self, inframe, outframe=None):

        # Start measuring image processing time:
        self.timer.start()

        # No targets found yet
        self.tgtAvailable = False
        self.curTargets = []

        # Capture image from camera
        inimg = inframe.getCvBGR()
        self.frame += 1

        # Mark start of pipeline time
        pipeline_start_time = datetime.now()

        ###############################################
        # Start Image Processing Pipeline
        ###############################################
        # Move the image to HSV color space
        hsv = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)

        # Create a mask of only pixels which match the HSV color space thresholds we've determined
        hsv_mask = cv2.inRange(hsv, self.hsv_thresh_lower, self.hsv_thresh_upper)

        # Erode image to remove noise if necessary.
        hsv_mask = cv2.erode(hsv_mask, None, iterations=3)
        # Dilate image to fill in gaps
        hsv_mask = cv2.dilate(hsv_mask, None, iterations=3)

        # Find all contours of the outline of shapes in that mask
        contours = cv2.findContours(hsv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[-2]

        # Filter out contours with smaller perimeters
        contours = [x for x in contours if not cv2.arcLength(x, True) < 50]

        # If we have more than 2 contours, figure out which two is the target.
        if len(contours) > 1:

            # Hold the areas of all contours
            area_array = []

            # Loop through contours and add to area array
            for i, j in enumerate(contours):
                area = cv2.contourArea(j)
                area_array.append(area)

            # Sort the area array based on area
            self.sortedArray = sorted(zip(area_array, contours), key=lambda x: x[0], reverse=True)

            # Find the nth largest contour [n-1][n] only if there is more than one contour
            first_target = self.sortedArray[0][1]
            second_target = self.sortedArray[1][1]

            # Find information on first target
            x, y, w, h = cv2.boundingRect(first_target)
            moment = cv2.moments(first_target)
            if moment["m00"] != 0:
                center_x = int(moment["m10"] / moment["m00"])
                center_y = int(moment["m01"] / moment["m00"])
                self.curTargets.append(Target(center_x, center_y, w, h))

            # Find information on second target
            x1, y1, w1, h1 = cv2.boundingRect(second_target)
            moment = cv2.moments(second_target)
            if moment["m00"] != 0:
                center_x = int(moment["m10"] / moment["m00"])
                center_y = int(moment["m01"] / moment["m00"])
                self.curTargets.append(Target(center_x, center_y, w1, h1))

            # Check aspect ratio of our target to make sure they are correct ratio
            if (1.5/15 <= w/h <= 3.5/17 and 1.5/15 <= w1/h1 <= 3.5/17) and len(self.curTargets) > 1:
                self.tgtAvailable = True
                self.tgtAngle = (((self.curTargets[0].X + self.curTargets[1].X) / 2) / 352 * 65) - 65 / 2

        ###############################################
        # End Image Processing Pipeline
        ###############################################

        # Mark end of pipeline
        # For accuracy, Must be done as close to sending the serial data as possible
        pipeline_end_time = datetime.now() - pipeline_start_time
        self.pipelineDelay_us = pipeline_end_time.microseconds

        # Send processed data about target location and current status
        # Note the order and number of params here must match with the roboRIO code.
        # jevois.sendSerial("{{{},{},{},{},{},{},{},{}}}\n".format(self.frame,("T" if self.tgtAvailable else "F"),self.tgtAngle, self.tgtRange,self.frame_rate_fps,self.CPULoad_pct,self.CPUTemp_C,self.pipelineDelay_us))

        # Broadcast the frame if we have an output sink available
        if outframe is not None:
            # Even if we're connected, don't send every frame we process. This will
            # help keep our USB bandwidth usage down.
            if self.frame % self.frame_dec_factor == 0:
                # Generate a debug image of the input image, masking non-detected pixels
                outimg = inimg

                # Overlay target info if found
                if self.tgtAvailable:
                    top = int(self.curTargets[0].Y - self.curTargets[0].height / 2)
                    bottom = int(self.curTargets[0].Y + self.curTargets[0].height / 2)
                    left = int(self.curTargets[0].X - self.curTargets[0].width / 2)
                    right = int(self.curTargets[0].X + self.curTargets[0].width / 2)
                    cv2.rectangle(outimg, (right, top), (left, bottom), (0, 255, 0), 2, cv2.LINE_4)

                    top1 = int(self.curTargets[1].Y - self.curTargets[1].height / 2)
                    bottom1 = int(self.curTargets[1].Y + self.curTargets[1].height / 2)
                    left1 = int(self.curTargets[1].X - self.curTargets[1].width / 2)
                    right1 = int(self.curTargets[1].X + self.curTargets[1].width / 2)
                    cv2.rectangle(outimg, (right1, top1), (left1, bottom1), (0, 255, 0), 2, cv2.LINE_4)

                # We are done with the output, ready to send it to host over USB:
                outframe.sendCvBGR(outimg)

        # Track Processor Statistics
        results = self.pattern.match(self.timer.stop())
        if results is not None:
            self.frame_rate_fps = results.group(1)
            self.CPULoad_pct = results.group(2)
            self.CPUTemp_C = results.group(3)

    # ###################################################################################################
    # Parse a serial command forwarded to us by the JeVois Engine, return a string
    def parseSerial(self, command):

        if command.strip() == "":
            # For some reason, the jevois engine sometimes sends empty strings.
            # Just do nothing in this case.
            return ""

        jevois.LINFO("parseserial received command [{}]".format(command))
        if command == "target":
            return self.target()
        return "ERR: Unsupported command."

    # ###################################################################################################
    # Return a string that describes the custom commands we support, for the JeVois help message
    def supportedCommands(self):
        # use \n separator if your module supports several commands
        return "target - print target information"

    # ###################################################################################################
    # Internal method that gets invoked as a custom command
    def target(self):
        return "{{{},{}}}\n".format(("T" if self.tgtAvailable else "F"), self.tgtAngle)


class Target(object):
    def __init__(self, center_x, center_y, width_in, height_in):
        self.X = center_x
        self.Y = center_y
        self.width = width_in
        self.height = height_in

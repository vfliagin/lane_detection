import numpy as np
import pandas as pd
import cv2
import time

class DetectLane:

    def __call__(self, img_path):

        self.orig_img = cv2.imread(img_path)        

        gray_img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        
        gauss_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

        edges = cv2.Canny(gauss_blur, 50, 150)
        
        mask = self.create_mask(edges)
        masked_image = cv2.bitwise_and(edges, mask)

        hough_lines = cv2.HoughLinesP(masked_image, rho = 1, theta = np.pi/180, threshold = 20,
                               minLineLength = 20, maxLineGap = 500)

        processed_lines = self.lane_lines(self.orig_img, hough_lines)
        
        intersection_point = self.line_intersection(processed_lines[0], processed_lines[1])

        result = self.draw_lane_lines(self.orig_img, processed_lines)
        result = cv2.circle(result, intersection_point, radius=7, color=(0, 255, 0), thickness=-1)
        
        return result

    def create_mask(self, img):
        """Создание маски для выделения части изображения с дорогой"""
        
        mask = np.zeros_like(img)   

        ignore_mask_color = 255
        
        rows, cols = img.shape[:2]
        
        bottom_left  = [cols * 0.1, rows * 0.95]
        bottom_right = [cols * 1, rows * 0.95]
        top_left     = [cols * 0.475, rows * 0.35]
        top_right    = [cols * 0.62, rows * 0.35]
        
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        return mask

    def average_lines(self, lines):
        """
        Нахождение среднего из линий Хафа, 
        разделенных на левую/правую полосу по наклону
        """
        
        left_lines    = []
        right_lines   = [] 
     
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue

            pol_line = np.polyfit((x1,x2), (y1,y2), 1)
            slope = pol_line[0]
            intercept = pol_line[1]

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))
    
        left_lane  = np.average(left_lines, axis=0)
        right_lane = np.average(right_lines, axis=0)

        return left_lane, right_lane

    def to_points(self, y1, y2, line):
        """
        Нахождение координат концов линии 
        """
        
        if line is None:
            return None
        
        slope, intercept = line
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        
        return ((x1, y1), (x2, y2))

    def lane_lines(self, image, lines):
        """
        Создание линий
        """
        
        left_lane, right_lane = self.average_lines(lines)

        y1 = image.shape[0]
        y2 = y1 * 0.4
        left_line  = self.to_points(y1, y2, left_lane)
        right_line = self.to_points(y1, y2, right_lane)
        
        return left_line, right_line

    def line_intersection(self, line1, line2):
        """
        Нахождение пересечения прямых
        """
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        
        return int(x), int(y)

    def draw_lane_lines(self, image, lines, thickness=10):
        """
        Добавление линий на изображение
        """
        
        color = [255, 255, 255]
        
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line,  color, thickness)
        
        img_clear = cv2.addWeighted(image, 1.0, line_image, -1.0, 0.0)
        
        color = [0, 255, 0]
        
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line,  color, thickness)
        
        res = cv2.addWeighted(img_clear, 1.0, line_image, 1.0, 0.0)
        
        return res

if __name__ == '__main__':
    detector = DetectLane()
    res = detector('road1.png')
    cv2.imwrite('Lane_highlighted.png', res)

    cv2.imshow('Lane highlighted', res)
    cv2.waitKey()
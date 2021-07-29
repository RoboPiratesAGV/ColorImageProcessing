import numpy as np 
import cv2 

webcam = cv2.VideoCapture(0) 

while(1): 

    _, imageFrame = webcam.read() 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

  

    # Black

    black_lower = np.array([0, 0, 0], np.uint8) 

    black_upper = np.array([105, 105, 105], np.uint8) 

    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper) 

    # Brown

    brown_lower = np.array([19, 69, 139], np.uint8) 

    brown_upper = np.array([42, 42, 165], np.uint8) 

    brown_mask = cv2.inRange(hsvFrame, brown_lower, brown_upper) 

    # Blue 

    blue_lower = np.array([94, 80, 2], np.uint8) 

    blue_upper = np.array([120, 255, 255], np.uint8) 

    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

      

    # Morphological Transform, Dilation 

    # for each color and bitwise_and operator 

    # between imageFrame and mask determines 

    # to detect only that particular color 

    kernal = np.ones((5, 5), "uint8") 

      

    # For red color 

    black_mask = cv2.dilate(black_mask, kernal) 

    res_black = cv2.bitwise_and(imageFrame, imageFrame,  

                              mask = black_mask) 

      

    # For green color 

    brown_mask = cv2.dilate(brown_mask, kernal) 

    res_brown = cv2.bitwise_and(imageFrame, imageFrame, 

                                mask = brown_mask) 

      

    # For blue color 

    blue_mask = cv2.dilate(blue_mask, kernal) 

    res_blue = cv2.bitwise_and(imageFrame, imageFrame, 

                               mask = blue_mask) 

   

    # Creating contour to track red color 

    contours, hierarchy = cv2.findContours(black_mask, 

                                           cv2.RETR_TREE, 

                                           cv2.CHAIN_APPROX_SIMPLE) 

      

    for pic, contour in enumerate(contours): 

        area = cv2.contourArea(contour) 

        if(area > 300): 

            x, y, w, h = cv2.boundingRect(contour) 

            imageFrame = cv2.rectangle(imageFrame, (x, y),  

                                       (x + w, y + h),  

                                       (0, 0, 0), 2) 

              

            cv2.putText(imageFrame, "Black Colour", (x, y), 

                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 

                        (0, 0, 0))     

  

    # Creating contour to track green color 

    contours, hierarchy = cv2.findContours(brown_mask, 

                                           cv2.RETR_TREE, 

                                           cv2.CHAIN_APPROX_SIMPLE) 

      

    for pic, contour in enumerate(contours): 

        area = cv2.contourArea(contour) 

        if(area > 300): 

            x, y, w, h = cv2.boundingRect(contour) 

            imageFrame = cv2.rectangle(imageFrame, (x, y),  

                                       (x + w, y + h), 

                                       (42, 42, 165), 2) 

              

            cv2.putText(imageFrame, "Brown Colour", (x, y), 

                        cv2.FONT_HERSHEY_SIMPLEX,  

                        1.0, (42, 42, 165)) 

  

    # Creating contour to track blue color 

    contours, hierarchy = cv2.findContours(blue_mask, 

                                           cv2.RETR_TREE, 

                                           cv2.CHAIN_APPROX_SIMPLE) 

    for pic, contour in enumerate(contours): 

        area = cv2.contourArea(contour) 

        if(area > 300): 

            x, y, w, h = cv2.boundingRect(contour) 

            imageFrame = cv2.rectangle(imageFrame, (x, y), 

                                       (x + w, y + h), 

                                       (255, 0, 0), 2) 

              

            cv2.putText(imageFrame, "Blue Colour", (x, y), 

                        cv2.FONT_HERSHEY_SIMPLEX, 

                        1.0, (255, 0, 0)) 

              

    # Program Termination 

    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 

    if cv2.waitKey(10) & 0xFF == ord('q'): 

        cap.release() 

        cv2.destroyAllWindows() 

        break
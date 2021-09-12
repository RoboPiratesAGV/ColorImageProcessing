import numpy as np 
import cv2 

webcam = cv2.VideoCapture(0) 

while(1): 

    _, imageFrame = webcam.read() 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

  

    # Red

    red_lower = np.array([136, 87, 111], np.uint8) 

    red_upper = np.array([180, 255, 255], np.uint8) 

    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    # Brown

    brown_lower = np.array([19, 69, 139], np.uint8) 

    brown_upper = np.array([42, 42, 165], np.uint8) 

    brown_mask = cv2.inRange(hsvFrame, brown_lower, brown_upper) 

    # Blue 

    blue_lower = np.array([94, 80, 2], np.uint8) 

    blue_upper = np.array([120, 255, 255], np.uint8) 

    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

    # Green

    green_lower = np.array([25, 52, 72], np.uint8) 

    green_upper = np.array([102, 255, 255], np.uint8) 

    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    # Blue 

    yellow_lower = np.array([22,60,200], np.uint8) 

    yellow_upper = np.array([60,255,255], np.uint8) 

    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 


      


    kernal = np.ones((5, 5), "uint8") 

      

    # For red color 

    red_mask = cv2.dilate(red_mask, kernal) 

    res_red = cv2.bitwise_and(imageFrame, imageFrame,  

                              mask = red_mask) 

      

    # For brown color 

    brown_mask = cv2.dilate(brown_mask, kernal) 

    res_brown = cv2.bitwise_and(imageFrame, imageFrame, 

                                mask = brown_mask) 


    # For green color 
    green_mask = cv2.dilate(green_mask, kernal) 

    res_green = cv2.bitwise_and(imageFrame, imageFrame, 

                                mask = green_mask) 


    # For blue color 

    blue_mask = cv2.dilate(blue_mask, kernal) 

    res_blue = cv2.bitwise_and(imageFrame, imageFrame, 

                               mask = blue_mask) 

    # For yellow color 

    yellow_mask = cv2.dilate(yellow_mask, kernal) 

    res_ = cv2.bitwise_and(imageFrame, imageFrame, 

                               mask = yellow_mask)                            

   

    # Creating contour to track red color 

    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (0, 0, 255), 2)
              
            cv2.putText(imageFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))    

                        
    # Creating contour to track green color 

    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h),
                                       (0, 255, 0), 2)
              
            cv2.putText(imageFrame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0))


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

     # Creating contour to track yellow color 
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255,255,0), 2)
            cv2.putText(imageFrame, "Yellow Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,  (255,255,0))

    #Creating contour to track brown color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 75, 150), 2)
              
            cv2.putText(imageFrame, "Brown Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 75, 150))

              

    # Program Termination 

    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 

    if cv2.waitKey(10) & 0xFF == ord('q'): 

        cap.release() 

        cv2.destroyAllWindows() 

        break

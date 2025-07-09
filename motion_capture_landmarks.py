import cv2, os, time, csv
import mediapipe as mp 
import numpy as np 

activity = 'run'
data_path = os.path.join('WISDM', activity)
all_files = os.listdir(data_path)

# Create output path
output_path = os.path.join('Motion_landmarks', activity) 
os.makedirs(output_path, exist_ok=True)

# Check and store frame nummbers
frame_counts = {}
for file in all_files:
    file_path = os.path.join(data_path, file)

    cap = cv2.VideoCapture(file_path) # Load video
    if not cap.isOpened():
        print(f"Could not open {file}")
        frame_counts[file] = None
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counts[file] = total_frames
    print(f"{file}: {total_frames} frames")

    cap.release() # work like file.close()

# Save frame counts to CSV
csv_filename = os.path.join(output_path, 'frame_counts.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'frame_count'])
    for fname, count in frame_counts.items():
        writer.writerow([fname, count])

#==============================================
    # Pose detection and landmark extraction
#==============================================
class poseDetector():
    def __init__(self, mode=False, model =1, smooth = True, detection = 0.5, tracking = 0.5):
        self.mode = mode
        self.model = model
        self.smooth = smooth
        self.detection = detection
        self.tracking = tracking
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection,
            min_tracking_confidence=self.tracking
            )

    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #method is used to convert an image from one color space to another.
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks: #if True
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS) #draw landmarks on img

        return(img)

    def getPosition(self,img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark): #it sends whole 33 landmarks
                 h, w, c = img.shape # shape of img and color
                 cx, cy = int(lm.x*w), int(lm.y*h) #convert landmark to original image size
                 lmList.append([cx,cy])
                 lm_array = np.asarray(lmList)

                 # Initialize the x and y sums to zero
                 sum_x, sum_y = 0, 0

                 # Loop over each coordinate and add its x and y values to the sums
                 for coord in lm_array:
                     sum_x += coord[0]
                     sum_y += coord[1]

                 # Calculate the centroid by dividing the sums by the number of coordinates
                 num_coords = len(lm_array)
                 centroid_x = sum_x / num_coords
                 centroid_y = sum_y / num_coords

        # Return the centroid as a tuple of (x, y) values
        return centroid_x, centroid_y


#===================================
    # Run code
#===================================

def main():

    for f in all_files:
        filename = os.path.splitext(f)[0]
        file_path = os.path.join(data_path, f)
        cap = cv2.VideoCapture(file_path) 
        detector = poseDetector()
        frame_number = frame_counts.get(f)

        # Create an empty aray for collecting centroid x,y from 33 landmarks
        data = np.zeros((frame_number, 2))

        i = 0
        pTime = time.time()  # Initialize it
        while True:
            
            success, img = cap.read()  # return condition (True or False) and image arrays
            if not success:
                print("End of video or cannot read frame.")
                break

            i+=1
            print(i)

            print(np.shape(img))
            img = detector.findPose(img)
            centroid_x, centroid_y = detector.getPosition(img) # lmlist is array shape (33,2)

            if centroid_x is None or centroid_y is None:
                centroid_x, centroid_y = np.nan, np.nan

            data[i-1] = centroid_x, centroid_y

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 5)  # show frame rate

            # Display the image with the circle
            cv2.circle(img, (round(centroid_x), round(centroid_y)), 10, (255, 0, 255), -1) # pink circle for centroid
            
            # Resize img to show in the window:
            img_resized = cv2.resize(img, (800, 600))
            cv2.imshow("Image", img_resized)
            cv2.waitKey(1)  # wait duration for next frame
            cv2.destroyAllWindows()

            if i >= frame_number:
                np.savetxt(os.path.join(output_path, f"{filename}.csv"), data, fmt='%.2f', delimiter=',')  
                break

if __name__ == "__main__":
    main()
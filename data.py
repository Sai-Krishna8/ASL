from function import *

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for action in actions:
   
        for sequence in range(no_sequences):
           
            for frame_num in range(sequence_length):

               
                frame=cv2.imread('Image/{}/{}.png'.format(action,sequence))
 
                image, results = mediapipe_detection(frame, hands)

                draw_styled_landmarks(image, results)
                
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (20,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} '.format(action), (20,15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} '.format(action), (20,15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
               
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                    
    # cap.release()
    cv2.destroyAllWindows()



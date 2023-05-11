import cv2

# Set the desired frame rate
desired_fps = 2


def vid_capture():

    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Failed to open the camera")
        exit()

    # Set the frame rate of the VideoCapture object
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('../data/video/outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:   #the frame is empty
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Close all the frames
    cv2.destroyAllWindows()





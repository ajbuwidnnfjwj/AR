import cv2 as cv
import numpy as np

def select_img_from_video(video, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
            display = img.copy()
            complete, pts = cv.findChessboardCorners(img, board_pattern)
            cv.drawChessboardCorners(display, board_pattern, pts, complete)
            cv.imshow(wnd_name, display)
            if cv.waitKey(wait_msec) == 27:
                break
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    video = cv.VideoCapture("./chessboard.MOV")
    board_pattern = (10, 7)
    board_cellsize = 0.025

    img_select = select_img_from_video(video, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    board_cellsize = 0.025
    lower = board_cellsize * np.array([[4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]])
    upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    video = cv.VideoCapture("./chessboard.MOV")
    while video.isOpened():
        valid, img = video.read()
        if not valid:
            break 

        complete, img_points = cv.findChessboardCorners(img, board_pattern)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            line_lower, _ = cv.projectPoints(lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
            for b, t in zip(line_lower, line_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            R, _ = cv.Rodrigues(rvec)
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow("AR", img)
            
            if cv.waitKey(10) == 27:
                break

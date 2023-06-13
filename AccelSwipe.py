import cv2
import os
import glob
import numpy as np
import time

frames_path = 'frame/'
frame_names = sorted(glob.glob(frames_path+"/*.png"))
print(frame_names[0])
fps = 30
file_list = []
with open('frame/list.txt', 'r') as f:
    file_list = [line.strip() for line in f.readlines()]


def create_video(mode, output_file='output.mp4'):

    frame = cv2.imread(frame_names[0])
    # frame = cv2.imread(os.path.join(frames_path, frame_names[0]))
    height, width, channels = frame.shape

    if mode == 'save':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_writer = cv2.VideoWriter(
            output_file, fourcc, fps, (1920, 1080))

    for frame_name in frame_names:
        # frame_path = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_name)
        if width == 3840:
            frame = cv2.resize(frame, (1920, 1080))

        if mode == 'save':
            video_writer.write(frame)

        elif mode == 'prev':
            cv2.imshow('preview', frame)
            cv2.waitKey(10)

    if mode == 'save':
        video_writer.release()
        print('video is saved as ', output_file)

        cv2.destroyAllWindows()


def clone_frames(frames, start_idx, end_idx, num_clones):
    for frame in frames[start_idx-1: end_idx]:
        # frame_path = os.path.join(frames_path, frame)
        frame_name, frame_ext = os.path.splitext(frame)
        for j in range(num_clones):
            clone_path = f"{frame_name}_{j}{frame_ext}"
            img = cv2.imread(frame)
            cv2.imwrite(clone_path, img)
            print('cloned : ', clone_path)


def interpolate_frame(input1, input2, alpha):
    path1 = frames_path+input1
    path2 = frames_path+input2
    print(path1)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    assert img1.shape == img2.shape

    # alpha = 0.1
    assert 0.0 <= alpha <= 1.0, "Alpha value must be between 0 and 1"

    result = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)

    filepath, extension = path1.rsplit(".", 1)
    new_path = f"{filepath}_{(alpha):.1f}.{extension}"

    result = cv2.GaussianBlur(result, (9, 9), 0)

    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    cv2.imwrite(new_path, result)
    # cv2.destroyAllWindows()


def optical_flow_interpolation(frame1, frame2, weight):

    prev = cv2.imread(frame1)
    next = cv2.imread(frame2)

    prev = cv2.resize(prev, (960, 540), cv2.INTER_AREA)
    next = cv2.resize(next, (960, 540), cv2.INTER_AREA)
    cv2.imshow('test', prev)
    cv2.waitKey(0)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=13,
                                        iterations=5,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)

    alpha = 0.5
    inter = np.zeros_like(prev)

    for y in range(prev.shape[0]):
        for x in range(prev.shape[1]):
            dx, dy = flow[y, x]
            x2 = min(max(int(x+dx), 0), prev.shape[1]-1)
            y2 = min(max(int(y+dy), 0), prev.shape[0]-1)
            inter[y, x] = (1-alpha) * prev[y, x] + alpha * next[y2, x2]

    filepath, extension = frame1.rsplit(".", 1)
    new_path = f"{filepath}_{(alpha):.1f}.{extension}"
    print(new_path)
    inter = cv2.resize(inter, (1920, 1080))
    cv2.imshow('test', inter)
    cv2.waitKey(0)
    cv2.imwrite(new_path, inter)

    return inter


def line_detect(frame1, frame2):
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    cv2.namedWindow('test')
    cv2.moveWindow('test', -1920, 0)

    cv2.imshow('test', img1)
    cv2.waitKey(0)
    cv2.imshow('test', gray1)
    cv2.waitKey(0)

    thre = 50
    thre_max = 110

    edges1 = cv2.Canny(gray1, thre, thre_max, apertureSize=3)
    edges2 = cv2.Canny(gray2, thre, thre_max, apertureSize=3)
    thresh = 100
    min = 100
    max = 10
    lines1 = cv2.HoughLinesP(edges1, rho=1, theta=np.pi /
                             180, threshold=thresh, minLineLength=min, maxLineGap=max)
    lines2 = cv2.HoughLinesP(edges2, rho=1, theta=np.pi /
                             180, threshold=thresh, minLineLength=min, maxLineGap=max)

    line_img1 = np.zeros_like(img1)
    for line in lines1:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    line_img2 = np.zeros_like(img2)
    for line in lines2:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('test', line_img1)
    cv2.waitKey(0)
    cv2.imshow('test', line_img2)
    cv2.waitKey(0)

    good_matches = []
    for i, line1 in enumerate(lines1):
        x1, y1, x2, y2 = line1[0]
        len1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        for j, line2 in enumerate(lines2):
            x1, y1, x2, y2 = line2[0]
            len2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle2 = np.arctan2(y2 - y1, x2 - x1)
            if abs(len1 - len2) < 10 and abs(angle1 - angle2) < np.pi / 36:
                good_matches.append((i, j))

    for match in good_matches:
        line1 = lines1[match[0]]
        line2 = lines2[match[1]]
        x1, y1, x2, y2 = line1[0]
        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        x1, y1, x2, y2 = line2[0]
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('test', img1)
    cv2.waitKey(0)
    cv2.imshow('test', img2)
    cv2.waitKey(0)

    # good_matches로부터 points1, points2 리스트 생성
    points1 = np.float32([lines1[i][0][:2]
                         for i, _ in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([lines2[j][0][:2]
                         for _, j in good_matches]).reshape(-1, 1, 2)

    # H 매트릭스 계산
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # 이미지 warp
    result = cv2.warpPerspective(
        img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # result[0:img1.shape[0], 0:img1.shape[1]] = img1
    result = cv2.resize(result, (1920, 1080))

    filepath, extension = frame1.rsplit(".", 1)
    new_path = f"{filepath}_mp.{extension}"
    print(new_path)

    cv2.imshow('test', result)
    cv2.waitKey(0)
    # cv2.imwrite(new_path, result)


def morphing_image(frame1, frame2):
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)
    cv2.namedWindow('test')
    cv2.moveWindow('test', -1920, 0)
    cv2.imshow('test', img1)
    cv2.waitKey(0)
    cv2.imshow('test', img2)
    cv2.waitKey(0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # mask
    center_x, center_y = w1//2, h1//2

    mask = np.zeros((h1, w1), dtype=np.uint8)
    radius = min(center_x, center_y)
    cv2.circle(mask, (center_x, center_y), radius,
               (255, 255, 255), -1, cv2.LINE_AA)
    mask = cv2.bitwise_not(mask)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, mask=mask)
    kp2, des2 = sift.detectAndCompute(img2, mask=mask)

    # keypoints
    img1_kp = cv2.drawKeypoints(
        img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(
        img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('test', img1_kp)
    cv2.waitKey()
    cv2.imshow('test', img2_kp)
    cv2.waitKey()

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:30]

    # check key
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:30], None, flags=2)
    matched_img = cv2.resize(matched_img, None, fx=0.5, fy=0.5)
    cv2.imshow('test', matched_img)
    cv2.waitKey(0)

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    result = cv2.warpPerspective(img1, M, (w2, h2))

    filepath, extension = frame1.rsplit(".", 1)
    new_path = f"{filepath}_mp.{extension}"
    print(new_path)

    cv2.imshow('test', result)
    cv2.waitKey(0)
    cv2.imwrite(new_path, result)


def absdiff(frame1, frame2):
    prev = cv2.imread(frame1)
    next = cv2.imread(frame2)

    diff = cv2.absdiff(prev, next)

    inter_view = np.uint8(prev + diff/2)
    return inter_view


mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    # global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(param, (x, y), 5, (0, 255, 0), 2)
            mouse_pts.append([x, y])
            print("Point selected: ", [x, y])


def calib_warp(frame1, frame2):
    # cv2.namedWindow('test')
    # cv2.moveWindow('test', 0, 0)
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)

    src_pts = []
    dst_pts = []
    pts_num = 4

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(src_pts) < pts_num:
                src_pts.append([x, y])
                cv2.circle(img1, (x, y), 5, (0, 255, 0), -1)
            elif len(dst_pts) < pts_num:
                dst_pts.append([x, y])
                cv2.circle(img2, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow('img1')
    cv2.setMouseCallback('img1', mouse_callback)

    while True:
        cv2.imshow('img1', img1)

        if cv2.waitKey(1) == ord('q') or len(dst_pts) == pts_num:
            break

    cv2.destroyAllWindows()

    cv2.namedWindow('img2')
    cv2.setMouseCallback('img2', mouse_callback)

    while True:
        cv2.imshow('img2', img2)

        if cv2.waitKey(1) == ord('q') or len(dst_pts) == pts_num:
            break
    cv2.destroyAllWindows()

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    print(src_pts)
    print(dst_pts)

    mid_pts = (src_pts + dst_pts)/2
    print(mid_pts)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    H_mid, _ = cv2.findHomography(src_pts, mid_pts)

    print(H)
    print(H_mid)

    # H = H+H_inv
    # print(H)
    # result = cv2.warpPerspective(img1, H_mid, (img1.shape[1], img1.shape[0]))
    # result1 = cv2.warpPerspective(
    #     img1, H_mid, (img1.shape[1], img1.shape[0]), borderMode=cv2.BORDER_REFLECT)
    result = cv2.warpPerspective(
        img1, H_mid, (img1.shape[1], img1.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite('frame-157-.png', result)
    # cv2.imwrite('BORDER_REPLICATE.png', result2)
    # cv2.imshow('test', result)
    # cv2.waitKey()
    # cv2.imshow('test', result1)
    # cv2.waitKey()
    cv2.imshow('test', result)
    cv2.waitKey()

    cv2.destroyAllWindows()


cnt = 0


def warp_3d(frame1, frame2):
    global cnt
    world_img = cv2.imread('Soccer_Half.png')

    prev = cv2.imread(frame1)
    next = cv2.imread(frame2)

    img_pts = []
    obj_pts = []

    img_chk = True

    def mouse_callback(event, x, y, flags, param):
        global cnt
        if event == cv2.EVENT_LBUTTONDOWN:
            if img_chk:
                img_pts.append([x, y])
            else:
                obj_pts.append([x, y, 0])
            cnt += 1
            print("selected point : ", cnt)

    cv2.namedWindow('click')
    cv2.setMouseCallback('click', mouse_callback)

    # while True:
    #     cv2.imshow('click', prev)

    #     if cv2.waitKey(1) == 27:
    #         break
    # print(img_pts)

    # cv2.destroyAllWindows()

    # cv2.namedWindow('click 2')
    # cv2.setMouseCallback('click 2', mouse_callback)

    # pts_num = cnt
    # print('all pts ', pts_num)
    # cnt = 0
    # img_chk = False

    # while True:
    #     cv2.imshow('click 2', world_img)

    #     if cv2.waitKey(1) == 27:
    #         break

    # # for pt in img_pts:
    # #     pt = tuple(pt.astype(np.int))
    # #     cv2.circle(prev, pt, 5, (0, 255, 0), -1)

    # obj_pts = np.array([obj_pts], dtype=np.float32)
    # img_pts = np.array([img_pts], dtype=np.float32)

    # print(obj_pts)
    # print(img_pts)
    # print(obj_pts.dtype)
    # print(obj_pts.shape)
    h, w = prev.shape[:2]
    obj_pts = np.array([[311, 710,   0], [312, 764,   0], [490, 763,   0], [490, 710,   0], [400, 655,   0], [
                       329, 602,   0], [202, 602,   0], [399, 709,  0], [369, 765,   0], [421, 765,   0]])
    img_size = (w, h)
    obj_pts = obj_pts/img_size[::-1]

    img_pts = np.array([[645, 423], [250, 476], [922, 843], [1361, 770], [1367, 518], [
                       1440, 355], [987, 187], [943, 569], [418, 570], [689, 716]])

    obj_pts = np.array([obj_pts], dtype=np.float32)
    img_pts = np.array([img_pts], dtype=np.float32)

    _, camera_mat, dist_coef, _, _ = cv2.calibrateCamera(
        [obj_pts], [img_pts], (prev.shape[1], prev.shape[0]), None, None)

    # dist_coef = np.array([0, 0, 0, 0])
    np.set_printoptions(precision=4, suppress=True)
    print('camera matrix : \n', camera_mat)
    print('distortion coef : ', dist_coef)
    # K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])

    retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_mat, dist_coef)

    warped = cv2.warpPerspective(prev, cv2.Rodrigues(rvec)[0], (w, h))
    # T = np.matrix(tvec)
    # R, _ = cv2.Rodrigues(rvec)
    # R = R.T
    # # if np.linalg.det(R) < 0:
    # #     R = -R.T
    # #     T = -T

    # h, w = prev.shape[:2]
    # new_K, _ = cv2.getOptimalNewCameraMatrix(
    #     camera_mat, dist_coef, (w, h), 1, (w, h))
    # new_rot_mat = np.dot(new_K, R)

    # warped = cv2.warpPerspective(
    #     prev, np.dot(new_K, np.hstack((R, T))), (w, h))

    cv2.imshow('test', warped)
    cv2.waitKey()
    cv2.destroyAllWindows()


def stereo(frame1, frame2):
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)

    gray_left = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    win_size = 5
    max_disp = 16

    # win_size = 5
    # max_disp = 16

    stereo = cv2.StereoBM_create(numDisparities=max_disp, blockSize=win_size)
    focal_length = 0.8
    baseline = 50

    while True:

        disparity = stereo.compute(gray_left, gray_right)

        # focal_length = 1000
        # baseline = 2
        depth = (focal_length * baseline)/disparity

        # cv2.imwrite('depth_5,16.png', depth)
        cv2.imshow('disparity', disparity/255.)
        cv2.imshow('depth', depth)
        key = cv2.waitKey(1)
        if key == ord('q'):  # q를 누르면 종료
            break
        elif key == ord('w'):  # w를 누르면 윈도우 크기 증가
            win_size += 2  # 홀수만 들어가도록
            stereo.setBlockSize(win_size)
        elif key == ord('s'):  # s를 누르면 윈도우 크기 감소
            win_size -= 2  # 홀수만 들어가도록
            if win_size < 5:
                win_size = 5
            stereo.setBlockSize(win_size)
        elif key == ord('e'):  # e를 누르면 최대 검색 거리 증가
            max_disp += 16
            stereo.setNumDisparities(max_disp)
        elif key == ord('d'):  # d를 누르면 최대 검색 거리 감소
            max_disp -= 16
            if max_disp < 16:
                max_disp = 16
            stereo.setNumDisparities(max_disp)

        elif key == ord('r'):
            focal_length += 1
        elif key == ord('f'):
            focal_length -= 1
        elif key == ord('t'):
            baseline += 5
        elif key == ord('g'):
            baseline -= 5

    print('win size : ', win_size)
    print('focal length : ', focal_length)
    print('base line: ', baseline)
    print('max disp: ', max_disp)
    cv2.destroyAllWindows()

    # img = cv2.imread('depth_11,32.png', cv2.IMREAD_ANYDEPTH)
    # depth_normalized = cv2.normalize(
    #     img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # cv2.imshow('heatmap', heatmap)
    # cv2.waitKey(0)


def segment(frame1, frame2):
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)

    mask1 = cv2.imread('mask-157.png', cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread('mask-158.png', cv2.IMREAD_GRAYSCALE)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mask_add = cv2.bitwise_and(mask1, mask2)
    mask_add = cv2.bitwise_not(mask_add)
    # cv2.imshow('add mask', mask_add)
    # cv2.waitKey()

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, mask_add)
    kp2, des2 = sift.detectAndCompute(gray2, mask_add)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    h, w = img1.shape[:2]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # stitched[0:img1.shape[0], 0:img1.shape[1]] = img1
    # print('x ', dx)
    # print('y ', dy)

# 이미지를 표시
    # cv2.imshow("Masked Image", stitched)
    # cv2.imwrite('frame-157-dx.png', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multi_band(frame1, frame2):
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)

    mask1 = cv2.imread('mask-157.png', cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread('mask-158.png', cv2.IMREAD_GRAYSCALE)
    mask_add = cv2.bitwise_and(mask1, mask2)
    mask_add = cv2.bitwise_not(mask_add)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, mask_add)
    kp2, des2 = orb.detectAndCompute(img2, mask_add)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w, _ = img1.shape
    img2_transformed = cv2.warpPerspective(img2, M, (w*2, h))
    M_t = np.float32([[1, 0, 100], [0, 1, 0]])
    img2_transformed = cv2.warpAffine(img2_transformed, M_t, (w, h))
    cv2.imshow('warp', img2_transformed)
    cv2.moveWindow('warp', 0, 0)
    cv2.waitKey(0)
    result = np.zeros((h, w*2, 3), dtype=np.uint8)
    alpha = 0.5
    # result[:, :w, :] = img1 * 0.5
    result[:, w:, :] = img2_transformed[:, w:, :]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2

    # 매칭 결과 시각화
    for match in matches:
        # 추출한 키포인트의 좌표
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt
        # 키포인트 연결
        cv2.line(vis, (int(x1), int(y1)),
                 (int(x2) + w1, int(y2)), (0, 255, 0), 1)

    # cv2.imshow('vis', vis)

    cv2.waitKey(0)
    cv2.imshow('Multi-band blending result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# test
# multi_band(frames_path+file_list[156], frames_path+file_list[157])
# segment(frames_path+file_list[156], frames_path+file_list[157])
# stereo(frames_path+file_list[156], frames_path+file_list[157])
# warp_3d(frames_path+file_list[156], frames_path+file_list[157])
# calib_warp(frames_path+file_list[156], frames_path+file_list[157])
# morphing_image(frames_path+file_list[156], frames_path+file_list[157])
# line_detect(frames_path+file_list[156], frames_path+file_list[157])
# cv2.destroyAllWindows()
# start = time.time()
# for i in range(156, 165):
#     mid_frame = optical_flow_interpolation(
#         frames_path+file_list[i], frames_path+file_list[i+1], 0.5)
# end = time.time()
# print(f"{end-start:.3f} sec")
# # mid_frame = absdiff(frames_path+file_list[155], frames_path+file_list[156])


# for i in range(155, 165):
#     interpolate_frame(file_list[i], file_list[i+1], 0.3)
# interpolate_frame(file_list[i], file_list[i+1], 0.3)
# interpolate_frame(file_list[i], file_list[i+1], 0.1)


# synthesis_view(file_list[161], file_list[162])
# synthesis_view(file_list[163], file_list[164])


# clone_frames(frame_names, 626, 639, 4)
# clone_frames(frame_names, 142, 162, 2)
create_video('prev', 'output.mp4')
# create_video('save', 'copy_test.mp4')

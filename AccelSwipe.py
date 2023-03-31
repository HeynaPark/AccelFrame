import cv2
import os
import glob

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
        frame_path = os.path.join(frames_path, frame)
        frame_name, frame_ext = os.path.splitext(frame_path)
        for j in range(num_clones):
            clone_path = f"{frame_name}_{j}{frame_ext}"
            img = cv2.imread(frame_path)
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

    # cv2.imshow('result', result)
    cv2.imwrite(new_path, result)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


# for i in range(156, 165):
#     interpolate_frame(file_list[i], file_list[i+1], 0.5)
#     interpolate_frame(file_list[i], file_list[i+1], 0.3)
#     interpolate_frame(file_list[i], file_list[i+1], 0.1)
# synthesis_view(file_list[161], file_list[162])
# synthesis_view(file_list[163], file_list[164])


# clone_frames(frame_names, 626, 639, 4)
# clone_frames(frame_names, 613, 625, 2)
create_video('prev', 'output.mp4')
# create_video('save', 'output_inter_decel.mp4')

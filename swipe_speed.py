import cv2


# input
# data1
video_path = "video/SubRecord(9)_2023_04_05_11_50_17.mp4"
start_frame = [315, 330, 380]
end_frame = [329, 354, 439]
output_path = 'output/output.mp4'

# data2
# video_path = "video/soccer_hide_stabil.mp4"
# start_frame = [118]
# end_frame = [164]
# output_path = 'output/output2.mp4'


def duplicate_frame(delay_list, frame_idx, frame, output, flag):
    ratio = 0.1
    size = delay_list[1] - delay_list[0]

    n = 1
    if flag == 'front':
        if delay_list[0] <= frame_idx <= delay_list[0] + int(size*ratio):
            n = 6
        elif delay_list[1] - int(size*ratio) <= frame_idx <= delay_list[1]:
            n = 1

    if flag == 'back':
        if delay_list[0] <= frame_idx <= delay_list[0] + int(size*ratio):
            n = 1
        elif delay_list[1] - int(size*ratio) <= frame_idx <= delay_list[1]:
            n = 1

    if delay_list[0] <= frame_idx <= delay_list[1]:
        count = 0
        while count < n:
            print(frame_idx, n)
            output.write(frame)
            count += 1


def make_movie(video_path, start_frame, end_frame, output_path):

    video = cv2.VideoCapture(video_path)

    swipe_idx = 0
    front_delay = [start_frame[swipe_idx], start_frame[swipe_idx] +
                   int((end_frame[swipe_idx]-start_frame[swipe_idx])*0.15)]
    back_delay = [end_frame[swipe_idx] -
                  int((end_frame[swipe_idx]-start_frame[swipe_idx])*0.15), end_frame[swipe_idx]]

    print('delay : ', front_delay, back_delay)

    frame_cnt = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:

        ret, frame = video.read()
        if not ret:
            break

        frame_cnt += 1
        output.write(frame)

        duplicate_frame(front_delay, frame_cnt, frame, output, 'front')
        duplicate_frame(back_delay, frame_cnt, frame, output, 'back')
        # if front_delay[0] <= frame_cnt <= front_delay[1]:
        #     print(frame_cnt)
        #     # cv2.putText(frame, "duplited frame", (1920, 1600),
        #     # cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 2)
        #     output.write(frame)
        #     output.write(frame)

        # if back_delay[0] <= frame_cnt <= back_delay[1]:
        #     print(frame_cnt)
        #     output.write(frame)
        #     output.write(frame)

        if frame_cnt == back_delay[1]:
            swipe_idx += 1
            print('swipe change')
            if len(end_frame) == swipe_idx:
                continue
            else:
                front_delay = [start_frame[swipe_idx], start_frame[swipe_idx] +
                               int((end_frame[swipe_idx]-start_frame[swipe_idx])*0.15)]
                back_delay = [end_frame[swipe_idx] -
                              int((end_frame[swipe_idx]-start_frame[swipe_idx])*0.15), end_frame[swipe_idx]]

    video.release()
    output.release()

    print("end")


make_movie(video_path, start_frame, end_frame, output_path)

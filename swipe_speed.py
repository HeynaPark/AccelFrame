
import cv2
import math


# input
# data1
# file_name = "SubRecord(9)_2023_04_05_11_50_17.mp4"
# start_frame = [315, 330, 380]
# end_frame = [329, 354, 439]


# data2
file_name = "soccer_hide_stabil.mp4"
start_frame = 121
end_frame = 165
size = end_frame-start_frame

video_path = "video/" + file_name
output_path = 'output/' + file_name

front_delay = 0
back_delay = 0


def duplicate_frame(start, delay, frame_idx, frame, output, flag):
    ratio = 0.3
    interval = math.ceil(ratio*abs(delay - start))

    n = 0

    if flag == 'front':
        if start <= frame_idx <= start + interval:
            n = 2
        elif start <= frame_idx <= front_delay:
            n = 1

    if flag == 'back':
        if start - interval <= frame_idx < start:
            n = 3
        elif delay <= frame_idx < start:
            n = 2

    if start_frame <= frame_idx <= end_frame:
        count = 0
        while count < n:
            cv2.putText(frame, "+ " + str(count), (400, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 125), 3)
            print(frame_idx, n)
            output.write(frame)
            count += 1

# def duplicate_frame(delay_list, frame_idx, frame, output, flag):
#     ratio = 0.2
#     size = delay_list[1] - delay_list[0]
#     interval = math.ceil(ratio*size)

#     n = 1

#     if flag == 'front':
#         if delay_list[0] <= frame_idx <= delay_list[0] + interval:
#             n = 2
#         elif delay_list[0] <= frame_idx <= front_delay:
#             n = 1
#         # elif delay_list[1] - int(size*ratio) <= frame_idx <= delay_list[1]:
#         #     n = 1

#     if flag == 'back':
#         # if delay_list[0] <= frame_idx < delay_list[0] + math.ceil(size*ratio):
#         #     n = 2
#         if delay_list[1] - interval <= frame_idx < delay_list[1]:
#             n = 3
#         elif back_delay <= frame_idx < delay_list[1] - interval:
#             n = 2

#     if delay_list[0] <= frame_idx <= delay_list[1]:
#         count = 0
#         while count < n:
#             cv2.putText(frame, "+ " + str(count), (400,
#                                                    200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 125), 3)
#             print(frame_idx, n)
#             output.write(frame)
#             count += 1


def make_movie(video_path, start_frame, end_frame, output_path):
    global front_delay
    global back_delay
    video = cv2.VideoCapture(video_path)

    swipe_idx = 0
    ratio = 0.25
    # front_delay = [start_frame[swipe_idx], start_frame[swipe_idx] +
    #                int((end_frame[swipe_idx]-start_frame[swipe_idx])*ratio)]
    # back_delay = [end_frame[swipe_idx] -
    #               int((end_frame[swipe_idx]-start_frame[swipe_idx])*ratio), end_frame[swipe_idx]]

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

        cv2.putText(frame, "frame num : " + str(frame_cnt), (100,
                    100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 25), 3)

        frame_cnt += 1
        output.write(frame)

        duplicate_frame(start_frame, front_delay,
                        frame_cnt, frame, output, 'front')
        duplicate_frame(end_frame, back_delay,
                        frame_cnt, frame, output, 'back')
        # duplicate_frame(back_delay, frame_cnt, frame, output, 'back')

        # if frame_cnt == back_delay[1]:
        #     swipe_idx += 1
        #     print('swipe change')
        #     if len(end_frame) == swipe_idx:
        #         continue
        #     else:
        #         pass
        # front_delay = [start_frame[swipe_idx], start_frame[swipe_idx] +
        #                int((end_frame[swipe_idx]-start_frame[swipe_idx])*0.15)]
        # back_delay = [end_frame[swipe_idx] -
        #               int((end_frame[swipe_idx]-start_frame[swipe_idx])*0.15), end_frame[swipe_idx]]

    video.release()
    output.release()

    print("end")


def get_min_value():
    return start_frame


def get_max_value():
    return end_frame


def set_front_value(v):
    global front_delay
    front_delay = v
    print('front_delay', front_delay)


def set_back_value(v):
    global back_delay
    back_delay = v
    print('back_delay', back_delay)


def make():
    make_movie(video_path, start_frame, end_frame,
               output_path)

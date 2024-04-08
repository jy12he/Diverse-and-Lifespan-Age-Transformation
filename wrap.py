import numpy as np
import cv2 as cv
import argparse
import dlib

def img_txt_wrap(img_path, landmarks_path):

    srt_img = cv.imread(img_path)
    srt_landmark = np.zeros((81, 2))  
    gray = cv.cvtColor(srt_img, cv.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/media/gh/data1/xjc/DLFS/DLFS-main/shape_predictor_81_face_landmarks.dat")
    try:
        dets = detector(gray, 1)
        for i in range(len(dets)):
            shape = predictor(srt_img, dets[i])  
            index = 0
            for pt in shape.parts():
                srt_landmark[index][0] = pt.x
                srt_landmark[index][1] = pt.y
                index = index + 1

        num_ctrl_points = 84
        pad_landmarks = np.zeros((84 - 81, 2), np.uint8)
   
        # srt_landmark = np.loadtxt(input_land_path)
        srt_landmark = np.vstack((srt_landmark, pad_landmarks))
        dst_landmark = np.loadtxt(landmarks_path)
        dst_landmark = np.vstack((dst_landmark, pad_landmarks))
        # print(dst_landmark)
 
        boundary = np.array([[0, 0, 0, 0, 0, 64, 128, 192, 255, 64, 128, 192, 255, 255, 255, 255],
                            [0, 64, 128, 192, 255, 255, 255, 255, 255, 0, 0, 0, 0, 64, 128, 192]], dtype=float)
        for i in range(0, 16):
            srt_landmark[i + 64, 0] = boundary[0, i]
            srt_landmark[i + 64, 1] = boundary[1, i]
            dst_landmark[i + 64, 0] = boundary[0, i]
            dst_landmark[i + 64, 1] = boundary[1, i]

        wv = np.ones((num_ctrl_points + 3, 2))
        wv = CalculateCoeff(wv=wv, srt_landmark=srt_landmark, dst_landmark=dst_landmark)

        w = srt_img.shape[1]
        h = srt_img.shape[0]
        dst_img = np.zeros((256, 256, 3), np.uint8)

        result = WarpMapping(dst_img, srt_img, dst_landmark, wv)

        return result.astype(np.uint8)
    
    except:
        print("detect false")
        return np.zeros((256, 256, 3), dtype=np.uint8)






def CalculateCoeff(wv, srt_landmark, dst_landmark):
    num = srt_landmark.shape[0] 
    A = np.zeros((num + 3, num + 3))
    b = np.zeros((num + 3, 2))

    for i in range(0, num):
        row = num - i
        # tmp_mat1 = dst_landmark.block(i, 0, row, 1).array() - dst_landmark(i, 0)
        tmp_mat1 = dst_landmark[i:i+row, 0] - dst_landmark[i, 0]
        tmp_mat2 = dst_landmark[i:i+row, 1] - dst_landmark[i, 1]
        A[i:i+row, i] = np.sqrt(np.square(tmp_mat1) + np.square(tmp_mat2))
        tmp_mat1 = A[i:i+row, i]
        A[i, i:i+row] = tmp_mat1.T
    # np.savetxt('A.txt',A, fmt='%.5e')
    A[num, 0:num] = np.ones((1, num), dtype=float)
    A[0:num, num:num+1] = np.ones((num, 1), dtype=float)
    A[num+1:num+3, 0:num] = dst_landmark.T
    A[0:num, num+1:num+3] = dst_landmark
    b[0:num, 0:2] = srt_landmark - dst_landmark

    wv[0:num+3, 0] = np.squeeze(np.linalg.lstsq(A, b[0:num+3, 0], rcond=-1)[0])
    # print(np.linalg.lstsq(A,b[0:num+3, 0:1])[0].shape)
    wv[0:num+3, 1] = np.squeeze(np.linalg.lstsq(A, b[0:num+3, 1], rcond=-1)[0])

    return wv



def WarpMapping(dst, srt, dst_landmark, wv):
    h = dst.shape[0]
    w = dst.shape[1]
    num = dst_landmark.shape[0]

    new_x_mat = np.zeros((256, 256))
    new_y_mat = np.zeros((256, 256))
    base_mat = np.zeros((256, 256))
    tmp_mat = np.zeros((256, 256))
    tmp_vec = np.zeros(256)

    for i in range(256):
        tmp_mat[0:256, i:i+1] = i * np.ones((256, 1))
        tmp_vec[i] = i
    for i in range(num):
        base_l = np.square(tmp_mat - dst_landmark[i, 0])
        base_r = np.tile(np.square(tmp_vec - dst_landmark[i, 1]).reshape(256, 1), 256)
        base_mat = np.sqrt(base_l + base_r)

        new_x_mat = new_x_mat + base_mat * wv[i, 0]
        new_y_mat = new_y_mat + base_mat * wv[i, 1]
    new_x_mat = (new_x_mat + wv[num + 2, 0] * tmp_mat.T
                 + (wv[num + 1, 0] + 1) * tmp_mat) + wv[num,0]
    new_y_mat = (new_y_mat + (wv[num + 2, 1] + 1) * tmp_mat.T
                 + wv[num + 1, 1] * tmp_mat) + wv[num,1]
    for i in range(256):
        for j in range(256):
            x = new_x_mat[i][j]
            y = new_y_mat[i][j]
            floor_x = int(np.floor(x))
            floor_y = int(np.floor(y))
            if ((floor_x < 0) or ((floor_x + 1) >= w) or (floor_y < 0) or ((floor_y + 1) >= h)):
                continue
            u = x - floor_x
            v = y - floor_y

            dst[i][j][0] = (1-u) * (1-v) * int(srt[floor_y][floor_x][0]) + u * (1 - v) * int(srt[floor_y][floor_x + 1][0]) \
                               + v * (1-u) * int(srt[floor_y + 1][floor_x][0]) + u * v * int(srt[floor_y + 1][floor_x + 1][0])

            dst[i][j][1] = (1-u) * (1-v) * int(srt[floor_y][floor_x][1]) + u * (1 - v) * int(srt[floor_y][floor_x + 1][1]) \
                               + v * (1-u) * int(srt[floor_y + 1][floor_x][1]) + u * v * int(srt[floor_y + 1][floor_x + 1][1])

            dst[i][j][2] = (1-u) * (1-v) * int(srt[floor_y][floor_x][2]) + u * (1 - v) * int(srt[floor_y][floor_x + 1][2]) \
                               + v * (1-u) * int(srt[floor_y + 1][floor_x][2]) + u * v * int(srt[floor_y + 1][floor_x + 1][2])
            # print(result[i][j])

    return dst




if __name__ == '__main__':

    parser = argparse.ArgumentParser("wrap")
    parser.add_argument('--input_img_path', type=str, help='input image')
    parser.add_argument('--dst_land_path', type=str,  help='target landmarks')
    parser.add_argument('--save_path', type=str,  help='target img')
    args = parser.parse_args()

    input_img_path = args.input_img_path
    # input_land_path = args.input_land_path
    dst_land_path = args.dst_land_path
    save_path = args.save_path


    wrap_result = img_txt_wrap(input_img_path, dst_land_path)
    cv.imwrite(save_path, wrap_result)
    print("wrap ok")
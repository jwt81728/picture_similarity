from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import cv2
import time
import tkinter as tk
import tkinter.messagebox

def pic_capture():
    timenow = time.time()
    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        count = 5 - int(time.time() - timenow)
        ret, img = cap.read()
        if ret == True:
            imgcopy = img.copy()
            cv2.putText(imgcopy, str(count), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 15, (0, 0, 255), 35)  # 在複製影像上畫倒數秒數
            cv2.imshow("frame", imgcopy)
            k = cv2.waitKey(100)
            if k == ord("z") or k == ord("Z") or count == 0:
                cv2.imwrite("images/0000.jpg", img)
                break
    cap.release()
    cv2.destroyWindow("frame")

# 計算相似矩陣
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def main():

    def confirm_to_quit():
        if tk.messagebox.askokcancel('溫馨提示', '确定要退出嗎'):
            window.quit()

    def picture_similarity():
        pic_capture()
        # 自 images 目錄找出所有 JPEG 檔案
        y_test = []
        x_test = []
        for img_path in os.listdir("images"):
            if img_path.endswith(".jpg"):
                img = image.load_img("images/" + img_path, target_size=(224, 224))
                y_test.append(img_path[:])
                x = image.img_to_array(img)
                x = np.expand_dims(x, 0)
                if len(x_test) > 0:
                    x_test = np.concatenate((x_test, x))
                else:
                    x_test = x
        # 轉成 VGG 的 input 格式
        x_test = preprocess_input(x_test)

        # include_top=False，表示會載入 VGG16 的模型，不包括加在最後3層的卷積層，通常是取得 Features (1,7,7,512)
        model = VGG16(weights='imagenet', include_top=False)

        # 萃取特徵
        features = model.predict(x_test)
        # 計算相似矩陣
        features_compress = features.reshape(len(y_test), 7 * 7 * 512)
        sim = cosine_similarity(features_compress)

        # 找出會員照片中與0000.jpg最像似的照片
        top = np.argsort(-sim[0], axis=0)[0:2]

        # 列出與0000.jpg最相似
        recommend = [y_test[i] for i in top]
        name = recommend[1].split('.')[0]
        response_name = '歡迎登入 ' + name
        respond_label.configure(text=response_name)

    # window frame setting
    window = tk.Tk()
    window.geometry('400x300')
    window.title('吾人商店系統')

    label_online_reg = tk.Button(window, text='會員登入\n (僅限於線上上註冊的會員)', font='Arial -16', fg='green',command=picture_similarity)
    label_online_reg.pack(expand=1)

    #response
    respond_label = tk.Label(window)
    respond_label.pack(expand=1)

    # quit windows
    quit_frame = tk.Frame(window)

    quit_button = tk.Button(window, text='退出', command=confirm_to_quit)
    quit_button.pack()

    quit_frame.pack(side='bottom')

    window.mainloop()


if __name__ == "__main__":
    main()


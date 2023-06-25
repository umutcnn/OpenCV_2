#python para_saydir.py -i coin.jpg
import cv2
import numpy as np
import argparse

def segment_coins(image_path):
    # görseli alma
    coins = cv2.imread(image_path)

    # görüntüyü gri tonlamaya dönüştürme
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    
    # goruntuyu threshle 
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)

    #arka plandaki gürültüyü kaldırmak için morphologyEx fonsiyonu kullandım
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow("opening", opening)

    # arka planı belirleme
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # paraları algılamak için distanceTransform kullandım
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    cv2.imshow("dist_transform", dist_transform)

    # parayı 8 bit unsigned integer'a dönüştürme
    sure_fg = np.uint8(sure_fg)
    #fotoğrafları birlestirme
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    # arka planın 0 değil 1 olması için tüm etiketlere bir tane ekleyin
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(coins,markers)
    coins[markers == -1] = [255,0,0]

    cv2.imshow("Coins", coins)
    cv2.waitKey(0)

    return markers

def calculate_coins(image_path):
    markers = segment_coins(image_path)
    unique_markers = np.unique(markers)
    coin_count = len(unique_markers) - 1

    total_value = 0
    for marker in unique_markers:
        if marker == 0:  # 0 değerli marker arkaplan için kullanılır, bu nedenle atlanmalıdır.
            continue
        coin_area = np.sum(markers == marker)
        if coin_area <= 50:  # alanı küçük olan paralar
            total_value += 0.10  # 10 kurus
        elif coin_area <= 1000:  # alanı orta boy olan paralar
            total_value += 0.25  # 25 kurus
        elif coin_area <= 2500:  # alanı daha büyük olan paralar
            total_value += 0.50  # 50 kurus
        else:  # en büyük paralar
            total_value += 1  # 1 TL

    print(f"Toplamda {coin_count} adet para bulunmaktadır.")
    print(f"Toplam para değeri: {total_value} TL")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

image_path = args['image']
calculate_coins(image_path)
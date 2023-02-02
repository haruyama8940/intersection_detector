# intersection_detector
単眼カメラを用いた交差点検出用パッケージ

分類クラス：４つ  
[0]:straight_road [1]:3_way [2]:cross_road [3]:corridor


ネットワーク
時系列考慮なし
1.CNN３ ＋　全結合層２
時系列考慮あり（LRCN）
2.mobilenetv2 + lstm + fc
## INSTALL
```
git clone https://github.com/haruyama8940/intersection_detector.git
git clone https://github.com/haruyama8940/scenario_navigation_msgs.git
```

## RUN
plase start simulation and
1.のネットワーク
```
roslaunch intersection_detector intersection_detect.launch 
```
2.のネットワーク
```
roslaunch intersection_detector intersection_detect_lrcn.launch 
```

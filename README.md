# intersection_detector
単眼カメラを用いた交差点検出用パッケージ

分類クラス：４つ  
[0]:straight_road [1]:3_way [2]:cross_road [3]:corridor


ネットワーク
CNN３ ＋　全結合層２

##INSTALL
```
git clone https://github.com/haruyama8940/intersection_detector.git
git clone https://github.com/haruyama8940/scenario_navigation_msgs.git
```

## RUN
plase start simulation and
```
roslaunch intersection_detector intersection_detect.launch 
```

## TODO
LSTMの導入

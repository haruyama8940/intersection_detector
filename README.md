# intersection_detector
単眼カメラを用いた交差点検出用パッケージ

分類クラス：４つ  
[0]:straight_road [1]:3_way [2]:cross_road [3]:corridor


### Publish topic
・/passage_type[(std_msgs/String)](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html "std_msgs/string")

### Subscribe topic
・/camera/rgb/image_raw[(sensor_msgs/Image)](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html "sensor_msgs/Image ")

### ネットワーク
・CNN３ ＋　全結合層２  
・mobilenetv2
## RUN
```
roslaunch intersection_detector intersection_detect.launch 
```

## TODO
LSTMの導入
動画の追加

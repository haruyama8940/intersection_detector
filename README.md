# intersection_detector
単眼カメラを用いた交差点検出用パッケージ
分類クラス：４つ  
[0]:straight_road [1]:3_way [2]:cross_road [3]:corridor


### Publish topic
・/passage_type[(scenario_navigation_msgs/cmd_dir_intersection)](https://github.com/haruyama8940/scenario_navigation_msgs.git "scenarioa_navigation_msgs/cmd_dir_intersection")

### Subscribe topic
・/camera/rgb/image_raw[(sensor_msgs/Image)](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html "sensor_msgs/Image ")

### ネットワーク
・CNN３ ＋　全結合層２  
・mobilenetv2
## Instll 
https://github.com/haruyama8940/scenario_navigation_msgs.git
https://github.com/haruyama8940/intersection_detector.git
## RUN
```
roslaunch intersection_detector intersection_detect.launch 
```

## TODO
LSTMの導入
動画の追加

CPLID 已经划分好8:1:1 可直接用于YOLO训练
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af4ad8062c0e40929d1fb341a60cd6fd.png)

中国输电线路绝缘子数据集(Chinese Power Line Insulator Dataset，CPLID)是第一个公开的绝缘子数据集，包含无人机拍摄的正常绝缘子图像和合成的绝缘子缺陷图像共848张，其中正常绝缘子图像600张，绝缘子缺陷图像248张，图像尺寸都为1152×684。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31cd4215d1bc44cb8ba985f53a1246a6.png)
1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c9ec2a819b424af392e1f484767e0f34.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bb66691109ae46ab9a8f95c7991e73bb.png)
# 中国输电线路绝缘子数据集 CPLID（YOLO目标检测）
## 一、数据集信息表
| 项目 | 详细说明 |
| ---- | ---- |
| 数据集名称 | 中国输电线路绝缘子数据集（CPLID） |
| 任务类型 | 输电线路绝缘子缺陷检测（YOLO目标检测） |
| 总图片数量 | 848 张 |
| 图像分辨率 | 1152 × 684 |
| 数据构成 | 正常绝缘子：600 张<br>缺陷绝缘子：248 张 |
| 预设划分比例 | 8:1:1（训练集:验证集:测试集） |
| 训练集 | 678 张 |
| 验证集 | 85 张 |
| 测试集 | 85 张 |
| 标注格式 | YOLO TXT 标准标注 |
| 灵活拓展 | 支持按自定义比例重新划分数据集 |
| 适用场景 | 无人机电力巡检、输电线路缺陷检测、算法竞赛、毕业设计 |

## 二、数据集配置文件 `cplid.yaml`
```yaml
# CPLID 绝缘子数据集配置
train: ./images/train
val: ./images/val
test: ./images/test

nc: 2
names:
  0: normal_insulator
  1: defect_insulator
```

## 三、目录结构
```
CPLID/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── cplid.yaml
```

## 四、环境依赖安装
```bash
pip install ultralytics torch opencv-python
```

## 五、YOLOv8 训练代码
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8s.pt")

# 开始训练
model.train(
    data="cplid.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    patience=20,
    project="CPLID_Insulator",
    name="yolov8_train",
    save=True
)

# 模型验证评估
model.val()
```

## 六、YOLOv11 训练代码
```python
from ultralytics import YOLO

model = YOLO("yolov11s.pt")

model.train(
    data="cplid.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    patience=20,
    project="CPLID_Insulator",
    name="yolov11_train",
    save=True
)

model.val()
```

## 七、单图推理测试代码
```python
from ultralytics import YOLO
import cv2

# 加载训练完成的最优模型
model = YOLO("runs/detect/yolov8_train/weights/best.pt")

img_path = "test.jpg"
results = model(img_path, conf=0.25)

# 绘制并保存结果
result_img = results[0].plot()
cv2.imwrite("detect_result.jpg", result_img)
cv2.imshow("绝缘子检测结果", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

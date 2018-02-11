## This is the MXNet implement of MobileNet V2 (train on ImageNet dataset)

Paper: [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segment](128.84.21.199/abs/1801.04381)


# Usage

### Prepare data

This code takes ImageNet dataset as example. You can download ImageNet dataset and translate it into `.rec` using `~/incubator-mxnet/tools/im2rec.py`.


### Train

* If you want to train from scratch, you can run as follows:

```
python train.py --batch-size 256 --gpus 0,1,2,3 --num-epoch 200 --data-train path/to/train.rec --data-val path/to/val.rec
```

* If you want to train from one checkpoint, you can run as follows(for example train from `output/mobilenetv2/mobilenetv2-0010.params`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
python train.py --batch-size 256 --gpus 0,1,2,3 --num-epoch 200 --data-train path/to/train.rec --data-val path/to/val.rec --resume output/mobilenetv2/mobilenetv2 --start-epoch 10
```

### Pretrained model
Will update soon
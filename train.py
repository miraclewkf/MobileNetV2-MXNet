import argparse
import mxnet as mx
import os, sys
import logging
from mobilenetv2 import get_symbol

def multi_factor_scheduler(args, epoch_size):
    factor = args.factor
    step = [i for i in range(1, args.num_epoch)]
    step_ = [epoch_size * (x - args.start_epoch) for x in step if x - args.start_epoch > 0]
    lr = args.lr
    for s in step:
        if args.start_epoch >= s:
            lr *= factor
    if lr != args.lr:
        logging.info("Adjust learning rate to {:6f} for epoch {}".format(lr, args.start_epoch))
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None)

def train_model(args, kv='device'):
    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_train,
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=(3, 224, 224),
        batch_size=args.batch_size,
        rand_crop=args.random_crop,
        rand_mirror=args.random_mirror,
        shuffle=True,
        num_parts=kv.num_workers,
        resize=256,
        max_random_scale=1,
        min_ranodm_scale=0.08,
        max_aspect_ratio=0.25,
        max_rotate_angle=10,
        max_random_contrast=0.125,
        max_random_illumination=0.125,
        part_index=kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec=args.data_val,
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=(3, 224, 224),
        batch_size=args.batch_size,
        rand_crop=False,
        rand_mirror=False,
        shuffle=False,
        num_parts=kv.num_workers,
        resize=224,
        part_index=kv.rank)

    sym = get_symbol(num_classes=args.num_classes, dropout=args.dropout)
    arg_params = None
    aux_params = None

    # If train from one checkpoint
    if args.resume:
        sym, arg_params, aux_params = mx.model.load_checkpoint(args.resume, args.start_epoch)

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    lr, lr_scheduler = multi_factor_scheduler(args, epoch_size)

    optimizer_params = {
        'learning_rate': lr,
        'momentum': args.mom,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler}
    initializer = mx.init.Xavier(
        rnd_type='gaussian', factor_type="in", magnitude=2)

    if args.gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    model = mx.module.Module(
        context=devs,
        symbol=sym
    )

    checkpoint = mx.callback.do_checkpoint(os.path.join(args.save_result + "-" + str(args.layers),
                                                        args.save_name + "-" + str(args.layers)))

    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(['Accuracy'])
    eval_metric.add(['CrossEntropy'])

    model.fit(train,
              begin_epoch=args.start_epoch,
              num_epoch=args.num_epoch,
              eval_data=val,
              eval_metric=eval_metric,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,  # for new fc layer
              batch_end_callback=mx.callback.Speedometer(args.batch_size, 20),
              epoch_end_callback=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-train', type=str, default='/mnt/data/ImageNet/rec/train.rec')
    parser.add_argument('--data-val', type=str, default='/mnt/data/ImageNet/rec/val.rec')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num-epoch', type=int, default=25)
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result', type=str, help='the save path', default='output/mobilenetv2')
    parser.add_argument('--num-examples', type=int, default=1281167)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.00004, help='weight decay for sgd')
    parser.add_argument('--save-name', type=str, help='the save name of model', default='mobilenetv2')
    parser.add_argument('--random-crop', type=int, default=1, help='if or not randomly crop the image')
    parser.add_argument('--random-mirror', type=int, default=1, help='if or not randomly flip horizontally')
    parser.add_argument('--layers', type=int, default=19, help='layers of se_resnext')
    parser.add_argument('--resume', type=str, default='', help='train from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='epoch of resume checkpoint')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--factor', type=float, default=0.98, help='step factor for learning rate')
    args = parser.parse_args()

    kv = mx.kvstore.create(args.kv_store)

    if not os.path.exists(args.save_result + "-" + str(args.layers)):
        os.makedirs(args.save_result + "-" + str(args.layers))

    # create a logger and set the level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # this handler is used to record information in train.log
    hdlr = logging.FileHandler(os.path.join(args.save_result + "-" + str(args.layers), 'train.log'))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # this handler is used to print information in terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # record the information of args
    logging.info(args)

    train_model(args, kv=kv)

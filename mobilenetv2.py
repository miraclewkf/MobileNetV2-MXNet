import mxnet as mx
import numpy as np

def inverted_residual_unit(data, num_filter, stride, dim_match, name, expension, bn_mom=0.9, workspace=256, memonger=False):

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*expension), kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*expension), kernel=(3,3), stride=stride, pad=(1,1),
                               no_bias=True, workspace=workspace, name=name + '_conv2', num_group=int(num_filter*expension))
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut

def mobilenetv2(units, num_stages, filter_list, num_classes, expension, bn_mom=0.9, workspace=256, dtype='float32', memonger=False, dropout=0.5):

    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(2,2), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')

    for i in range(num_stages):
        body = inverted_residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), expension=expension[i], workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = inverted_residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                          expension=expension[i], workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    flat = mx.sym.Dropout(data=flat, p=dropout)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')

def get_symbol(num_classes, conv_workspace=256, dtype='float32', dropout=0.5, **kwargs):

    filter_list = [32, 16, 24, 32, 64, 96, 160, 320, 1280]

    num_stages = 7

    units = [1, 2, 3, 4, 3, 3, 1]
    expension = [1, 6, 6, 6, 6, 6, 6]

    return mobilenetv2(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  expension   = expension,
                  workspace   = conv_workspace,
                  dtype       = dtype,
                  dropout     = dropout)
from core import *
from torch_backend import *
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('rankscale', type=int, help='Factor to scale the rank used in'
        ' low-rank layers.')
parser.add_argument('-d', action='store_true', help='Enable'
        'compression ratio weight decay scaling')

#Network definition
class GenericLowRank(nn.Module):
    """A generic low rank layer implemented with a linear bottleneck, using two
    Conv2ds in sequence. Preceded by a depthwise grouped convolution in keeping
    with the other low-rank layers here."""
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
        assert groups == 1
        super(GenericLowRank, self).__init__()
        if kernel_size > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
        else:
            self.grouped = None
        self.contract = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, out_channels, 1, bias=bias)
    def forward(self, x):
        if self.grouped is not None:
            x = self.grouped(x)
        x = self.contract(x)
        return self.expand(x)

# compression scaling factor
cscale = 2

def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
                if c_in==3 else 
                GenericLowRank(c_in, c_out, kernel_size=3,
                    rank=max(1,c_out//cscale), stride=1, padding=1,
                    bias=False), 
        #'conv': nn.Conv2d(c_in, c_out, kernel_size=3,
        #    stride=1, padding=1, bias=False), 
        #'conv_p': nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False), 
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw), 
        'relu': nn.ReLU(True)
    }

def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }

def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'classifier': Mul(weight),
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    s = 1
    channels = channels or {'prep': s*64, 'layer1': s*128, 'layer2': s*256, 'layer3': s*512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)       
    return n

losses = {
    'loss':  (nn.CrossEntropyLoss(reduce=False), [('classifier',), ('target',)]),
    'correct': (Correct(), [('classifier',), ('target',)]),
}

class TSVLogger():
    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']
    def append(self, output):
        epoch, hours, acc = output['epoch'], output['total time']/3600, output['test acc']*100
        self.log.append(f'{epoch}\t{hours:.8f}\t{acc:.2f}')
    def __str__(self):
        return '\n'.join(self.log)
   
def main():
    args = parser.parse_args()
    global cscale
    cscale = args.rankscale
    #DATA_DIR = './data'
    DATA_DIR = '/disk/scratch/gavin/data'

    print('Downloading datasets')
    dataset = cifar10(DATA_DIR)

    #epochs = 24
    epochs = 128
    #lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    lr_schedule = Cosine(epochs, 0.2)
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = Network(union(net(), losses)).to(device).half()
    
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(dataset['test']['labels']) % batch_size]:
        warmup_cudnn(model, size)
    
    print('Starting timer')
    timer = Timer()
    
    print('Preprocessing training data')
    train_set = list(zip(transpose(normalise(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))
    print(f'Finished in {timer():.2} seconds')
    
    TSV = TSVLogger()
    
    train_batches = Batches(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    test_batches = Batches(test_set, batch_size, shuffle=False, drop_last=False)
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size
    weight_decay = 5e-4 #*batch_size
    if args.d:
        opt = SGD(trainable_params(model, weight_decay), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
   
    run = train(model, opt, train_batches, test_batches, epochs, loggers=(TableLogger(), TSV), timer=timer, test_time_in_total=False)
    
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    results.append((args.__dict__,run))
    with open("results.json", 'w') as f:
        json.dump(results, f)

    name = 'cr_weight_decay' if args.d else 'normal'
    with open('logs/%s_%i.tsv'%(name, args.rankscale), 'w') as f:
        f.write(str(TSV))        
        
main()

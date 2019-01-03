
12th December 2018
==================

Running some experiments here as part of a project involving low-rank
approximations in deep learning. Would like the experiments to run quickly.
Tested the code on a Titan X (Pascal) GPU. `dawn.py` ran in 212 seconds and
reached 94% accuracy.

Running the same script on our V100 GPU, it took 223 seconds to run the
same 24 epochs. Which is strange, because that GPU should be just as fast
as the GPU on the amazon instance used in the post.

Switched the full convolutions in `conv_bn` for separable convolutions and
the network now converges to 92% accuracy in 171 seconds.

Switched in a low-rank approximation using a generic bottleneck, and it
only converged to 36% in 163 seconds.

Using the same low-rank approximation, and changing the weight decay to be
scaled by the compression factor, got 21.7% accuracy in 209 seconds.

Oh no, I must have had the rank set to something different. Running it
again with the old weight decay settings and it reaches 21.3% in 209
seconds.

Converged to 81.2% with a compression scaling factor of 2, which sets the
rank to be integer division 2 of the number of input channels. This is
barely a compression of the separable network, so this might not be the
best way to do this.

At a compression ratio of around 20%, with the scaling factor set to 10, it
converges to about 78.3%. With just the original weight decay it converges
to 77.8%.

I suppose the problem is that this network doesn't really have time to
worry about underfitting in the short training schedule. In both
experiments, the train loss is *above* the test loss. What if we increase
the training schedule by 4 times?

Still doesn't overfit. The weight decay setting used here is already much
higher than normal (it's the default setting of 5e-4 multiplied by 512).

13th December 2018
==================

Tried training with weight decay set to 0 and it *still* doesn't overfit.
Train loss still tracks validation loss.

Trying the same thing, but disabling the low-rank layers, maybe those just
can't be fit.

That worked, was able to get it to overfit.

Experiment Plan
---------------

I can't trust the experiment settings here to really replicate the kind of
training I'm doing in my other experiments. So, I'm going to make this
work more like those experiments. Changes:

1. Use a 100 epoch cosine annealing schedule.
2. Use default `5e-4` weight decay.
3. Parse args for setting the rank scaling.
4. Save results to a shared csv.
5. Save detailed logs to different file names, under dir `logs`.

With these changes I'm expecting it to take much longer, in the region of
1000-2000 seconds. But, this is still half an hour, which is several times
faster than running these experiments with my own code (about 8 hours).

Changed it to 128 epochs ans we get some overfitting. The train loss ends
at 0.04, with test at 0.265. Test accuracy was 92.8%.

Then, enabling the
default 5e-4 weight decay, the exact same thing happens. Train loss 0.04,
test 0.272. Test accuracy at 92.7%.

14th December 2018
==================

Experiment completed quite quickly, running over 4 GPUs. Results are in the
notebook on appropriate weight decay. It does seem to make a difference,
and works slightly better. Whether some other setting for weight decay is
not certain though. It may be worth trying the 8.8e-6 setting we were using
in experiments originally, for example.

21st December 2018
==================

To design experiments involving the tensor-train decomposition, we have run
into a problem deciding what type of decomposition we ought to use. It's
not clear from the theory what might be better, so I'm just going to run a
large number of experiments here and interpret the results.

Two parameters control most of the design of a tensor-train decomposition:
the rank of the cores used to represent the tensor and the number of
dimensions in the tensor being decomposed. We're going to vary both and see
what effect it has on the accuracy of a trained network. In addition we're
interested in the number of parameters used by the network in each case.

Note: with the current settings using float32 instead of float16 (tntorch
doesn't support float16), the final test error with a "normal" network is
93.5%. It would really be better if it were 94%.

28th December 2018
==================

The experiment is now completed. The notebook is
[here](https://gist.github.com/gngdb/2d29e5afbb21869e24952284cc287388).
Discussed in the deficient-efficient research log
[here](https://github.com/BayesWatch/deficient-efficient/blob/master/research-log.md#28th-december-2018).

It was run with a symlink for the `decomposed.py` file in
dificient-efficient. Not terribly good for replication, but it made
updating and editing easier.

29th December 2018
==================

Was going to repeat the experiment with CP-decomposed Tensors, but found
that I would get the following error:

```
Traceback (most recent call last):
  File "dawn.py", line 144, in <module>
    main()
  File "dawn.py", line 98, in main
    model = Network(union(net(), losses)).to(device)# .half()
  File "dawn.py", line 58, in net
    n = basic_net(channels, weight, pool, **kw)
  File "dawn.py", line 46, in basic_net
    'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw),
pool=pool),
  File "dawn.py", line 27, in conv_bn
    stride=1, padding=1, bias=False), 
  File "/disk/scratch/gavin/repos/cifar10-fast/decomposed.py", line 173, in
__init__
    bias=bias)
  File "/disk/scratch/gavin/repos/cifar10-fast/decomposed.py", line 59, in
__init__
    self.tn_weight = self.TnConstructor(self.weight.data.squeeze(),
ranks=self.rank)
  File "/disk/scratch/gavin/repos/cifar10-fast/decomposed.py", line 170, in
cp
    return tn.Tensor(tensor, ranks_cp=ranks)
  File "/disk/scratch/gavin/repos/tntorch/tntorch/tensor.py", line 103, in
__init__
    prod *= grams[m]
RuntimeError: The size of tensor a (8) must match the size of tensor b (2)
at non-singleton dimension 1
```

This happened when the Tensor we are attempting to compress is of size:

```
torch.Size([16, 16, 16, 2])
```

Doing some messing around with tntorch interactively, it seems like this
happens when a trailing dimension is of size 2 (for a given setting of
rank_cp). This is not terribly scientific, but that seems to be the case,
so adding some code to catch those instances and merge the trailing two
dimensions in that case.

Oh no, that's dumb, it happens when there is a dimension of smaller size
than the ranks_cp setting.

1st January 2019
================

Looking at the results for the CP-decomposition, all of the Tensors just
got squashed to two dimensions, because often the prescribed rank of the
low-rank tensor decomposition was so large that this was necessary to keep
each dimension of the original tensor larger than the low-rank
approximation. Illustration of the results of that experiment are
[here](https://gist.github.com/gngdb/9873cf7550aaee7b14ed86c8a2bc84bd). As
the rank option is repeated over the dimensions of the tensor being
approximated, it seems like we ought to be dividing that rank option by the
number of dimensions.

But, that means we've been doing the wrong thing with the TT and Tucker
decomposition experiments as well, so we may as well run them again as
well.

Unfortunately, fixing that with the CP-decomposition raises the following
error:

```
python dawn.py 15 15
/disk/scratch/gavin/miniconda3/envs/pytorch/lib/python3.6/site-packages/torch/nn/_reduction.py:49: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Downloading datasets
Files already downloaded and verified
Files already downloaded and verified
Traceback (most recent call last):
  File "dawn.py", line 146, in <module>
    main()
  File "dawn.py", line 98, in main
    model = Network(union(net(), losses)).to(device)# .half()
  File "dawn.py", line 58, in net
    n = basic_net(channels, weight, pool, **kw)
  File "dawn.py", line 46, in basic_net
    'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
  File "dawn.py", line 27, in conv_bn
    stride=1, padding=1, bias=False), 
  File "/disk/scratch/gavin/repos/cifar10-fast/decomposed.py", line 179, in __init__                                                                                                             bias=bias)
  File "/disk/scratch/gavin/repos/cifar10-fast/decomposed.py", line 59, in __init__
    self.tn_weight = self.TnConstructor(self.weight.data.squeeze(), ranks=self.rank)
  File "/disk/scratch/gavin/repos/cifar10-fast/decomposed.py", line 176, in cp
    return tn.Tensor(tensor, ranks_cp=ranks_cp)
  File "/disk/scratch/gavin/repos/tntorch/tntorch/tensor.py", line 94, in __init__
    grams = [None] + [self.cores[n].t().matmul(self.cores[n]) for n in range(1, self.dim())]
  File "/disk/scratch/gavin/repos/tntorch/tntorch/tensor.py", line 94, in <listcomp>
    grams = [None] + [self.cores[n].t().matmul(self.cores[n]) for n in range(1, self.dim())]
RuntimeError: invalid argument 8: lda should be at least max(1, 2), but have 1 at /opt/conda/conda-bld/pytorch_1544174967633/work/aten/src/TH/generic/THBlas.cpp:330
```

Appears this happens because `ranks_cp` gets set to 0 for this tensor
(<tensor size> <ranks_cp>):

```
torch.Size([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]) 0
```

Reasonable fix is to not let ranks_cp be 0, set it to be `max(ranks_cp, 1)`.

2nd January 2019
================

While investigating the size of models produced when we substitute in these
low-rank approximations, found that the way we set the rank is seriously
lacking. It didn't provide good control over the number of parameters used
because it failed to take into account how the size of dimensions changes
when dimensions are added.

I've changed the code to use a rank scaling factor that looks at the size
of the dimensions produced when `dimensionize` increases the number of
dimensions being used and scales the rank used on each dimension according
to this. I made a notebook to graph how the number of parameters used
expressing a tensor and expressing a network changes when this is done,
[here](https://gist.github.com/gngdb/b94684f7fbf46fbced859b83ae84df46).

With these changes though, the results we have running experiments with
this network now need to be repeated. I'm going to drop CP decompositions
from experiments. It looks like the number of parameters used drops so
quickly with the number of dimensions in a tensor that it's not likely to
be a useful way to express a tensor. I think that it should be possible to
use more parameters, but I keep hitting errors with tntorch, so dropping it
for now is the easiest thing to do.

Started experiment looking at reasonably sized models with just
Tensor-Train and Tucker-TT decompositions.

3rd January 2018
================

Results of the experiment started yesterday are illustrated
[here](https://gist.github.com/gngdb/9f091b368ca317dfb445ea87a9c0a7f0).
They seem to suggest that either 3 or 4 dimensions is best for both Tucker
and TT decompositions. It seems that having just 2 dimensions can also work
in many cases, but it would be more interesting to focus on the higher
dimensional cases.


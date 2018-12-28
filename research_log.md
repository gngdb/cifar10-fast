
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


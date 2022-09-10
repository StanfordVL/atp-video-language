## Atemporal Probe (ATP): Code Details

This folder contains implementation code for ATP and some example scripts for usage.

### Quick Check
Before you run anything here, please make sure you install the dependencies listed in the main repo's `README.md`. You can verify the dependencies are installed correctly by running the `test_atp.py` file in this directory.

``` 
(atp) user@server:~/atp-video-language/src$ python test_atp.py
creating ATPConfig...
creating torch tensor inputs...
creating ATPSelectorModel from config...
running forward pass on random inputs...
dumping portion of random output from forward pass:
tensor([[1.0000, 0.8011, 0.0100,  ..., 0.9397, 0.4787, 0.7079],
        [0.5100, 0.2009, 0.1808,  ..., 0.0039, 0.3936, 0.8106],
        [0.9948, 0.5784, 0.5959,  ..., 0.4245, 0.9106, 0.2128],
        [0.7394, 0.3341, 0.1757,  ..., 0.7212, 0.2083, 0.7384],
        [0.8953, 0.9088, 0.2519,  ..., 0.7266, 0.5621, 0.7366]],
       grad_fn=<SumBackward1>)
** success! (note: this script doesn't test cuda/gpu)
```
*(note: the exact values for the random output don't matter)*


### Files Overview

A breakdown of the files provided here:
- `atp.py` - main file with the implementation of `ATPSelectorModel` and supporting classes.
- `atp_utils.py` - supporting file for `atp.py`
- `test_atp.py` - quick script to test your dependencies are set-up correctly
- `data.py` - contains example dataset/dataloader for an ATP training script.
- `training_example.py` - contains (a simple/extensible) example script for training ATP.

*(This list will continue to be updated as needed.)*

For pre-processing and dataset links, please see `data/` folder from the main repo.

For additional details / questions, please email `shyamal@cs.stanford.edu`. Thanks!

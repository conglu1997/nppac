# Non-Parametric Prior Actor-Critic (N-PPAC)

This repository contains the code for

**_On Pathologies in KL-Regularized Reinforcement Learning from Expert Demonstrations_**, Tim G. J. Rudner*, Cong Lu*, Michael A. Osborne, Yarin Gal, Yee Whye Teh. Conference on Neural Information Processing Systems (NeurIPS), 2021.

**Abstract:** KL-regularized reinforcement learning from expert demonstrations has proved successful in improving the sample efficiency of deep reinforcement learning algorithms, allowing them to be applied to challenging physical real-world tasks. However, we show that KL-regularized reinforcement learning with behavioral policies derived from expert demonstrations suffers from hitherto unrecognized pathological behavior that can lead to slow, unstable, and suboptimal online training. We show empirically that the pathology occurs for commonly chosen behavioral policy classes and demonstrate its impact on sample efficiency and online policy performance. Finally, we show that the pathology can be remedied by specifying non-parametric behavioral policies and that doing so allows KL-regularized RL to significantly outperform state-of-the-art approaches on a variety of challenging locomotion and dexterous hand manipulation tasks.

<p align="center">
  <a href="https://openreview.net/forum?id=sS8rRmgAatA">View on OpenReview</a>
</p>

In particular, the code implements:
- Scripts for estimating behavioral reference policies using non-parametric Gaussian processes;
- Scripts for KL-regularized online training using different behavioral expert policies.

## How to use this package
We provide a Docker setup which may be built as follows:
```
docker build -t torch-nppac .
```
If not using docker, please note we used: https://github.com/conglu1997/mj_envs which contains some bug fixes.

To train the GP policies offline:
```
bash exp_scripts/paper_clone_gp.sh
```

To run online training (N-PPAC):
```
bash exp_scripts/paper_configs.sh
```

Pre-trained GP policies using `final_clone_gp.sh` are provided in the folder `nppac/trained_gps/`.

By default, all data will be stored in `data/`.

## Reference

If you found this repository useful, please cite our paper as follows:
```
@inproceedings{
    rudner2021pathologies,
    title={On Pathologies in {KL}-Regularized Reinforcement Learning from Expert Demonstrations},
    author={Tim G. J. Rudner and Cong Lu and Michael A. Osborne and Yarin Gal and Yee Whye Teh},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=sS8rRmgAatA}
}
```

## License
The repository is based on [RLkit](https://github.com/rail-berkeley/rlkit), which may contain further useful scripts. The license for this is contained under the `rlkit/` folder.

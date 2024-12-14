# Portfolio Optimization using Deep Reinforcement Learning

## cleanrl pf4
1. install `pip install -r requirement_cleanrl.txt` && `pip install -r requirements.txt`
2. to run basic pf4
```
python3 cleanrl/pf4.py \
    --dataset-file "./data/brownian_stats.npy" \
    --agent "nn" \
    --save-graph \
    --save-model \
    --total-timesteps 1000000
```
3. models are saved in `/runs/`



[Research Paper](https://ishwargov.github.io/Portfolio_RL_Camera_Ready.pdf)
`Ishwar Govind , Jerry John Thomas, Chandrashekar Lakshminarayan`

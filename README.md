# Graph Isomorphism RL Model

## How to run

```bash
bash run.sh
```

The script creates a Python virtual environment, installs dependencies, trains the model, evaluates it, and writes artifacts to `outputs/`.

## Outputs

- `outputs/results.json`
- `outputs/evaluation_results.csv`
- `outputs/hard_graph_results.csv`
- `outputs/actor_critic_model.pt`
- `outputs/training_loss.png`
- `outputs/evaluation_accuracy.png`
- `outputs/rl_vs_vf2_accuracy.png`
- `logs/run.log`
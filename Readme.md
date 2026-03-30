# Delay- and Resource-Aware: Auto-Scaling Microservice Orchestration in Edge Network via Multiobjective Deep Reinforcement Learning

This repository implements a framework to optimize the auto-scaling  orchestration of  microservices across edge nodes. It employs a decomposed Deep Deterministic Policy Gradient (DDPG) approach to optimize microservice deployment and resource scheduling within Edge Computing environments, focusing on minimizing service latency and maintaining node load balance.


## 📁 Project Structure

- `mian.py`: The main entry point. Handles environment initialization, the training loop, and logging.
- `my_ddpg.py`: Core RL algorithm implementation, including Actor/Critic networks and Reward Memory Shaping logic.
- `EDGE_ENV_my.py`: The simulated edge environment, defining state transitions, reward calculations, and resource consumption.
- `EDGE_DEFINE.py`: Global configuration file containing environment constants and simulation parameters.
- `plotfig_seed.ipynb`: Jupyter Notebook for visualizing training results and comparing different seeds/models.

## 🛠️ Requirements

Ensure you have Python 3.7+ installed. The project depends on the following libraries:

- **TensorFlow**: >= 2.0.0
- **TensorLayer**: >= 2.0.0
- **NumPy**: 1.24.2
- **Matplotlib**: For plotting results

Install dependencies via pip:
```bash
pip install tensorflow>=2.0.0 tensorlayer>=2.0.0 numpy==1.24.2 matplotlib
```
## ⚙️ Parameter Configuration

To customize the simulation or the AI behavior, you can modify the following parameters in their respective files:
1. Environment Configuration (EDGE_DEFINE.py): These parameters define the scale of the edge computing network and its resource properties.
---
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `USER_NUM` | `15` | Number of mobile terminal users generating requests. |
| `NODE_NUM` | `4` | Number of available Edge Computing nodes (Servers). |
| `MS_NUM` | `10` | Total types of Microservices available in the system. |
| `APP_CLASS` | `8` | Number of unique Application Service Chains. |
| `RANDOMSEED` | `1037` | Seed used to ensure deterministic results across runs.|
| `cpu_rate` | `1` | Weight of CPU resources when calculating the state vector. |
| `memory_rate` | `0.01` | Weight of Memory resources when calculating the state vector. |
---

2. RL Hyperparameters (my_ddpg.py): These parameters control the training efficiency and convergence of the DDPG agent.
---
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `LR_A` | `0.005` | Learning rate for the Actor network (Policy). |
| `LR_C` | `0.005` | Learning rate for the Critic network (Value estimation). |
| `GAMMA` | `0.95` | Discount factor ($\gamma$) for future rewards.. |
| `TAU` | `0.01` | Soft update coefficient for Target Network synchronization. |
| `MEMORY_CAPACITY` | `10000` | Size of the Replay Buffer for storing experiences.|
| `BATCH_SIZE` | `32` | Number of experience samples per training step. |
| `MAX_EPISODES` | `500` | Total number of training iterations. |
| `VAR` | `8` | Initial exploration noise variance (decays during training). |
| `HIDDEN_SIZE` | `32` | Number of units in the neural network hidden layers. |
---

## 🚀 Usage Guide
Run the primary script to begin the training process:
```
python mian.py
```
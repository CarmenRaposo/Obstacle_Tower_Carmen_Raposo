import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import numpy as np


# Globals
#Ts, rewards, Qs, floors, best_avg_reward = [], [], [], [], -1e10
Ts, rewards, floors, best_avg_reward = [], [], [], -1e10


# Test
def test(T, model, evaluate=False, realtime=True, env=None):

    global Ts, rewards, best_avg_reward

    evaluation_episodes = 10

    Ts.append(T)
    T_rewards = []
    T_floors = []

    # Test performance over several episodes
    done = True
    for _ in range(evaluation_episodes):
        while True:
            if done:
                state = env.reset()
                reward_sum = 0
                floors_sum = 0
                done = False

            next_action, next_state = model.predict(state)
            state, reward, done, _ = env.step(next_action)  # Step
            reward_sum += reward
            if reward > 0.99:
                floors_sum += 1

            if done:
                T_rewards.append(reward_sum)
                T_floors.append(floors_sum)
                break

    avg_reward = sum(T_rewards) / len(T_rewards)
    avg_floor = sum(T_floors) / len(T_floors)

    if not evaluate:
        # Append to results
        rewards.append(T_rewards)
        floors.append(T_floors)

        # Plot
        _plot_line(Ts, rewards, 'Reward', path='results')
        _plot_line(Ts, floors, 'Floors', path='results')

        # Save model parameters if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            model.save('results')

    # Return average reward and floor
    return avg_reward, avg_floor


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
        1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.',
                          showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour),
                         name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent),
                          name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)

Reinforcement learning
========================

- is an agent that relies on reward and punishment system
- to determine the reward we use bellman equation

![alt text](image.png)

Reward
=======

- Living (you stay too long you get punishment)
 - R(s) = 0 normal exploration
 - R(s) = -0.04    make a path long or not long
 - R(s) = -0.5     get to end quickly
 - R(s) = -10      taking punishment path (fire) is better than longer path 
- sparse (no reward until you get to the goal)
- dense (the closer you are to the goal the better reward you get)


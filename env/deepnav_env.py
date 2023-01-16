from pettingzoo.utils.env import ParallelEnv
import rvo2
from Circle import Circle
import numpy as np

from typing import Optional, Dict
from gymnasium.spaces import Box
class DeepNav(ParallelEnv):
    def __init__(self, width=255, height= 255, timestep : float = 0.25 , neighbor_dists : float = 1.0, 
                 time_horizont : float=10.0, time_horizont_obst : float = 20.0, radius : float=2.0, 
                 max_speed : float=3.5) -> None:
        super().__init__()
        
        self.timestep = timestep
        self.T = None
        self.neighbor_dists = neighbor_dists
        self.max_neig = self.num_agents - 1
        self.time_horizont = time_horizont
        self.time_horizont_obst = time_horizont_obst
        self.radius = radius
        self.max_speed = max_speed
        self.pos, self.goals = Circle(self.num_agents).getAgentPosition()
        self.obs = None
        self.sim = rvo2.PyRVOSimulator(self.timestep, self.neighbor_dists, self.max_neig, 
                    self.time_horizont, self.time_horizont_obst, self.radius, self.max_speed)
        self.height = height
        self.width = width

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.T = 0
        self.pos, self.goals = Circle(self.num_agents).getAgentPosition()
        return{
            a : self.getAgentState(a)
            for a in range(self.num_agents)
        }
    
    def getAgentState(self, agent):

        dirs = [0, 45, 90, 135, 180, 225, 270, 315]
        a_pos = self.pos[agent]
        state = [None] * 10
        state [0] = a_pos[0]
        state [1] = a_pos[1]
        for poss in self.pos:
            x = a_pos[0] - poss[0]
            y = a_pos[1] - poss[1]
            ang = np.degrees(np.arctan(y/x))
            norm = np.linalg.norm((x, y))
            ang_int = np.degrees(np.arcsin(self.radius/norm))
            ang_inf = np.round((ang - ang_int) + 0.5)
            ang_sup = np.round((ang + ang_int) - 0.5)
            ang_range = list(range(ang_inf, ang_sup + 1))
            for indx, a in enumerate(dirs):
                if a in ang_range:
                    state[indx + 2] = norm - self.radius
        # Missing no collision measurements
        for i in range(len(state)):
            if state[i] is None:
                x_dist = a_pos[0] - self.width
                y_dist = a_pos[1] - self.height
                state[i] = np.linalg.norm([x_dist, y_dist])
        return state

    def step(self, actions ):
        self.take_action(actions)

        return{a : self.getAgentState(a)
                for a in (self.num_agents)}, self.calculate_rewards(), self.is_done()[0]
    
    def calculate_dist(self, x1, x2, y1, y2):
        return np.linalg.norm((x1 - x2, y1- y2))
    
    def is_done(self) -> tuple:
        if self.T > self.time_horizont:
            return True, False
        for i in range(self.num_agents):
            if not self.is_done(i):
                return False, False
        
        return True, True

    def is_done(self, agent):
        pos_agnt = self.pos[agent]
        pos_goal = self.goals[agent]

        if self.calculate_dist(pos_agnt[0], pos_goal[0], pos_agnt[1], pos_goal[1]) <= self.radius:
            return True
        return False
    def take_action(self, actions):
        actions = np.squeeze(np.array(actions))
        for i in range(self.num_agents):
            self.sim.setAgentPrefVelocity(i, tuple(actions[i]))
        self.sim.doStep()
        self.pos = [self.sim.getAgentPos(i) for i in range(self.num_agents)]

    def calculate_rewards(self):
        rwd_glbl = 0
        rwd_lcal = np.zeros(self.num_agents)
        done, success = self.is_done()
        if done and success:
            rwd_glbl += 50
            return rwd_glbl + rwd_lcal

        if done and not success:
            rwd_glbl -= 10
            return rwd_glbl + rwd_lcal

        for i in range(self.num_agents):
            pos_agnt = self.pos[i]
            pos_goal = self.goals[i]
            rwd_lcal[i] -= self.calculate_dist(pos_agnt[0], pos_goal[0], pos_agnt[1], pos_goal[1])
            rwd_lcal += 10 * self.is_done(i)
        
        return rwd_lcal

    def render(self):
        pass

    def observation_space(self, agent):
        return Box(-255.0, 255.0, 9)

    def action_space(self, agent):
        return Box(-1.0, 1.0, 2)
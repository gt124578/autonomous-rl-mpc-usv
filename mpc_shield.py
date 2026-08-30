import numpy as np
from scipy.optimize import minimize
from USV_dynamics_model import USVParameters, USVDynamics

class PredictiveSafetyFilter:
    def __init__(self, params):
        self.params = params
        self.dynamics = USVDynamics(self.params)
        self.action_bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    def _cost_function(self, action_mpc, action_rl):
        """
        Objective: Find an MPC action that is as close as possible to the RL action.
        J = || u_mpc - u_rl ||^2
        """
        return np.sum((action_mpc - action_rl)**2)

    def _safety_constraint(self, action_mpc, current_state, obs_x, obs_y, obs_vx, obs_vy):
        """
        Predicts future states (horizon = 2 steps) for the USV and the obstacle.
        Ensures the minimum distance threshold (2.0) is respected.
        """
        # Predict USV state 2 steps ahead
        next_state, _ = self.dynamics.step(current_state, action_mpc)
        next_state_2, _ = self.dynamics.step(next_state, action_mpc)
        
        # Predict obstacle state 2 steps ahead
        future_obs_x = obs_x + (2.0 * obs_vx)
        future_obs_y = obs_y + (2.0 * obs_vy)
        
        # True 2D distance between future USV and future obstacle
        dist = np.sqrt((next_state_2[0] - future_obs_x)**2 + (next_state_2[1] - future_obs_y)**2)
        
        # Constraint is valid if > 0 (dist - 2.0 > 0)
        return dist - 2.0

    def get_safe_action(self, current_state, action_rl, obs_x, obs_y, obs_vx=0.0, obs_vy=0.0):
        """
        Runs SLSQP optimizer to filter unsafe RL actions.
        """
        constraints = {
            'type': 'ineq',
            'fun': lambda u_mpc: self._safety_constraint(u_mpc, current_state, obs_x, obs_y, obs_vx, obs_vy)
        }
        
        result = minimize(
            fun=self._cost_function,
            x0=action_rl,
            args=(action_rl,),
            bounds=self.action_bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 30, 'disp': False}
        )

        if result.success:
            return result.x
        else:
            # Fallback safe action (e.g. full reverse/stop) if optimizer fails
            return np.array([-1.0, -1.0])

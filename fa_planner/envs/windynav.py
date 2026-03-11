import numpy as np
from fa_planner.rendering import render_world_frame, crop_and_resize, world_to_pixel


class WindyNavEnv:
    def __init__(self, cfg):
        self.dt = cfg["dt"]
        self.T = cfg["T"]
        self.gamma = cfg["gamma"]
        self.v_max = cfg["v_max"]
        self.a_max = cfg["a_max"]
        self.w_max = cfg["w_max"]
        self.k_max = cfg["k_max"]
        self.bounce_beta = cfg["bounce_beta"]
        self.success_radius = cfg["success_radius"]
        self.size = cfg.get("image_size", 64)
        self.ego_crop = cfg.get("ego_crop", 32)
        self.edge_margin = cfg.get("edge_margin", 0.08)
        self.grid_spacing_px = cfg.get("grid_spacing_px", 4)
        self.grid_width = cfg.get("grid_width", 1)
        self.region_count = cfg.get("region_count", 4)
        self.region_w_scale = cfg.get("region_w_scale", 0.6)
        self.region_colors = cfg.get(
            "region_colors",
            [
                (60, 70, 80),
                (70, 80, 70),
                (80, 70, 60),
                (70, 60, 80),
            ],
        )
        if len(self.region_colors) != self.region_count:
            raise ValueError("region_colors must match region_count")
        self.state = None
        self.goal = None
        self.region_w0 = None
        self.region_k = None
        self.t = 0

    def reset(self, seed=None, goal=None, wind=None, start_mode="edge", goal_mode="edge"):
        if seed is not None:
            np.random.seed(seed)
        if start_mode == "edge":
            pos = np.array(
                [np.random.uniform(-0.8, 0.8), 1.0 - self.edge_margin],
                dtype=np.float32,
            )
        elif start_mode == "random":
            pos = np.random.uniform(-0.8, 0.8, size=(2,)).astype(np.float32)
        else:
            raise ValueError("Unknown start_mode")
        vel = np.zeros(2, dtype=np.float32)
        self.state = np.concatenate([pos, vel]).astype(np.float32)
        if goal is None:
            if goal_mode == "edge":
                self.goal = np.array(
                    [np.random.uniform(-0.8, 0.8), -1.0 + self.edge_margin],
                    dtype=np.float32,
                )
            elif goal_mode == "random":
                self.goal = np.random.uniform(-0.8, 0.8, size=(2,)).astype(np.float32)
            else:
                raise ValueError("Unknown goal_mode")
        else:
            self.goal = np.array(goal, dtype=np.float32)
        if wind is None:
            w_scale = self.region_w_scale
            w0 = np.random.uniform(-self.w_max * w_scale, self.w_max * w_scale, size=(self.region_count, 2))
            k = np.zeros((self.region_count, 2), dtype=np.float32)
        else:
            w0, k = wind
            w0 = np.array(w0, dtype=np.float32)
            k = np.array(k, dtype=np.float32)
            if w0.shape == (2,) and k.shape == (2,):
                w0 = np.tile(w0, (self.region_count, 1))
                k = np.tile(k, (self.region_count, 1))
        self.region_w0 = w0.astype(np.float32)
        self.region_k = k.astype(np.float32)
        self.t = 0
        return self.state.copy()

    def _region_index(self, y):
        idx = int(((y + 1.0) / 2.0) * self.region_count)
        return min(max(idx, 0), self.region_count - 1)

    def _wind_at(self, y, t):
        idx = self._region_index(y)
        w = self.region_w0[idx]
        return np.clip(w, -self.w_max, self.w_max)

    def current_wind(self):
        return self._wind_at(self.state[1], self.t)

    def step(self, action):
        action = np.clip(action, -self.a_max, self.a_max)
        x, y, vx, vy = self.state
        v = np.array([vx, vy], dtype=np.float32)
        v = self.gamma * v + action * self.dt
        v = np.clip(v, -self.v_max, self.v_max)
        w = self._wind_at(y, self.t)
        p = np.array([x, y], dtype=np.float32)
        p = p + (v + w) * self.dt
        p, v = self._handle_bounds(p, v)
        self.state = np.concatenate([p, v]).astype(np.float32)
        self.t += 1
        done = self.t >= self.T
        dist = np.linalg.norm(p - self.goal)
        success = dist <= self.success_radius
        return self.state.copy(), success, done, {"wind": w.copy(), "dist": dist}

    def _handle_bounds(self, p, v):
        x, y = p
        vx, vy = v
        if abs(x) > 1.0:
            x = np.sign(x) * 1.0
            vx = -self.bounce_beta * vx
        if abs(y) > 1.0:
            y = np.sign(y) * 1.0
            vy = -self.bounce_beta * vy
        return np.array([x, y], dtype=np.float32), np.array([vx, vy], dtype=np.float32)

    def render_world(self, show_goal=True):
        x, y, vx, vy = self.state
        frame = render_world_frame(
            size=self.size,
            agent_pos=(x, y),
            goal_pos=self.goal,
            velocity=(vx, vy),
            show_goal=show_goal,
            region_colors=self.region_colors,
            grid_spacing_px=self.grid_spacing_px,
            grid_width=self.grid_width,
        )
        return frame

    def render_ego(self, world_frame=None):
        if world_frame is None:
            world_frame = self.render_world(show_goal=True)
        agent_px = world_to_pixel(self.state[:2], world_frame.shape[0])
        return crop_and_resize(world_frame, agent_px, self.ego_crop, world_frame.shape[0])

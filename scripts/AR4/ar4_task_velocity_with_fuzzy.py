"""
ar4_task_velocity_with_fuzzy.py

Task-space velocity controller + fuzzy supervisor for AR4 in Isaac Lab / Isaac Sim.

Features:
 - Uses Differential IK (Jacobians) to convert desired end-effector twist -> joint velocities.
 - Adds null-space posture motion: qdot = J_pinv * v_task + (I - J_pinv * J) * qdot_posture
 - Fuzzy supervisor observes stability metrics and outputs a posture_scale in [0,1]
 - Small online adaptation step nudges fuzzy membership centers to reduce a stability loss

Notes:
 - Adapt ARTICULATION_PATH to the AR4 prim path in your stage (e.g., "/World/ar4")
 - The code uses isaaclab controllers / articulation utils. Depending on your IsaacLab/IsaacSim version
   you may need to adapt import paths; see IsaacLab docs for DifferentialIKController and Articulation API.
   (Docs & examples: IsaacLab Differential IK / task-space controller).
   References: Ekumen AR4 Isaac integration and IsaacLab DifferentialIK examples. 
"""

import numpy as np
import time
import math

# Fuzzy
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# IsaacLab / Isaac Sim imports (adapt if your install uses slightly different module paths)
# The isaaclab docs show controllers.DifferentialIKController and Articulation interaction.
try:
    from isaaclab.controllers import DifferentialIKController  # preferred high-level controller
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core import World
except Exception:
    # fallback imports (some IsaacLab versions use different paths)
    from isaaclab.controllers import DifferentialIKController
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core import World

# ---------- USER CONFIG ----------
ARTICULATION_PATH = "/World/ar4"   # change to your AR4 USD path on the stage
EE_LINK_NAME = "ee_link"          # the end-effector link name (change if different)
DT = 0.02                         # control timestep (s)

# stability metric defaults (you can replace these with the metrics you prefer)
USE_METRICS = ["ee_error", "manipulability", "max_joint_vel", "max_torque_est"]


# ---------- Utilities ----------
def manipulability_index(J):
    # Yoshikawa manipulability: sqrt(det(J * J^T)) for a 6xN jacobian (task-space)
    # numeric fallback: sqrt(prod(singular_values)) or simply product of singular values
    if J is None:
        return 0.0
    try:
        s = np.linalg.svd(J, compute_uv=False)
        # avoid zero/negative underflow; use geometric mean-like scalar
        return float(np.prod(s)**(1.0 / max(1, len(s))))
    except Exception:
        return 0.0


def compute_stability_loss(metrics, weights=None):
    # simple stability loss to minimize: combination of tracking error, high torques, low manipulability
    if weights is None:
        weights = {"ee_error": 1.0, "manipulability": -0.5, "max_joint_vel": 0.2, "max_torque_est": 0.5}
    loss = 0.0
    loss += weights.get("ee_error", 0.0) * metrics.get("ee_error", 0.0)
    # manipulability should *reduce* loss when higher -> negative weight
    loss += weights.get("manipulability", 0.0) * metrics.get("manipulability", 0.0)
    loss += weights.get("max_joint_vel", 0.0) * metrics.get("max_joint_vel", 0.0)
    loss += weights.get("max_torque_est", 0.0) * metrics.get("max_torque_est", 0.0)
    return loss


# ---------- Build fuzzy supervisor (2 inputs -> posture_scale) ----------
# Inputs: ee_error (m), manipulability index (unitless), max_joint_vel (rad/s), max_torque_est (Nm)
ee_err = ctrl.Antecedent(np.linspace(0.0, 0.2, 7), 'ee_err')      # 0..0.2 m
manip = ctrl.Antecedent(np.linspace(0.0, 1.0, 7), 'manip')        # 0..1 normalized
jvel = ctrl.Antecedent(np.linspace(0, 2.0, 7), 'jvel')           # 0..2 rad/s
torq = ctrl.Antecedent(np.linspace(0.0, 5.0, 7), 'torq')         # estimated torque range (adjust)

posture_scale = ctrl.Consequent(np.linspace(0.0, 1.0, 7), 'posture_scale')

# membership functions (triangular / trapezoid)
ee_err['small'] = fuzz.trimf(ee_err.universe, [0.0, 0.0, 0.04])
ee_err['med']   = fuzz.trimf(ee_err.universe, [0.03, 0.07, 0.12])
ee_err['large'] = fuzz.trimf(ee_err.universe, [0.1, 0.2, 0.2])

manip['low'] = fuzz.trimf(manip.universe, [0.0, 0.0, 0.3])
manip['med'] = fuzz.trimf(manip.universe, [0.2, 0.45, 0.7])
manip['high']= fuzz.trimf(manip.universe, [0.6, 1.0, 1.0])

jvel['low'] = fuzz.trimf(jvel.universe, [0.0, 0.0, 0.6])
jvel['high']= fuzz.trimf(jvel.universe, [0.5, 2.0, 2.0])

torq['low'] = fuzz.trimf(torq.universe, [0.0, 0.0, 1.0])
torq['high']= fuzz.trimf(torq.universe, [0.8, 5.0, 5.0])

# posture_scale: how strongly we apply null-space posture motion
posture_scale['low']  = fuzz.trimf(posture_scale.universe, [0.0, 0.0, 0.3])
posture_scale['med']  = fuzz.trimf(posture_scale.universe, [0.2, 0.5, 0.8])
posture_scale['high'] = fuzz.trimf(posture_scale.universe, [0.6, 1.0, 1.0])

# rules (interpretable)
rules = [
    ctrl.Rule(ee_err['large'] | jvel['high'] | torq['high'], posture_scale['low']),
    ctrl.Rule(manip['low'] & ee_err['small'], posture_scale['med']),
    ctrl.Rule(manip['high'] & ee_err['small'] & jvel['low'] & torq['low'], posture_scale['high']),
]

fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)


# ---------- Main controller class ----------
class AR4TaskVelocityController:
    def __init__(self, articulation_path, ee_link_name, dt=DT):
        self.world = World(stage_units_in_meters=True)  # wrapper to access stage/world
        self.dt = dt

        # Find the articulation in the world
        self.art = Articulation(articulation_path)  # high-level wrapper
        # number of actuated joints
        self.joint_names = self.art.get_joint_names()
        self.nq = len(self.joint_names)

        # Differential IK controller (uses IsaacLab DifferentialIK internally)
        self.diffik = DifferentialIKController(articulation_prim_path=articulation_path,
                                                body_name=ee_link_name,
                                                command_type="pose",  # we will feed twist/pose deltas
                                                ik_method="pinv",
                                                ik_params={"damping": 0.01, "pinv_rcond": 1e-4})
        # posture target (joint angles) — initialize to current pose
        self.q_target_posture = np.array(self.art.get_joint_positions(), dtype=np.float64)
        self.posture_gain = 1.0   # used to compute qdot_posture (proportional)
        # logger / buffers
        self.prev_time = time.time()

        # small adaptation step memory
        self.last_loss = None
        self.adapt_step = 0.001  # tiny step to nudge membership centers

    def get_state(self):
        q = np.array(self.art.get_joint_positions(), dtype=np.float64)
        qd = np.array(self.art.get_joint_velocities(), dtype=np.float64)
        # end-effector pose & velocity (use Articulation helpers)
        ee_pose = self.art.get_link_world_pose(EE_LINK_NAME)
        ee_vel = self.art.get_link_world_velocity(EE_LINK_NAME)
        # simple torque estimate placeholder (Isaac may provide joint efforts if configured)
        efforts = np.abs(np.array(self.art.get_joint_efforts() or [0.0]*self.nq, dtype=np.float64))
        return {"q": q, "qd": qd, "ee_pose": ee_pose, "ee_vel": ee_vel, "efforts": efforts}

    def task_to_joint_velocity(self, v_task, q, qd):
        """
        Convert a 6-dim end-effector twist (vx,vy,vz, wx,wy,wz) in root frame to joint velocities:
            qdot_task = J_pinv * v_task
            qdot_posture = Kp * (q_target_posture - q)
            qdot = qdot_task + N * qdot_posture  where N = (I - J_pinv * J)
        """
        # request Jacobian from Articulation / DifferentialIK utilities
        # The DifferentialIKController in IsaacLab can compute the required joint delta directly;
        # here we show an explicit Jacobian path for clarity and to insert null-space posture control.
        J = self.art.compute_geometric_jacobian(link_name=EE_LINK_NAME)  # shape (6, nq)
        if J is None:
            # fallback: ask DifferentialIKController to compute a joint delta for a small pose step
            dq = self.diffik.compute_joint_delta_from_twist(v_task, q, qd)
            return np.array(dq, dtype=np.float64)

        # compute pseudoinverse with damping
        lam = 1e-4
        JT = J.T
        # damped least squares pseudo-inverse (Moore-Penrose Tikhonov)
        try:
            JJT = J @ JT + lam * np.eye(6)
            J_pinv = JT @ np.linalg.inv(JJT)
        except np.linalg.LinAlgError:
            J_pinv = np.linalg.pinv(J)

        qdot_task = J_pinv @ v_task  # shape (nq,)

        # posture velocity (simple proportional toward desired posture)
        qdot_posture = self.posture_gain * (self.q_target_posture - q)

        # null-space projection
        N = np.eye(self.nq) - J_pinv @ J
        qdot = qdot_task + N @ qdot_posture

        return qdot

    def compute_metrics(self, state, qdot_cmd):
        # compute the stability metrics used by the fuzzy controller
        # ee tracking error (we assume a reference pose - for demo, reference is zero pose delta)
        # in practice provide ee_ref from planner / trajectory generator
        ee_pos = np.array(state["ee_pose"].position)
        ee_ref_pos = np.array([0.0, 0.0, 0.5])  # placeholder reference position - replace with your planner
        ee_err = float(np.linalg.norm(ee_pos - ee_ref_pos))

        # manipulability
        J = self.art.compute_geometric_jacobian(link_name=EE_LINK_NAME)
        manip = manipulability_index(J) if J is not None else 0.0

        max_jvel = float(np.max(np.abs(state["qd"])))
        max_torque_est = float(np.max(np.abs(state["efforts"]))) if state["efforts"].size else 0.0

        metrics = {"ee_error": ee_err, "manipulability": manip,
                   "max_joint_vel": max_jvel, "max_torque_est": max_torque_est}
        return metrics

    def fuzzy_posture_scale(self, metrics):
        # feed fuzzy inputs (clamp to membership ranges)
        fuzzy_sim.input['ee_err'] = np.clip(metrics['ee_error'], 0.0, 0.2)
        fuzzy_sim.input['manip'] = np.clip(metrics['manipulability'], 0.0, 1.0)
        fuzzy_sim.input['jvel'] = np.clip(metrics['max_joint_vel'], 0.0, 2.0)
        fuzzy_sim.input['torq'] = np.clip(metrics['max_torque_est'], 0.0, 5.0)
        fuzzy_sim.compute()
        return float(fuzzy_sim.output['posture_scale'])

    def online_adapt_fuzzy(self, metrics, current_loss):
        """
        Tiny online adaptation: if loss increased, gently nudge membership centers to try to reduce loss.
        This is intentionally conservative: only tiny per-loop nudges to avoid destabilizing the controller.
        You can replace this with a stronger optimizer offline (CMA-ES, ANFIS training, gradient descent, etc).
        """
        if self.last_loss is None:
            self.last_loss = current_loss
            return

        # simple rule: if loss increased -> reduce aggressiveness of posture (shift ee_err membership slightly)
        if current_loss > self.last_loss:
            # shift 'ee_err' small/med boundaries slightly toward smaller values to be conservative
            # (We update the trimmed membership functions by modifying the universe anchors)
            # NOTE: scikit-fuzzy Antecedent objects do not expose direct setters for triangle points,
            # so for a quick hack we re-create the member functions with slightly shifted support.
            shift = self.adapt_step
            # Rebuild ee_err memberships with small shift; for simplicity we only update the 'small' MF
            new_small = fuzz.trimf(ee_err.universe, [0.0, 0.0, 0.04 - shift])
            ee_err['small'] = new_small
            # (In a robust implementation you'd maintain explicit parameter arrays and update them more cleanly)
        else:
            # if loss decreased, we could slowly relax (not implemented here to remain simple)
            pass

        self.last_loss = current_loss

    def step(self, v_task):
        """
        One control step:
         - read state
         - compute joint velocity command from task twist + posture nullspace
         - compute metrics, run fuzzy to get posture_scale
         - scale null-space term by posture_scale and send velocities
        """
        state = self.get_state()
        q = state["q"]
        qd = state["qd"]

        # compute raw qdot (task + nullspace)
        qdot_raw = self.task_to_joint_velocity(v_task, q, qd)

        # compute metrics and posture scale
        metrics = self.compute_metrics(state, qdot_raw)
        posture_scale = self.fuzzy_posture_scale(metrics)

        # recompute qdot with scaled posture term:
        # we'll re-run task->joint transform but scale the posture component explicitly
        # compute J/J_pinv again (for clarity; could refactor to avoid duplicate compute)
        J = self.art.compute_geometric_jacobian(link_name=EE_LINK_NAME)
        if J is None:
            qdot_final = qdot_raw
        else:
            # compute pinv
            JT = J.T
            lam = 1e-4
            try:
                JJT = J @ JT + lam * np.eye(6)
                J_pinv = JT @ np.linalg.inv(JJT)
            except np.linalg.LinAlgError:
                J_pinv = np.linalg.pinv(J)
            qdot_task = J_pinv @ v_task
            qdot_posture = self.posture_gain * (self.q_target_posture - q)
            N = np.eye(self.nq) - J_pinv @ J
            qdot_final = qdot_task + posture_scale * (N @ qdot_posture)

        # send qdot_final as velocity command to the articulation
        # Articulation likely has a set_joint_velocity API; use it (adapt if your API differs)
        self.art.set_joint_velocity_targets(self.joint_names, qdot_final.tolist())  # adapt API name to your version

        # tiny online adaptation
        loss = compute_stability_loss(metrics)
        self.online_adapt_fuzzy(metrics, loss)

        return {"qdot": qdot_final, "metrics": metrics, "posture_scale": posture_scale}


# ---------- Example usage ----------
def main_loop():
    ctl = AR4TaskVelocityController(ARTICULATION_PATH, EE_LINK_NAME, DT)

    # simple constant task twist: move forward in z slowly (0.01 m/s)
    # twist expressed as [vx, vy, vz, wx, wy, wz]
    v_task = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])

    try:
        while True:
            t0 = time.time()
            out = ctl.step(v_task)
            # logging for debugging
            print(f"posture_scale={out['posture_scale']:.3f} metrics={out['metrics']}")
            # step sim - in IsaacLab context your simulation loop/runner will advance the world.
            # Here we simply sleep for the control dt; the actual sim stepping is handled elsewhere.
            dt = max(0.0, DT - (time.time() - t0))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Controller stopped.")


if __name__ == "__main__":
    main_loop()

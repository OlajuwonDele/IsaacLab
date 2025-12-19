import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# variables
ee_err = ctrl.Antecedent(np.linspace(0, 0.2, 5), 'ee_err')         # meters
jvel = ctrl.Antecedent(np.linspace(0, 2.0, 5), 'jvel')            # rad/s
scale = ctrl.Consequent(np.linspace(0, 1.0, 5), 'scale')

# membership functions (triangular)
ee_err['small'] = fuzz.trimf(ee_err.universe, [0.0, 0.0, 0.05])
ee_err['med']   = fuzz.trimf(ee_err.universe, [0.03, 0.06, 0.10])
ee_err['large'] = fuzz.trimf(ee_err.universe, [0.08, 0.2, 0.2])

jvel['low']  = fuzz.trimf(jvel.universe, [0.0, 0.0, 0.8])
jvel['high'] = fuzz.trimf(jvel.universe, [0.5, 2.0, 2.0])

scale['low']  = fuzz.trimf(scale.universe, [0.0, 0.0, 0.4])
scale['med']  = fuzz.trimf(scale.universe, [0.2, 0.5, 0.8])
scale['high'] = fuzz.trimf(scale.universe, [0.6, 1.0, 1.0])

# rules
rules = [
    ctrl.Rule(ee_err['large'] | jvel['high'], scale['low']),
    ctrl.Rule(ee_err['med'] & ~jvel['high'], scale['med']),
    ctrl.Rule(ee_err['small'] & jvel['low'], scale['high']),
]

fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# usage
def fuzzy_supervisor(ee_error, max_joint_vel):
    fuzzy_sim.input['ee_err'] = ee_error
    fuzzy_sim.input['jvel'] = max_joint_vel
    fuzzy_sim.compute()
    return float(fuzzy_sim.output['scale'])

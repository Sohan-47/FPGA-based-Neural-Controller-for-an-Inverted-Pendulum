import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import csv
import numpy as np

# constants
SCALE_FACTOR = 2048 
CLK_PERIOD_NS = 10  

# parameters
g = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5 
polemass_length = (masspole * length)
tau = 0.02        

def update_physics(state, force_int, step_idx):
    x, x_dot, theta, theta_dot = state
    
    # The verilog logic is Q11 (Scale 2048).
    # Therefore, 2048 = 1.0 (Full Action) = 10 Newtons.
    
    #  normalize based on Q11 scale
    action_val = force_int / 2048.0
    
    #  clip to -1.0 to 1.0 (just to be safe)
    action_val = max(min(action_val, 1.0), -1.0)
    
    #  Convert to force
    force = action_val * 10.0
    
    
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (g * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    
    return [x, x_dot, theta, theta_dot]

def to_fixed(float_val):
    int_val = int(round(float_val * SCALE_FACTOR))
    int_val = max(min(int_val, 32767), -32768)
    return int_val

@cocotb.test()
async def run_rocket_simulation(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    
    dut.rst.value = 1
    dut.x_in.value = 0
    dut.x_dot_in.value = 0
    dut.theta_in.value = 0
    dut.theta_dot_in.value = 0
    
    # reset sequence
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0  
    await RisingEdge(dut.clk)
    
    print("\n--- SIMULATION STARTED ---")
    print(f"Params: Scale=2048, Continuous Force (+/- 10N)")

    # initial tilt
    state = [0.0, 0.0, 0.2, 0.0]  # 0.05 rad = ~2.8 degrees
    data_file = open('flight_data.csv', 'w', newline='')
    writer = csv.writer(data_file)
    writer.writerow(['step', 'x', 'theta', 'force'])

    for i in range(1000):
        # update i/p
        dut.x_in.value = to_fixed(state[0])
        dut.x_dot_in.value = to_fixed(state[1])
        dut.theta_in.value = to_fixed(state[2])
        dut.theta_dot_in.value = to_fixed(state[3])

        await RisingEdge(dut.clk)
        
        try:
            val = int(dut.force_out.value)
        except ValueError:
            print(f"WARNING: Seen 'X' at step {i}, assuming 0.")
            val = 0
            
        if val > 32767: val -= 65536
            
        state = update_physics(state, val, i)

        deg = np.degrees(state[2])
        force_n = (val / 32767.0) * 10.0
        writer.writerow([i, state[0], state[2], force_n])
        
        # log 
        if i % 50 == 0 or abs(deg) > 1.0:
            force_n = (val / 32767.0) * 10.0
            print(f"Step {i:4d} | Pos: {state[0]:.4f} | Tilt: {deg:.2f} deg | Force: {force_n:.2f} N")
        
        if abs(deg) > 15.0:
             print(f"\n*** FAILURE: Rocket crashed at step {i} ***")
             assert False, "Simulation failed due to crash."

    print("\n*** SUCCESS: Rocket Balanced! ***")
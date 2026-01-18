import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# graphics
CART_WIDTH = 0.5
CART_HEIGHT = 0.3
POLE_LENGTH = 1.0  # visual length (not physics length)

try:
    df = pd.read_csv('flight_data.csv')
except FileNotFoundError:
    print("Error: flight_data.csv not found. Run the simulation first!")
    exit()


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-0.5, 2.0)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title("FPGA Neural Network Control Replay")
ax.set_xlabel("Position (m)")

# Drawing Elements
ground = ax.hlines(0, -2.4, 2.4, colors='black', linewidth=2)
cart = patches.Rectangle((0, 0), CART_WIDTH, CART_HEIGHT, fc='blue', alpha=0.8)
pole_line, = ax.plot([], [], 'r-', linewidth=4)
force_arrow = ax.arrow(0, 0, 0, 0, head_width=0.1, color='green')
info_text = ax.text(-2.2, 1.5, '', fontsize=12)

def init():
    ax.add_patch(cart)
    return cart, pole_line, info_text

def animate(i):

    row = df.iloc[i]
    x = row['x']
    theta = row['theta']
    force = row['force']
    
    # update cart
    cart_x = x - CART_WIDTH / 2
    cart.set_xy((cart_x, 0))
    
    # update pole
    pole_x_start = x
    pole_y_start = CART_HEIGHT
    pole_x_end = x + POLE_LENGTH * np.sin(theta)
    pole_y_end = CART_HEIGHT + POLE_LENGTH * np.cos(theta)
    pole_line.set_data([pole_x_start, pole_x_end], [pole_y_start, pole_y_end])
    
    # update force arrow
    arrow_len = force * 0.1 
    
    # update info
    info_text.set_text(f"Step: {int(row['step'])}\n"
                       f"Pos: {x:.3f} m\n"
                       f"Tilt: {np.degrees(theta):.2f} deg\n"
                       f"Force: {force:.2f} N")
    
    return cart, pole_line, info_text


# animation : interval=20ms i.e. 50 FPS, matches 0.02s physics step
ani = animation.FuncAnimation(fig, animate, frames=len(df), 
                              init_func=init, interval=20, blit=False)

plt.show()
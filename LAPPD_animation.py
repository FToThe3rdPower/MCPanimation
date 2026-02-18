#!/usr/bin/python3
# -*- coding: utf-8 -*-
# LAPPD Photon Detection Animation
# Authored by Trey, based on Daan Oppenhuis' pion in a Si detector animation,
#    Updated to include more physics with the help of Claude and Gemini

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import time

# gif file info
filePath = './'
fileName = f"{filePath}LAPPD_test.gif"

# Graphics params
numFrames = 200  # reduce for testing
fps = 30

# Particle coloring
photonColor = 'lightblue'
photoelectronColor = 'gold'

# Layer colors
topColor = "white"
photocathodeColor = 'darkgreen'
mcpBodyColor = 'grey'
channelColor = 'cornflowerblue'
backplateColor = 'brown'
vacuumColor = 'lavender'

# Layer thicknesses
abovePhotocathode = 2000
photocathodeThick = 400
vacuumMiddle = 1190
mcpThick = 1500
mcp1to2gap = 1000
vacuumBottom = 2500
backplateThick = 2000

# Plotframe parameters
height = abovePhotocathode + photocathodeThick + vacuumMiddle + mcpThick + mcp1to2gap + mcpThick + vacuumBottom + backplateThick
width = (height/3)*4

# Channel parameters
channelAngle = 20
channelDiameter = 700
numChannels = 12
wallTolerance = 30  # Increased to catch fast-moving electrons
labelSpacingFactor = 1.5

# Physics parameters
photonSize = 6
electronSize = 3
photonVelocity = 50
electronVelocity = 40
e_field_acceleration = 5  
electronsPerCollision = 3
MAX_ELECTRONS = 500      # Reduce for testing

# Calculate layer positions
emptyTopStart = 0
photocathodeTop = abovePhotocathode
photocathodeBottom = photocathodeTop + photocathodeThick
vacuumTop = photocathodeBottom
vacuumMiddleBottom = vacuumTop + vacuumMiddle
mcp1Top = vacuumMiddleBottom
mcp1Bottom = mcp1Top + mcpThick
mcpGapTop = mcp1Bottom
mcpGapBottom = mcpGapTop + mcp1to2gap
mcp2Top = mcpGapBottom
mcp2Bottom = mcp2Top + mcpThick
vacuumBottomTop = mcp2Bottom
vacuumBottomBottom = vacuumBottomTop + vacuumBottom
backplateTop = vacuumBottomBottom
backplateBottom = backplateTop + backplateThick

# Record start time
start_time = time.perf_counter()

# Create figure
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, width)
ax.set_ylim(0, backplateBottom)
ax.set_xlabel("μm")
ax.set_ylabel("μm")
ax.set_aspect('equal')
ax.invert_yaxis()  # Invert so top is at top
plt.title("LAPPD")

# Draw layers
ax.add_patch(patches.Rectangle((0, emptyTopStart), width, abovePhotocathode, color=topColor))
ax.add_patch(patches.Rectangle((0, photocathodeTop), width, photocathodeThick, color=photocathodeColor))
ax.add_patch(patches.Rectangle((0, vacuumTop), width, vacuumMiddle, color=vacuumColor, label="Vacuum"))
ax.add_patch(patches.Rectangle((0, mcp1Top), width, mcpThick, color=mcpBodyColor))
ax.add_patch(patches.Rectangle((0, mcpGapTop), width, mcp1to2gap, color=vacuumColor))
ax.add_patch(patches.Rectangle((0, mcp2Top), width, mcpThick, color=mcpBodyColor))
ax.add_patch(patches.Rectangle((0, vacuumBottomTop), width, vacuumBottom, color=vacuumColor))
ax.add_patch(patches.Rectangle((0, backplateTop), width, backplateThick, color=backplateColor))

# Label the important layers
ax.annotate(' Photocathode', xy=(5, photocathodeTop + photocathodeThick/2), 
            color='white', fontsize=8, ha='left', va='center')
ax.annotate(' MCP1', xy=(5, mcp1Top + mcpThick/2), 
            color='white', fontsize=8, ha='left', va='center')
ax.annotate(' MCP2', xy=(5, mcp2Top + mcpThick/2),
            color='white', fontsize=8, ha='left', va='center')
ax.annotate(' Anode', xy=(5, backplateTop + backplateThick/2), 
            color='white', fontsize=8, ha='left', va='center')

# Draw channels
mcp1_channel_angle_rad = np.radians(channelAngle)
mcp2_channel_angle_rad = -mcp1_channel_angle_rad
channel_spacing = width / (numChannels + 2)

for i in range(numChannels):
    cx_top1 = (i + labelSpacingFactor) * channel_spacing
    cx_bot1 = cx_top1 + mcpThick * np.tan(mcp1_channel_angle_rad)
    
    mcp1_pts = np.array([[cx_top1 - channelDiameter/2, mcp1Top], [cx_top1 + channelDiameter/2, mcp1Top],
                         [cx_bot1 + channelDiameter/2, mcp1Bottom], [cx_bot1 - channelDiameter/2, mcp1Bottom]])
    ax.add_patch(patches.Polygon(mcp1_pts, color=channelColor, edgecolor='black', linewidth=0.5))
    
    cx_top2 = cx_bot1
    cx_bot2 = cx_top2 + mcpThick * np.tan(mcp2_channel_angle_rad)
    
    mcp2_pts = np.array([[cx_top2 - channelDiameter/2, mcp2Top], [cx_top2 + channelDiameter/2, mcp2Top],
                         [cx_bot2 + channelDiameter/2, mcp2Bottom], [cx_bot2 - channelDiameter/2, mcp2Bottom]])
    ax.add_patch(patches.Polygon(mcp2_pts, color=channelColor, edgecolor='black', linewidth=0.5))

ax.add_patch(patches.Rectangle((0, 0), width, backplateBottom, linewidth=2, edgecolor='black', facecolor='none'))

# Initialize pre-allocated Matplotlib artists for blitting
photon_line, = ax.plot([], [], 'o', color=photonColor, markersize=photonSize, label='Photon')
electron_lines = [ax.plot([], [], 'o', color=photoelectronColor, markersize=electronSize)[0] for _ in range(MAX_ELECTRONS)]
electron_lines[0].set_label('Electrons')

# Track state
photon_state = {'active': True, 'x': np.random.uniform(width * 0.2, width * 0.8), 'y': 0}
active_electrons = [] # Format: [x, y, vx, vy]
collision_count = 0
total_generated = 0

def get_channel_bounds(x_pos, y_pos):
    """Finds the nearest channel bounds across both MCPs, and checks if strictly inside."""
    if mcp1Top <= y_pos <= mcp1Bottom:
        mcp_top, is_mcp1 = mcp1Top, True
    elif mcp2Top <= y_pos <= mcp2Bottom:
        mcp_top, is_mcp1 = mcp2Top, False
    else:
        return None, None, None, False

    progress = (y_pos - mcp_top) / mcpThick
    
    best_idx = None
    min_dist = float('inf')
    best_left, best_right = 0, 0
    in_hole = False
    
    for i in range(numChannels):
        cx_top1 = (i + labelSpacingFactor) * channel_spacing
        cx_bot1 = cx_top1 + mcpThick * np.tan(mcp1_channel_angle_rad)
        
        if is_mcp1:
            cx_center = cx_top1 + progress * (cx_bot1 - cx_top1)
        else:
            cx_top2, cx_bot2 = cx_bot1, cx_bot1 + mcpThick * np.tan(mcp2_channel_angle_rad)
            cx_center = cx_top2 + progress * (cx_bot2 - cx_top2)
            
        left_bound = cx_center - channelDiameter / 2
        right_bound = cx_center + channelDiameter / 2
        
        # Track the closest channel to prevent stepping into grey space
        dist = abs(x_pos - cx_center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            best_left = left_bound
            best_right = right_bound
            
        # Strict check for top/bottom surface collisions
        if left_bound <= x_pos <= right_bound:
            in_hole = True
            
    return best_idx, best_left, best_right, in_hole

def process_electron(x, y, vx, vy):
    """Universal physics engine for all electrons."""
    global collision_count, total_generated
    
    next_vy = vy + e_field_acceleration
    next_x, next_y = x + vx, y + next_vy
    
    bounced = False
    spawned = []
    wall_hit = None

    # 1. Outer Edges
    if next_x < 0:
        next_x, vx = 0, abs(vx)
    elif next_x > width:
        next_x, vx = width, -abs(vx)

    # 2. Surface Reflection Checks (Fixes teleportation)
    if y <= mcp1Top and next_y > mcp1Top:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp1Top + 1)
        if not in_hole:
            return next_x, mcp1Top - 1, vx, -abs(next_vy) * 0.8, False, []

    if y >= mcp1Bottom and next_y < mcp1Bottom:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp1Bottom - 1)
        if not in_hole:
            return next_x, mcp1Bottom + 1, vx, abs(next_vy) * 0.8, False, []

    if y <= mcp2Top and next_y > mcp2Top:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp2Top + 1)
        if not in_hole:
            return next_x, mcp2Top - 1, vx, -abs(next_vy) * 0.8, False, []

    if y >= mcp2Bottom and next_y < mcp2Bottom:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp2Bottom - 1)
        if not in_hole:
            return next_x, mcp2Bottom + 1, vx, abs(next_vy) * 0.8, False, []

    # 3. Inside Channels - Universal Clamping to prevent clipping
    if (mcp1Top < next_y < mcp1Bottom) or (mcp2Top < next_y < mcp2Bottom):
        idx, left, right, _ = get_channel_bounds(next_x, next_y)
        
        if idx is not None:
            # Channel wall collisions - Strictly clamp them into the bounds
            if next_x <= left + wallTolerance:
                next_x, vx = left + wallTolerance + 5, abs(vx) + 10  
                bounced = True
                wall_hit = 'left'
            elif next_x >= right - wallTolerance:
                next_x, vx = right - wallTolerance - 5, -(abs(vx) + 10)  
                bounced = True
                wall_hit = 'right'

    # 4. Universal Avalanche Generation with Directional Bias
    if bounced:
        collision_count += 1
        
        for _ in range(electronsPerCollision):
            # Bias secondary velocities away from the wall that was hit
            if wall_hit == 'left':
                sec_vx = np.random.uniform(5, 30)
            elif wall_hit == 'right':
                sec_vx = np.random.uniform(-30, -5)
            else:
                sec_vx = np.random.uniform(-30, 30)
            
            sec_vy = electronVelocity * 0.7
            spawned.append([next_x, next_y, sec_vx, sec_vy])

    return next_x, next_y, vx, next_vy, bounced, spawned

def update(frame):
    global total_generated

    # Update photon
    if photon_state['active']:
        photon_state['y'] += photonVelocity
        if photon_state['y'] >= photocathodeTop:
            photon_state['active'] = False
            photon_line.set_data([100000], [100000])
            # Spawn primary photoelectron
            active_electrons.append([photon_state['x'], photocathodeBottom, np.random.uniform(-40, 40), electronVelocity * 0.7])
            total_generated += 1
        else:
            photon_line.set_data([photon_state['x']], [photon_state['y']])

    # Update all electrons universally
    alive_electrons = []
    new_electrons = []

    for i in range(len(active_electrons)):
        x, y, vx, vy = active_electrons[i]
        next_x, next_y, next_vx, next_vy, bounced, spawned = process_electron(x, y, vx, vy)
        
        # Keep electron if it hasn't hit the backplate
        if next_y < backplateTop:
            alive_electrons.append([next_x, next_y, next_vx, next_vy])
            new_electrons.extend(spawned)

    # Add newly generated electrons (respecting the MAX_ELECTRONS cap)
    available_slots = MAX_ELECTRONS - len(alive_electrons)
    if available_slots > 0:
        allowed_new = new_electrons[:available_slots]
        alive_electrons.extend(allowed_new)
        total_generated += len(allowed_new)

    # Save state for next frame
    active_electrons[:] = alive_electrons

    # Map tracked electrons to Matplotlib artists
    for i in range(MAX_ELECTRONS):
        if i < len(active_electrons):
            electron_lines[i].set_data([active_electrons[i][0]], [active_electrons[i][1]])
        else:
            electron_lines[i].set_data([100000], [100000])

    return [photon_line] + electron_lines

# Create and save animation
ani = FuncAnimation(fig, update, frames=np.arange(numFrames), blit=True, interval=50)
ax.legend(loc='upper right', fontsize=8)

print("Generating Avalanche...")

ani.save(fileName, writer=PillowWriter(fps=fps))
print("Saving...")

print(f"Total Collisions: {collision_count}")
print(f"Total Electrons Tracked: {total_generated}")
print(f"Execution time: {(time.perf_counter() - start_time):.2f} seconds")
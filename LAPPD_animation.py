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

# Record start time
start_time = time.perf_counter()

# gif file info
filePath = '/Users/trey/Desktop/Razikave/Detector animations'
fileName = f"{filePath}HighContrast_LAPPD_animation.gif"

# Graphics params
numFrames = 900
fps = 60

# Particle coloring
photonColor = 'blue'
photoelectronColor = 'gold'

# Layer colors
topColor = "lavender"
photocathodeColor = 'darkgreen'
mcpBodyColor = 'grey'
channelColor = 'cornflowerblue'
backplateColor = 'brown'
vacuumColor = 'black'

# Layer thicknesses 
abovePhotocathode = 2600 #μm
photocathodeThick = 400 #μm
vacuumMiddle = 1190 #μm
mcpThick = 600 #μm
mcp1to2gap = 1370 #μm
vacuumBottom = 3000 #μm
backplateThick = 2000 #μm

# Plotframe parameters
height = abovePhotocathode + photocathodeThick + vacuumMiddle + mcpThick + mcp1to2gap + mcpThick + vacuumBottom + backplateThick
width = (height/3)*4

# Channel parameters
channelAngle = 20 # deg
numChannels = 32
channelDiameter = (width * 0.6)/numChannels
xOffset = channelDiameter/2 # properly centers the channels in the MCP
wallTolerance = 30
topAdj = 7 # adjust the top to truly align with the MCP
bottomAdj = 30 #same for the bottom

# Physics parameters
photonSize = 3
electronSize = 2
photonVelocity = 30  
electronVelocity = 15  
e_field_acceleration = 5  
bounceSidewaysV = 35
bounceHeightCoeff = 0.5
electronsPerCollision = 2
MAX_ELECTRONS = 1000
SUB_STEPS = 5

# Calculate layer positions FROM BOTTOM UP (backplate at y=0)
backplateBottom = 0
backplateTop = backplateBottom + backplateThick
vacuumBottomBottom = backplateTop
vacuumBottomTop = vacuumBottomBottom + vacuumBottom
mcp2Bottom = vacuumBottomTop
mcp2Top = mcp2Bottom + mcpThick
mcpGapBottom = mcp2Top
mcpGapTop = mcpGapBottom + mcp1to2gap
mcp1Bottom = mcpGapTop
mcp1Top = mcp1Bottom + mcpThick
vacuumMiddleBottom = mcp1Top
vacuumTop = vacuumMiddleBottom + vacuumMiddle
photocathodeBottom = vacuumTop
photocathodeTop = photocathodeBottom + photocathodeThick
emptyTopStart = photocathodeTop
emptyTopEnd = emptyTopStart + abovePhotocathode


# Create figure
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, width)
ax.set_ylim(0, emptyTopEnd)
ax.set_ylabel("mm")
ax.set_aspect('equal')
plt.title("LAPPD")

# Draw layers (from bottom up)
ax.add_patch(patches.Rectangle((0, emptyTopStart), width, abovePhotocathode, color=topColor, label="Outside"))
ax.add_patch(patches.Rectangle((0, photocathodeBottom), width, photocathodeThick, color=photocathodeColor))
ax.add_patch(patches.Rectangle((0, vacuumMiddleBottom), width, vacuumMiddle, color=vacuumColor, label="Vacuum"))
ax.add_patch(patches.Rectangle((0, mcp1Bottom), width, mcpThick, color=mcpBodyColor))
ax.add_patch(patches.Rectangle((0, mcpGapBottom), width, mcp1to2gap, color=vacuumColor))
ax.add_patch(patches.Rectangle((0, mcp2Bottom), width, mcpThick, color=mcpBodyColor))
ax.add_patch(patches.Rectangle((0, vacuumBottomBottom), width, vacuumBottom, color=vacuumColor))
ax.add_patch(patches.Rectangle((0, backplateBottom), width, backplateThick, color=backplateColor))

# Label the important layers
ax.annotate(' Outside', xy=(5, emptyTopStart + abovePhotocathode/2),
            color='black', fontsize=8, ha = 'left', va='center')
ax.annotate(' Photocathode', xy=(5, photocathodeBottom + photocathodeThick/2), 
            color='white', fontsize=8, ha='left', va='center')
ax.annotate(' MCP1', xy=(5, mcp1Bottom + mcpThick/2), 
            color='black', fontsize=8, ha='left', va='center')
ax.annotate(' MCP2', xy=(5, mcp2Bottom + mcpThick/2),
            color='black', fontsize=8, ha='left', va='center')
ax.annotate(' Anode', xy=(5, backplateBottom + backplateThick/2), 
            color='white', fontsize=8, ha='left', va='center')


# Draw channels
mcp1_channel_angle_rad = np.radians(channelAngle)
mcp2_channel_angle_rad = -mcp1_channel_angle_rad
channel_spacing = width / numChannels

for i in range(numChannels):
    # MCP1 channels (angled from bottom to top)
    cx_bot1 = i * channel_spacing
    cx_top1 = cx_bot1 + mcpThick * np.tan(mcp1_channel_angle_rad)
    
    mcp1_pts = np.array([[(cx_bot1 - channelDiameter/2)+xOffset, mcp1Bottom+bottomAdj], [(cx_bot1 + channelDiameter/2)+xOffset, mcp1Bottom+bottomAdj],
                         [(cx_top1 + channelDiameter/2)+xOffset, mcp1Top+topAdj], [(cx_top1 - channelDiameter/2)+xOffset, mcp1Top+topAdj]])
    ax.add_patch(patches.Polygon(mcp1_pts, color=channelColor, edgecolor='black', linewidth=0.5)) 
    
    # MCP2 channels (opposite angle)
    cx_bot2 = cx_top1
    cx_top2 = cx_bot2 + mcpThick * np.tan(mcp2_channel_angle_rad)
    
    mcp2_pts = np.array([[(cx_bot2 - channelDiameter/2)+xOffset, mcp2Bottom+bottomAdj], [(cx_bot2 + channelDiameter/2)+xOffset, mcp2Bottom+bottomAdj],
                         [(cx_top2 + channelDiameter/2)+xOffset, mcp2Top+topAdj], [(cx_top2 - channelDiameter/2)+xOffset, mcp2Top+topAdj]])
    ax.add_patch(patches.Polygon(mcp2_pts, color=channelColor, edgecolor='black', linewidth=0.5))

ax.add_patch(patches.Rectangle((0, 0), width, emptyTopEnd, linewidth=2, edgecolor='black', facecolor='none'))

# Initialize pre-allocated Matplotlib artists for blitting
photon_line, = ax.plot([], [], 'o', color=photonColor, markersize=photonSize, label='Photon')
electron_lines = [ax.plot([], [], 'o', color=photoelectronColor, markersize=electronSize)[0] for _ in range(MAX_ELECTRONS)]
electron_lines[0].set_label('Electrons')

# Track state - photon now starts at TOP and moves DOWN
photon_state = {'active': True, 'x': np.random.uniform(width * 0.2, width * 0.8), 'y': emptyTopEnd}
active_electrons = [] # Format: [x, y, vx, vy]
collision_count = 0
total_generated = 0



def get_channel_bounds(x_pos, y_pos):
    """Finds the nearest channel bounds across both MCPs, and checks if strictly inside."""
    if mcp1Bottom <= y_pos <= mcp1Top:
        mcp_bottom, is_mcp1 = mcp1Bottom, True
    elif mcp2Bottom <= y_pos <= mcp2Top:
        mcp_bottom, is_mcp1 = mcp2Bottom, False
    else:
        return None, None, None, False

    progress = (y_pos - mcp_bottom) / mcpThick
    
    best_idx = None
    min_dist = float('inf') #!!! 100000 
    best_left, best_right = 0, 0
    in_hole = False
    
    for i in range(numChannels):
        cx_bot1 = i * channel_spacing
        cx_top1 = cx_bot1 + mcpThick * np.tan(mcp1_channel_angle_rad)
        
        if is_mcp1:
            cx_center = xOffset + cx_bot1 + progress * (cx_top1 - cx_bot1)
        else:
            cx_bot2, cx_top2 = cx_top1, cx_top1 + mcpThick * np.tan(mcp2_channel_angle_rad)
            cx_center = xOffset + cx_bot2 + progress * (cx_top2 - cx_bot2)
            
        left_bound = cx_center - channelDiameter / 2
        right_bound = cx_center + channelDiameter / 2
        
        # Track the closest channel
        dist = abs(x_pos - cx_center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            best_left = left_bound
            best_right = right_bound
            
        # Strict check for holes
        if left_bound <= x_pos <= right_bound:
            in_hole = True
            
    return best_idx, best_left, best_right, in_hole



def process_electron_substep(x, y, vx, vy, dt=1.0):
    """
    Process one sub-step of electron movement with collision detection.
    Electrons now fall DOWN (negative vy).
    """
    global collision_count
    
    # Scale movement by timestep - acceleration is now NEGATIVE (pulling down)
    accel = -e_field_acceleration * dt
    next_vy = vy + accel
    next_x = x + vx * dt
    next_y = y + next_vy * dt
    
    bounced = False
    spawned = []
    wall_hit = None

    # 1. Outer Edges
    if next_x < 0:
        next_x, vx = 0, abs(vx)
    elif next_x > width:
        next_x, vx = width, -abs(vx)

    # 2. Surface Reflection Checks (flipped directions)
    # Hitting BOTTOM of MCP1 from above
    if y >= mcp1Top and next_y < mcp1Top:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp1Top - 1)
        if not in_hole:
            return next_x, mcp1Top + 1, vx, abs(next_vy) * bounceHeightCoeff, False, []

    # Hitting TOP of MCP1 from below
    if y <= mcp1Bottom and next_y > mcp1Bottom:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp1Bottom + 1)
        if not in_hole:
            return next_x, mcp1Bottom - 1, vx, -abs(next_vy) * bounceHeightCoeff, False, []

    # Hitting BOTTOM of MCP2 from above
    if y >= mcp2Top and next_y < mcp2Top:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp2Top - 1)
        if not in_hole:
            return next_x, mcp2Top + 1, vx, abs(next_vy) * bounceHeightCoeff, False, []

    # Hitting TOP of MCP2 from below
    if y <= mcp2Bottom and next_y > mcp2Bottom:
        _, _, _, in_hole = get_channel_bounds(next_x, mcp2Bottom + 1)
        if not in_hole:
            return next_x, mcp2Bottom - 1, vx, -abs(next_vy) * bounceHeightCoeff, False, []

    # 3. Inside Channels - Wall collision detection
    if (mcp1Bottom < next_y < mcp1Top) or (mcp2Bottom < next_y < mcp2Top):
        idx, left, right, in_hole = get_channel_bounds(next_x, next_y)
        
        if idx is not None and in_hole:
            # Inside a channel - check for wall collision
            if next_x <= left + wallTolerance:
                next_x = left + wallTolerance + 5
                vx = abs(vx) + bounceSidewaysV
                bounced = True
                wall_hit = 'left'
            elif next_x >= right - wallTolerance:
                next_x = right - wallTolerance - 5
                vx = -(abs(vx) + bounceSidewaysV)
                bounced = True
                wall_hit = 'right'
        elif idx is not None and not in_hole:
            # Outside channel but inside MCP - check if we tunneled through a wall
            prev_idx, prev_left, prev_right, prev_in_hole = get_channel_bounds(x, y)
            
            if prev_in_hole:
                # Was in a channel, now in gray - went through a wall!
                if next_x < prev_left:
                    # Went through left wall
                    next_x = prev_left + wallTolerance + 5
                    vx = abs(vx) + bounceSidewaysV
                    bounced = True
                    wall_hit = 'left'
                else:
                    # Went through right wall
                    next_x = prev_right - wallTolerance - 5
                    vx = -(abs(vx) + bounceSidewaysV)
                    bounced = True
                    wall_hit = 'right'

    # 4. Avalanche Generation
    if bounced:
        collision_count += 1
        
        for _ in range(electronsPerCollision):
            if wall_hit == 'left':
                sec_vx = np.random.uniform(5, bounceSidewaysV)
            elif wall_hit == 'right':
                sec_vx = np.random.uniform(-bounceSidewaysV, -5)
            else:
                sec_vx = np.random.uniform(-bounceSidewaysV, bounceSidewaysV)
            
            sec_vy = -electronVelocity * np.random.uniform(0, 1)  # Negative (falling down)
            spawned.append([next_x, next_y, sec_vx, sec_vy])

    return next_x, next_y, vx, next_vy, bounced, spawned



def process_electron(x, y, vx, vy):
    """Process one full frame with sub-stepping."""
    dt = 1.0 / SUB_STEPS
    all_spawned = []
    
    current_x, current_y = x, y
    current_vx, current_vy = vx, vy
    
    # Take multiple sub-steps per frame
    for step in range(SUB_STEPS):
        next_x, next_y, next_vx, next_vy, bounced, spawned = process_electron_substep(
            current_x, current_y, current_vx, current_vy, dt
        )
        
        current_x, current_y = next_x, next_y
        current_vx, current_vy = next_vx, next_vy
        all_spawned.extend(spawned)
        
        if bounced:
            pass
    
    return current_x, current_y, current_vx, current_vy, len(all_spawned) > 0, all_spawned

def update(frame):
    global total_generated

    # Update photon - now moving DOWN (negative velocity)
    if photon_state['active']:
        photon_state['y'] -= photonVelocity  # Moving down
        if photon_state['y'] <= photocathodeTop:
            photon_state['active'] = False
            photon_line.set_data([100000], [100000])
            
            # Spawn primary photoelectron with negative y-velocity (falling)
            active_electrons.append([photon_state['x'], photocathodeBottom, 
                                   np.random.uniform(-electronVelocity, electronVelocity), 
                                   -electronVelocity])  # Negative = falling
            total_generated += 1
        else:
            photon_line.set_data([photon_state['x']], [photon_state['y']])

    # Update all electrons with sub-stepping
    alive_electrons = []
    new_electrons = []

    for i in range(len(active_electrons)):
        x, y, vx, vy = active_electrons[i]
        next_x, next_y, next_vx, next_vy, bounced, spawned = process_electron(x, y, vx, vy)
        
        # Keep electron if it hasn't hit the backplate (now at bottom)
        if next_y > backplateTop:
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

# Hide the x axis, it's not accurate
ax.get_xaxis().set_visible(False)

# Set the y axis to mm instead of the μm defined in the code
yμm = ax.get_yticks()

#convert to mm, ignore the last point outside of the plot
ymm = yμm[:-1]/1000

# fix the photocathode inaccuracy (scale is a lie above the bottom of the photocathode, it's not really 400μm thick)
yμm[-2] = photocathodeBottom
ymm[-1] = photocathodeBottom/1000

#put them on the plot 
ax.set_yticks(yμm[:-1], ymm)




print("Generating Avalanche...")



ani.save(fileName, writer=PillowWriter(fps=fps))
print(f"Saving to {filePath}{fileName}...")

print(f"Total Collisions: {collision_count}")
print(f"Total Electrons Tracked: {total_generated}")
print(f"Execution time: {(time.perf_counter() - start_time):.2f} seconds")

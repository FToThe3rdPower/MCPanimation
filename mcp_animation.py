#!/usr/bin/python3
# -*- coding: utf-8 -*-
# MCP (Microchannel Plate) Photon Detection Animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import time

# Save parameters
filePath = './'
fileName = f"{filePath}MCP_animation.gif"

# Graphics params
numFrames = 300
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

# Geometry parameters
width = 150  # μm

# Layer thicknesses
vacuumTop = 37  # μm, space above photocathode
photocathodeThick = 4  # μm
vacuumMiddle = 40  # μm, vacuum between photocathode and MCP
mcpThick = 80  # μm, MCP body thickness
vacuumBottom = 20  # μm, space below MCP, above anode
backplateThick = 5  # μm

# Calculate total height
height = vacuumTop + photocathodeThick + vacuumMiddle + mcpThick + vacuumBottom + backplateThick

# Channel parameters
channelAngle = 20  # degrees from vertical
channelWidth = 8  # μm
channelDiameter = 6  # μm (for visualization)
numChannels = 3  # number of channels to show

# Particle parameters
photonSize = 8
electronSize = 4
photonVelocity = 2.5  # μm per frame
electronVelocity = 2.0  # μm per frame
electronsPerCollision = 3  # multiplication factor

# Calculate layer positions (from top)
emptyTopStart = 0
photocathodeTop = vacuumTop
photocathodeBottom = photocathodeTop + photocathodeThick
vacuumMiddleTop = photocathodeBottom
vacuumMiddleBottom = vacuumMiddleTop + vacuumMiddle
mcpTop = vacuumMiddleBottom
mcpBottom = mcpTop + mcpThick
vacuumBottomTop = mcpBottom
vacuumBottomBottom = vacuumBottomTop + vacuumBottom
backplateTop = vacuumBottomBottom
backplateBottom = backplateTop + backplateThick

# Record start time
start_time = time.perf_counter()

# Create figure
fig, ax = plt.subplots()
fig.set_size_inches(6, 8)

# Set axis limits
ax.set_xlim(0, width)
ax.set_ylim(0, backplateBottom)
ax.set_xlabel("μm")
ax.set_ylabel("μm")
ax.set_aspect('equal')
ax.invert_yaxis()  # Invert so top is at top

plt.title("Microchannel Plate Detector")

# Draw layers
ax.add_patch(patches.Rectangle((0, emptyTopStart), width, vacuumTop, color=topColor))
ax.add_patch(patches.Rectangle((0, photocathodeTop), width, photocathodeThick, color=photocathodeColor, label='Photocathode'))
ax.add_patch(patches.Rectangle((0, vacuumMiddleTop), width, vacuumMiddle, color=vacuumColor))
ax.add_patch(patches.Rectangle((0, mcpTop), width, mcpThick, color=mcpBodyColor, label='MCP Body'))
ax.add_patch(patches.Rectangle((0, vacuumBottomTop), width, vacuumBottom, color=vacuumColor, label='Vacuum'))
ax.add_patch(patches.Rectangle((0, backplateTop), width, backplateThick, color=backplateColor, label='Backplate/Anode'))

# Draw angled channels in the MCP
channel_angle_rad = np.radians(channelAngle)
channel_spacing = width / (numChannels + 2)

for i in range(numChannels):
    # Calculate channel position
    channel_x_top = (i + 1) * channel_spacing
    channel_x_bottom = channel_x_top + mcpThick * np.tan(channel_angle_rad)
    
    # Draw channel as a parallelogram
    channel_points = np.array([
        [channel_x_top - channelDiameter/2, mcpTop],
        [channel_x_top + channelDiameter/2, mcpTop],
        [channel_x_bottom + channelDiameter/2, mcpBottom],
        [channel_x_bottom - channelDiameter/2, mcpBottom]
    ])
    
    channel = patches.Polygon(channel_points, color=channelColor, edgecolor='black', linewidth=0.5)
    ax.add_patch(channel)
    
    #Grab the x position of the top of the second channel so the photon always goes into the channel
    if i == 1:
        photonSpwnX = channel_x_top - channelDiameter/2

# Add border
border = patches.Rectangle((0, 0), width, backplateBottom, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(border)

# Add labels
ax.annotate('Photocathode', xy=(5, photocathodeTop + photocathodeThick/2), 
            color='white', fontsize=8, ha='left', va='center')
ax.annotate('MCP', xy=(5, mcpTop + mcpThick/2), 
            color='white', fontsize=8, ha='left', va='center')
ax.annotate('Anode', xy=(5, backplateTop + backplateThick/2), 
            color='white', fontsize=8, ha='left', va='center')

# Initialize photon
photon, = ax.plot([], [], 'o', color=photonColor, markersize=photonSize, label='Photon')
photon_active = True
photon_x = photonSpwnX
photon_y = 0


# Initialize photoelectron (spawns after photon hits photocathode)
photoelectron, = ax.plot([], [], 'o', color=photoelectronColor, markersize=electronSize, label='Photoelectron')
photoelectron_active = False
photoelectron_x = 0
photoelectron_y = 0
photoelectron_vx = 0  # x-velocity
photoelectron_vy = electronVelocity  # y-velocity

# Choose middle channel for the photoelectron
chosen_channel = 1  # middle channel
channel_x_top = (chosen_channel + 1) * channel_spacing
channel_x_bottom = channel_x_top + mcpThick * np.tan(channel_angle_rad)

# Storage for secondary electrons: [x, y, vx, vy]
secondary_electrons = []
sec_electron_dots = []

# Track collision points in the channel
collision_count = 0

def get_channel_boundaries(y_pos):
    """Calculate left and right boundaries of the channel at a given y position"""
    if y_pos < mcpTop or y_pos > mcpBottom:
        return None, None
    
    # Linear interpolation along the channel
    progress = (y_pos - mcpTop) / mcpThick
    channel_x_center = channel_x_top + progress * (channel_x_bottom - channel_x_top)
    
    left_bound = channel_x_center - channelDiameter / 2
    right_bound = channel_x_center + channelDiameter / 2
    
    return left_bound, right_bound

def update(frame):
    global photon_active, photoelectron_active
    global photon_x, photon_y, photoelectron_x, photoelectron_y
    global photoelectron_vx, photoelectron_vy
    global collision_count
    
    # Update photon
    if photon_active:
        photon_y += photonVelocity
        
        # Check if photon hits photocathode
        if photon_y >= photocathodeTop:
            photon_active = False
            photon.set_data([1000], [1000])  # Hide photon
            
            # Spawn photoelectron with initial velocity
            photoelectron_active = True
            photoelectron_x = photon_x
            photoelectron_y = photocathodeBottom
            photoelectron_vx = np.random.uniform(-0.5, 0.5)  # Small random initial x velocity
            photoelectron_vy = electronVelocity
        else:
            photon.set_data([photon_x], [photon_y])
    
    # Update photoelectron
    if photoelectron_active:
        # Move with current velocity
        photoelectron_x += photoelectron_vx
        photoelectron_y += photoelectron_vy
        
        # Check boundaries and bounce if in MCP
        if mcpTop <= photoelectron_y <= mcpBottom:
            left_bound, right_bound = get_channel_boundaries(photoelectron_y)
            
            if left_bound and right_bound:
                # Check for collision with left wall
                if photoelectron_x <= left_bound:
                    photoelectron_x = left_bound  # Place at boundary
                    photoelectron_vx = abs(photoelectron_vx)  # Reverse to positive (bounce right)
                    
                    # Spawn secondary electrons
                    for _ in range(electronsPerCollision):
                        sec_vx = np.random.uniform(-1.5, 1.5)
                        sec_vy = electronVelocity + np.random.uniform(-0.5, 0.5)
                        secondary_electrons.append([photoelectron_x, photoelectron_y, sec_vx, sec_vy])
                        
                        # Create dot for this electron
                        dot, = ax.plot([], [], 'o', color=photoelectronColor, markersize=electronSize)
                        sec_electron_dots.append(dot)
                    
                    collision_count += 1
                
                # Check for collision with right wall
                elif photoelectron_x >= right_bound:
                    photoelectron_x = right_bound  # Place at boundary
                    photoelectron_vx = -abs(photoelectron_vx)  # Reverse to negative (bounce left)
                    
                    # Spawn secondary electrons
                    for _ in range(electronsPerCollision):
                        sec_vx = np.random.uniform(-1.5, 1.5)
                        sec_vy = electronVelocity + np.random.uniform(-0.5, 0.5)
                        secondary_electrons.append([photoelectron_x, photoelectron_y, sec_vx, sec_vy])
                        
                        # Create dot for this electron
                        dot, = ax.plot([], [], 'o', color=photoelectronColor, markersize=electronSize)
                        sec_electron_dots.append(dot)
                    
                    collision_count += 1
        
        # Check if hit backplate
        if photoelectron_y >= backplateTop:
            photoelectron_active = False
            photoelectron.set_data([1000], [1000])
        else:
            photoelectron.set_data([photoelectron_x], [photoelectron_y])
    
    # Update secondary electrons
    for i in range(len(secondary_electrons)):
        sec_x, sec_y, sec_vx, sec_vy = secondary_electrons[i]
        
        # Move with velocity
        sec_x += sec_vx
        sec_y += sec_vy
        
        # Check boundaries and bounce if in MCP
        if mcpTop <= sec_y <= mcpBottom:
            left_bound, right_bound = get_channel_boundaries(sec_y)
            
            if left_bound and right_bound:
                # Bounce off left wall
                if sec_x <= left_bound:
                    sec_x = left_bound
                    sec_vx = abs(sec_vx)  # Reverse to positive
                
                # Bounce off right wall
                elif sec_x >= right_bound:
                    sec_x = right_bound
                    sec_vx = -abs(sec_vx)  # Reverse to negative
        
        # Update stored values
        secondary_electrons[i] = [sec_x, sec_y, sec_vx, sec_vy]
        
        # Check if hit backplate
        if sec_y >= backplateTop:
            sec_electron_dots[i].set_data([1000], [1000])  # Hide
        else:
            sec_electron_dots[i].set_data([sec_x], [sec_y])
    
    return [photon, photoelectron] + sec_electron_dots

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(numFrames), blit=True, interval=50)

# Add legend
ax.legend(loc='upper right', fontsize=8)

# Save animation
print(f"Saving animation to {fileName}...")
ani.save(fileName, writer=PillowWriter(fps=fps))

# Calculate execution time
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Animation saved successfully!")
print(f"Execution time: {elapsed_time:.2f} seconds")
print(f"Total collisions: {collision_count}")
print(f"Total secondary electrons generated: {len(secondary_electrons)}")

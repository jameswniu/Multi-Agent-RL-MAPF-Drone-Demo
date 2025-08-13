# drone_superai_demo
**Instant-alignment drone AI for safe, adaptive, real-time flight control**

## Overview  
Instead of waiting for delayed success signals, this system scores every action instantly against a set of internal principles (the “Constitution”) so it can act safely and efficiently in real time.

**Drone Reinforcement Learning (RL) and Multi-Agent Path Finding (MAPF) Alignment System – Complete Overview**  
This drone artificial intelligence system runs on three core agents that sense, process, and decide actions in real time using internal values rather than delayed rewards. A Safety Controller blocks risky moves, and a Supervisor keeps every component running smoothly and safely.

## How the System Works  
The program simulates a distributed control system for a drone, built around three independent agents:

- **Ingestion Agent**  
  Reads raw sensor data such as camera feeds, global positioning system (GPS) readings, inertial measurement unit (IMU) readings, and barometer values at a fixed frequency. Pushes the data into bounded queues for processing.

- **Preprocess Agent**  
  Cleans, filters, and compresses incoming data into a compact set of features that are easier for the decision-making process to use.

- **Prediction Agent**  
  Uses the processed features to select the drone’s next action (hover, climb, turn left/right, move forward).

## Alignment Scoring  
Instead of delayed rewards, each action is scored immediately based on internal values such as:  
- Maintain safety margins  
- Save energy  
- Move smoothly  

Each value returns a score between 0 and 1, weighted by importance. Scores are combined and tested with small variations (structural inversion test) to ensure stability. The **Alignment Policy** chooses the action with the highest current alignment score to prevent unsafe or short-sighted decisions.

## Safety and Oversight  
- **Safety Controller**  
  Runs outside the main AI path and can override any decision if it detects danger (for example, entering a no-fly zone).  
- **Supervisor**  
  Starts all agents and the Safety Controller, monitors performance, restarts failed agents, and quarantines components that fail repeatedly.

## Operational Characteristics  
- Runs continuously and adapts to changing environmental conditions.  
- Uses continuous alignment scoring rather than delayed reinforcement learning.  
- Combines real-time decision making with hard safety overrides.

## System Design Files  
- `drone_high_lv_system_design.png` – High-level diagram showing drone agents, Safety Controller, Supervisor, server components, and data/control flow.  
- `drone_low_lv_system_design.png` – Low-level diagram showing process flows, interfaces, and message passing.  

## Core Code  
- `drone_main.py` – Main program entry point. Initializes the Supervisor, starts agents, manages the Safety Controller, and handles alignment scoring.

## Reference Material  
- `drones_matrix_RL.png` – Diagram comparing vertical vs. horizontal reward patterns in drone RL, showing trade-offs between event-based rewards and continuous alignment scoring.

## Design Document  
- `low_level_design` – File containing detailed subsystem designs, flowcharts, and specifications for each system module.

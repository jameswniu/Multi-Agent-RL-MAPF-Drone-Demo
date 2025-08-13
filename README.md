drone_superai_demo
Real time drone artificial intelligence with instant alignment scoring for safer, smarter flight

Instead of waiting for delayed success signals, this system scores every action instantly against a set of internal principles, referred to as the Constitution, so it can act safely and efficiently in real time.

Drone Reinforcement Learning (RL) and Multi-Agent Path Finding (MAPF) Alignment System – Complete Overview
This drone artificial intelligence system runs on three core agents that sense, process, and decide actions in real time using internal values rather than delayed rewards. A Safety Controller blocks risky moves, and a Supervisor keeps every component running smoothly and safely.

How the System Works
The program simulates a distributed control system for a drone, built around three independent agents:

Ingestion Agent
Reads raw sensor data such as camera feeds, global positioning system (GPS) readings, inertial measurement unit (IMU) readings, and barometer values at a fixed frequency
Pushes the data into bounded queues to prevent overload and ensure consistent latency

Preprocess Agent
Cleans and processes sensor inputs into a smaller, more useful set of features
Removes noise, normalizes values, and prepares structured input for decision making

Prediction Agent
Chooses the next action, such as hover, climb, turn left, turn right, or move forward, using the processed features
Actions are scored instantly using an alignment as reward approach, where each move is judged against a set of internal principles called the Constitution

Alignment as Reward Scoring
Example principles:

Maintain safety margin

Save energy

Move smoothly

Each principle returns a score between zero and one, weighted by importance

Scores are combined and validated through a structural inversion test, which is a small, temporary change in action choice that is simulated to confirm the decision remains stable

The Alignment Policy picks the action with the highest current score, avoiding unsafe or short sighted moves

Safety and Oversight
Safety Controller
Independent from the AI decision path
Has full authority to block or override any action if it detects immediate danger, such as approaching a no fly zone, collision risk, or unstable control inputs

Supervisor
Root process that starts and monitors all agents and the Safety Controller
Detects and responds to:

Errors or crashes

Performance degradation or slowdowns

Repeated failures, which trigger quarantine of that agent
Can restart or isolate components without halting the entire system

Black Box Logging
Records all sensor inputs, intermediate features, decisions, overrides, and system states for post flight analysis and debugging

Continuous Learning and Stability
Operates continuously, adapting decisions to the current environment without pausing for retraining cycles

Uses alignment scoring for every action rather than relying solely on delayed rewards from mission completion

Telemetry data is sent to the server for:

Analytics – Trends in performance, environment challenges, and operational safety margins

Weight Learner – Adjusts decision making policies over time

Configuration Service – Pushes updated settings back to the drone

High Level Design
Drone side:

Root Supervisor

Ingestion Agent

Preprocess Agent

Prediction Agent

Safety Controller

Bounded queues for data flow control

Black box logging for full traceability

Server side:

Telemetry ingest and storage

Analytics engine

Weight learner for policy updates

Configuration service for remote adjustments

Monitoring system for health checks and alerts

Links:

Solid arrows represent the data plane, which uses gRPC for real time sensor and decision data

Dashed arrows represent the control plane and telemetry, which use Representational State Transfer (REST) and gRPC for configuration, logging, and supervision

Repository Structure
System Design Files

drone_high_lv_system_design.png – High level diagram of the drone AI system showing agents, Safety Controller, Supervisor, server components, and data/control flow

drone_low_lv_system_design.png – Low level design showing process flows, interfaces, and message passing between components

Core Code

drone_main.py – Main entry point. Initializes the Supervisor, starts agents, manages the Safety Controller, and handles continuous alignment scoring

Reference Material

drones_matrix_RL.png – Concept diagram of vertical versus horizontal reward patterns in drone RL, showing stability trade offs between event based and continuous meaning rewards

Design Document

low_level_design – Detailed design document containing subsystem descriptions, flowcharts, and specifications for each system module

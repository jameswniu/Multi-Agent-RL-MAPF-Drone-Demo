# drone_superai_demo
This drone AI system runs on three core agents that sense, process, and decide actions in real time using internal values (not delayed rewards), with a SafetyController to block risky moves and a Supervisor to keep everything running smoothly and safely.


Explanation of Drone RL & MPFA Alignment System
This Python program simulates a distributed control system for a drone that uses three independent agents: IngestionAgent (reads sensor data), PreprocessAgent (extracts features), and PredictionAgent (decides actions). Instead of learning from delayed rewards, it uses an 'alignment-as-reward' approach, where each decision is scored instantly against internal values (the 'Constitution'). A SafetyController supervises the final output to enforce safety rules. The Supervisor starts, monitors, and restarts agents if needed, ensuring continuous, reliable operation.


Detailed Explanation
1. The system has three main agents working in sequence:
   - IngestionAgent: Reads sensor data (like drone cameras, GPS, etc.) at a fixed frequency and sends it to the next step.
   - PreprocessAgent: Cleans and processes the data into a smaller set of features that are easier for the decision-making system to use.
   - PredictionAgent: Uses the processed features to choose the best action for the drone (e.g., hover, climb, turn left/right, move forward).

2. Instead of waiting for a future 'reward' (like reaching a destination) to train the system, the program scores each action immediately using a set of internal principles called the 'Constitution'.
   - Example principles: Maintain safety margin, save energy, move smoothly.
   - Each principle gives a score from 0 to 1, weighted by importance.
   - The scores are combined and slightly tested with small changes (structural inversion test) to check stability.

3. The AlignmentPolicy picks the action with the highest current alignment score — this avoids unsafe or short-sighted actions.

4. The SafetyController is outside the AI path — it has the power to override any decision if it detects danger (like approaching a no-fly zone). This ensures safety always comes first.

5. The Supervisor manages everything:
   - Starts all agents and the safety controller.
   - Monitors them for errors, slowdowns, or crashes.
   - Restarts agents if they fail, but will quarantine them if they keep failing too often.

6. The program runs continuously, simulating how a real drone could operate safely while constantly adapting its actions based on present conditions — this is 'continuous learning' with alignment values, not just reward events.


High-level design
- Drone side: Root Supervisor, Ingestion Agent, Preprocess Agent, Prediction Agent, Safety Controller, bounded queues, black box logging.
- Server side: Telemetry ingest/store, analytics, weight learner, config service, monitoring.
- Links: Solid arrows for data plane (gRPC), dashed arrows for control plane/telemetry (REST + gRPC)

# RSU Digital Twin - Implementation Sprint

> **A comprehensive one-week implementation of RSU (Road Side Unit) digital twin for C-V2X infrastructure**

## Quick Start

```bash
# 1. Setup environment and dependencies
./quick_setup.sh

# 2. Start CARLA simulator
./start-carla.sh

# 3. Run Phase 1 demo (recommended for beginners)

conda activate rsu-fusion
./run_demo.sh phase1_batch

# 4. View results in data/outputs/
```
## Implementation Roadmap

### **Prerequisites & Setup**

  * **Environment:** Python 3.10+ with conda/venv
  * **Core Libraries:** PyTorch, OpenCV, NumPy, filterpy, ultralytics
  * **CARLA Simulator:** CARLA 0.9.15 with Docker/GPU support
  * **IDE:** VS Code recommended with Python extensions

**🚀 Run `./quick_setup.sh` to install everything automatically**

---

### **Day 1–2: Phase 1 - Single-Camera 2D Tracking (SORT)**

**Goal:** Build a functional single-camera tracker that can detect and assign persistent IDs to objects in a 2D video stream.

| Day | Morning Session (4 hours) | Afternoon Session (4 hours) | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Day 1** | **Setup & Object Detection:** <br> 1. Clone a robust YOLOv8 repository.[1] <br> 2. Download pre-trained YOLOv8 weights (e.g., `yolov8m.pt`). <br> 3. Write a script to read a single camera feed from CARLA. <br> 4. Integrate YOLOv8 to run inference on each frame and draw 2D bounding boxes on the output video.[2] | **Kalman Filter Basics:** <br> 1. Study the Kalman Filter's role in tracking: predicting an object's next position based on its current state.[3] <br> 2. Find a Python implementation (e.g., `filterpy` library or a simple NumPy-based class from an open-source project).[4] <br> 3. Implement the state vector `[x, y, area, aspect_ratio, vx, vy]` and the predict step. | A video stream from one CARLA camera with YOLOv8 detections drawn on it. A standalone, testable Kalman Filter class. |
| **Day 2** | **Data Association (Hungarian Algorithm):** <br> 1. For each frame, predict the new locations of existing tracks using your Kalman Filter. <br> 2. Calculate an Intersection over Union (IoU) cost matrix between the predicted boxes and the new YOLOv8 detections. <br> 3. Use `scipy.optimize.linear_sum_assignment` to solve the assignment problem, matching detections to tracks.[5] | **Putting It All Together (SORT):** <br> 1. Write the main tracking loop: Predict -> Associate -> Update. <br> 2. For matched pairs, update the Kalman Filter with the new detection. <br> 3. Implement track management: Create new tracks for unmatched detections and delete tracks that are lost for a set number of frames. <br> 4. Visualize the output with unique IDs and colored bounding boxes for each track. | A functional SORT tracker running on a single camera feed. You should see objects being assigned IDs that persist for as long as they are clearly visible. |

-----

### **Day 3–4: Phase 2 - Single-Camera 2D Hybrid Tracking (DeepSORT)**

**Goal:** Enhance the tracker's robustness to occlusion by adding an appearance-based Re-Identification (Re-ID) model.

| Day | Morning Session (4 hours) | Afternoon Session (4 hours) | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Day 3** | **Integrate Re-ID Model:** <br> 1. Find a lightweight, pre-trained Re-ID model. OSNet is an excellent choice.[6] Many open-source projects combine YOLO with StrongSORT/DeepSORT and OSNet. <br> 2. Write a feature extractor function that takes an image patch (the bounding box area) and returns a feature vector (embedding). <br> 3. Modify your Day 2 code to extract an appearance embedding for every new detection from YOLOv8. | **Hybrid Cost Matrix:** <br> 1. Study the two components of the DeepSORT cost matrix: Mahalanobis distance (motion) and Cosine distance (appearance).[7, 8] <br> 2. Implement the Mahalanobis distance calculation using the output of the Kalman Filter. <br> 3. Implement the Cosine distance calculation between a new detection's embedding and a gallery of embeddings stored for each track. | A script that now generates both a bounding box and a feature vector for each detected object. The core functions for calculating both distance metrics are complete. |
| **Day 4** | **Implement DeepSORT Logic:** <br> 1. Replace the IoU-based association from Phase 1 with a new function that computes the weighted, combined cost from the Mahalanobis and Cosine distances. <br> 2. Implement the matching cascade logic from DeepSORT, which prioritizes recently seen tracks.[7] <br> 3. Update the track management logic to store a gallery of the last N appearance features for each confirmed track. | **Testing & Refinement:** <br> 1. Test the new hybrid tracker on your CARLA stream, specifically focusing on scenarios with temporary occlusions. <br> 2. Observe how the tracker now successfully re-identifies an object after it has been hidden for a few seconds, which the SORT tracker would have failed to do. <br> 3. Tune the weighting parameter between the motion and appearance costs. | A robust single-camera hybrid tracker. You should see a significant reduction in ID switches compared to the Phase 1 implementation, especially when objects occlude each other. |

-----

### **Day 5–6: Phase 3 & 4 (Part 1) - Multi-Camera Fusion in BEV**

**Goal:** Project detections from all cameras into a unified Bird's-Eye View (BEV) and adapt the tracker to operate in this 3D space.

| Day | Morning Session (4 hours) | Afternoon Session (4 hours) | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Day 5** | **Multi-Camera Input & Calibration:** <br> 1. Modify your input script to read frames from all simulated cameras simultaneously. Ensure they are synchronized. <br> 2. Obtain the intrinsic and extrinsic camera calibration parameters for each camera from CARLA. Store these in a configuration file. <br> 3. Study the geometry of inverse perspective mapping to understand how to project a 2D pixel to a 3D ground plane point.[9] | **BEV Projection Implementation:** <br> 1. Run your Phase 2 tracker on each camera stream independently to get 2D detections with features. <br> 2. Write a projection function that takes a 2D bounding box and camera parameters, and outputs a 3D `(x, y)` coordinate on the BEV map. <br> 3. Create a unified list of detections at each timestamp, where each detection consists of `<x, y, embedding, class>`. <br> 4. Visualize the BEV detections as points on a 2D plot to verify correctness. | A system that processes multiple camera feeds in parallel. A verified BEV projection that shows all detected objects from all cameras on a single top-down map. |
| **Day 6** | **Adapt Tracker to BEV:** <br> 1. Modify the Kalman Filter's state vector to represent motion in 3D world coordinates (e.g., `[x, y, w, l, vx, vy]`) instead of 2D pixel coordinates. <br> 2. Update the motion model and the Mahalanobis distance calculation to work with these new 3D states. The appearance (Cosine) distance logic remains unchanged. | **Global Tracking & Management:** <br> 1. Feed the unified list of BEV detections into your adapted hybrid tracker. <br> 2. The tracker will now manage a single, global list of tracks for the entire intersection, fusing information from all cameras in real-time. <br> 3. Test and debug the full multi-camera pipeline. Pay close attention to the association of objects that are visible in the overlapping regions of multiple cameras. | A complete, multi-camera, multi-object hybrid fusion tracker. You can visualize the final tracked objects with their global IDs on the BEV map. |

-----

### **Day 7: Phase 4 (Part 2) - Finalizing Output for V2X**

**Goal:** Format the tracker's output into a structured track file and map it to the SAE J2735 CPM standard.

| Day | Morning Session (4 hours) | Afternoon Session (4 hours) | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Day 7** | **Global Track File Generation:** <br> 1. At each timestamp, format the output of your BEV tracker into a structured list (JSON is a good choice for readability).[10, 11] <br> 2. Each object in the list should have a persistent `track_id`, `class`, `position`, `velocity`, and `size`. <br> 3. Write this structured data to a log file. | **Mapping to SAE J2735 CPM:** <br> 1. Study the structure of the SAE J2735 Collective Perception Message (CPM), focusing on the `PerceivedObjectContainer`.[12, 13] <br> 2. Create a mapping function that takes an object from your global track file and populates the corresponding CPM fields (e.g., `track_id` -> `objectID`, `position` -> `xPos`, `yPos`). <br> 3. Pay close attention to unit conversions (e.g., meters/sec to cm/sec) as required by the standard.[14] <br> 4. (Stretch Goal) Find a Python ASN.1 library to perform the final UPER encoding of the CPM structure.[15] | A clean, structured global track file is generated in real-time. A Python script that can convert this track file into a data structure that mirrors the required CPM format, ready for encoding and transmission. |
**Project README – Data Flow & System Architecture
Overview**

This application transforms uploaded videos into structured, queryable, and visualizable 4D scene intelligence.
By combining Gaussian Splatting (4D reconstruction), Gemini object reasoning, Supabase vector storage, and LangChain-powered RAG, the system enables:
- Real-world object localization
- Semantic search over video scenes
- Safety risk detection
- 3D scene visualization
- Path and object mapping

**End-to-End Data Flow**
**Video Upload**

The pipeline begins when a user uploads a video.
The video is immediately sent to two parallel processing systems:
- Gaussian Splatting Engine
- Gemini Object Analysis

**4D Scene Extraction (Gaussian Splatting)**

Our Gaussian Splat (GSplat) technology reconstructs the scene and extracts 4D data (3D space + time):
Extracted Data:
- Camera positions over time
- Camera orientation
- Full 3D scene structure
- Temporal scene evolution

This provides a precise spatial understanding of the environment.
Additionally:
- GSplat directly powers our 3D visualization system
- Allows us to see the model’s understanding of the full scene

**Object Approximation (Gemini)**

Gemini processes the same video independently to extract:
Extracted Data:
- Object approximations
- Object bounding regions
- Object position relative to the camera
- Object semantic meaning

This provides semantic understanding layered on top of the reconstructed geometry.

**Real-World Object Position Calculation**

We combine:
- Camera pose data from GSplat
- Object-relative-to-camera data from Gemini

Result:
We compute the true global position of each object in the scene.
- Absolute object coordinates
- Time-based object trajectories
- Fully mapped spatial relationships
- Supabase Storage Layer
- All processed data is indexed into Supabase.

Stored Data Categories
1. Object Embeddings - Each object is embedded using Gemini Stored as vectors Enables semantic search
Example:
"Where is the forklift near the exit?" -> Embedding similarity lookup returns relevant objects

2. Video Metadata - Video identifiers, Timestamps, Processing metadata

3. Object & Camera Pose Data - Global object positions, Camera path data, Time-series pose tracking
This data is later used for: Rendering, Path reconstruction, Spatial querying


**LangChain Integration**
LangChain acts as the intelligence orchestration layer.

Capabilities
1. Querying Supabase - Models use LangChain to: Retrieve object embeddings Perform semantic search Access spatial metadata

2. RAG-Based Safety Risk Detection - LangChain enables Retrieval-Augmented Generation (RAG): Connects to safety knowledge bases Retrieves leading safety risk data Cross-checks detected objects Highlights risks found in the model output
Example: Forklift + pedestrian proximity, Missing safety equipment, Restricted zone violations

**Visualization Systems**1️
1. 3D Scene Visualization (Direct from GSplat) - Built directly from Gaussian Splat output Provides full spatial scene reconstruction Shows how the model interprets geometry

2. Path & Object Maps (From Supabase) - Using stored pose + object data: Camera path reconstruction, Object position overlays, Movement trajectory visualization, Scene interaction mapping

**Architecture Summary**
                ┌──────────────┐
                │ Video Upload │
                └──────┬───────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
┌──────────────────┐        ┌─────────────────┐
│ Gaussian Splat   │        │ Gemini Analysis │
│(4D Reconstruction)│      │(Object Detection)│
└─────────┬────────┘        └────────┬────────┘
          │                          │
          └──────────┬───────────────┘
                     │
        ┌──────────────────────────────┐
        │ Global Object Position Solver │
        └──────────────┬───────────────┘
                       │
               ┌───────────────┐
               │   Supabase    │
               │ (Vectors +    │
               │  Spatial Data)│
               └───────┬───────┘
                       │
        ┌──────────────┴─────────────┐
        │                            │
┌───────────────┐            ┌────────────────┐
│ LangChain RAG │            │ Visualizations │
└───────────────┘            └────────────────┘

**Core Capabilities**

✅ 4D scene reconstruction
✅ Global object localization
✅ Semantic object search
✅ Safety risk detection (RAG-powered)
✅ Camera path reconstruction
✅ Interactive 3D scene visualization

**What This Enables **
Industrial safety monitoring
Scene intelligence extraction
Spatial querying with natural language
Automated hazard detection
Advanced post-video spatial analytics

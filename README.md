# 6D Pose Estimation for Industrial Screw Detection

A vision pipeline for detecting screws and estimating their 6D pose 
to guide robotic manipulation.

## Pipeline

1. **Data Collection & Labeling**: 3 raw images labeled in Roboflow
2. **Augmentation**: Each image augmented ×100 → 300 training images
3. **Training**: Detection model trained on the augmented dataset
4. **Validation**: 3 original (non-augmented) images used for validation
5. **Testing**: 2 held-out images (outside the original set) for testing
6. **Pose Estimation**:
   - Plane normalization
   - Depth-based 3D localization
   - 6D pose computation using camera intrinsics

## Project Structure

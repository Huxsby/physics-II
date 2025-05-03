import numpy as np

θ = [0, 0, 10, 10, 0, 0, 10]  # Example joint angles

def limits(θ):
    """
    Function to limit the angles of the Niryo One robot.
    """
    # Define the limits for each joint
    limits = {
        'joint_1': (-3.14, 3.14),
        'joint_2': (-1.57, 1.57),
        'joint_3': (-1.57, 1.57),
        'joint_4': (-3.14, 3.14),
        'joint_5': (-1.57, 1.57),
        'joint_6': (-3.14, 3.14),
        'joint_7': (-1.57, 1.57)
    }
    
    # Check if the angles are within the limits
    for i in range(len(θ)):
        if θ[i] < limits[f'joint_{i+1}'][0] or θ[i] > limits[f'joint_{i+1}'][1]:
            return False, f"Joint {i+1} out of limits: {θ[i]}"
    return True

print("Joint angles:", θ)
print("Joint angles within limits:", limits(θ))
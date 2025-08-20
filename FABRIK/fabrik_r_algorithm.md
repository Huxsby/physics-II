# FABRIK-R Algorithm Documentation

## Overview

FABRIK-R (Forward And Backward Reaching Inverse Kinematics for Robotics) is an extension of the original FABRIK algorithm specifically designed to handle manipulator robots with kinematic chains composed exclusively of 1-DOF (one degree of freedom) joints. This algorithm addresses the fundamental limitations of the original FABRIK when applied to robotics applications where joints have movement constraints.

## Problem Context

The original FABRIK algorithm treats each joint independently, which creates difficulties when applied to robotic manipulators that commonly feature chains of 1-DOF joints such as hinge and pivot joints. Most industrial manipulators, including cylindrical, SCARA, and KUKA robots, as well as medical robots like Da Vinci and assistive robots like JACO, fall into this category. The movement restriction of these robots implies a movement dependency between two sequential joints, which the original FABRIK cannot handle effectively.

## Core Algorithm: FABRIK-R

### Algorithm 1: Main FABRIK-R Process

```
Algorithm 1: FABRIK-R
Input: 
- Joint positions pi (i = 1, ..., n)
- Target position t
- Distance between joints di = |pi+1 - pi| (i = 1, ..., n)

Output: New joint position p'next

1. DEFINE_Φprev()
2. [p̂i, v̂i] = CREATE_NEW_Pi(Φprev, p'prev)
3. θ = DEFINE_Φi(vprev, p'prev)
4. p'next = ROT_QUATERNIONS(vprev, p̂i, v̂i, θ)
   // This function rotates v̂i and p̂i around vprev by θ degrees
```

### Key Principles

The FABRIK-R algorithm operates on two fundamental rules when applying constraints to joint pi in the forward reaching stage:

1. **Constraint Preservation**: The constraints of the previous joint pprev will not be violated during the movement calculation
2. **Plane Definition**: The constraint plane Φi must contain both pnext and p'prev to ensure proper joint connectivity

### Mathematical Foundation

The algorithm determines the normal vector n⃗i to the constraint plane using the orthogonality condition:

```
n⃗ · (pnext - pprev) = 0
```

This ensures that the normal vector is orthogonal to the vector connecting the next and previous joints. The algorithm then uses quaternion algebra to rotate vectors around any axis while respecting joint orientation relationships:

```
n⃗ = cos(2θ)v⃗ + (1 - cos(2θ))(l⃗ · v⃗)l⃗ + sin(2θ)(l⃗ × v⃗)
```

### Algorithm 2: Plane Definition Process

```
Algorithm 2: DEFINE_Φi
Input: 
- Direction vector l⃗
- Previous joint position pprev

Output: Rotation angle θ

1. j = FIND_CONCURRENT(i, vinit)
2. (α, β, γ) = pprev - pj
3. v⃗ = GENERATE_RANDOM(vprev, vj)
4. t⃗ = l⃗ × v⃗
5. K1 = αv1 + βv2 + γv3
6. K2 = (l⃗ · v⃗)(αl1 + βl2 + γl3)
7. K3 = αt1 + βt2 + γt3
8. A.SOLVE(cos(2θ)K1 + (1 - cos(2θ))K2 + sin(2θ)K3)
9. S.ROT_QUATERNIONS(l⃗, v⃗, A)
10. θ = S.COMPARE_SOLUTION()
```

## Implementation Process

### Forward Reaching Stage

The forward reaching stage begins by positioning the end-effector at the target location. For each joint pi working backwards from the end-effector:

1. **Plane Definition**: Define the constraint plane Φprev based on the previous joint's movement restrictions
2. **Projection**: Create a new joint position p̂i that respects the previous joint's constraints and maintains the proper link distance
3. **Constraint Application**: Calculate the plane Φi that is orthogonal to the vector connecting distant joints while respecting local constraints
4. **Rotation**: Use quaternions to rotate the joint into the calculated constraint plane

### Backward Reaching Stage

The backward reaching stage fixes the base joint in its original position and repositions each subsequent joint using the same constraint principles but working from base to end-effector.

## Special Cases and Solutions

### Parallel Joint Directions

When joints have identical direction vectors, the standard plane definition fails. The algorithm addresses this by implementing the FIND_CONCURRENT function, which identifies the next joint with a different actuation vector to establish a valid constraint plane.

### Multiple Solutions

The nonlinear nature of the constraint equations often produces multiple valid solutions. The algorithm selects the solution that minimizes the distance to achieve optimal convergence, typically choosing the solution with the smallest radius of movement.

## Performance Characteristics

FABRIK-R maintains the key advantages of the original FABRIK algorithm while extending functionality to constrained systems:

- **Fast Convergence**: Achieves solutions in fewer iterations compared to traditional methods like CCD and Jacobian-based approaches
- **Low Computational Cost**: Maintains simple point-distance-line calculations without complex matrix operations
- **Singularity Avoidance**: Does not suffer from the singularity problems that affect Jacobian-based methods
- **Local Processing**: Each joint is processed considering only local constraints, avoiding dependency on the entire kinematic chain

## Limitations and Considerations

The algorithm shares some limitations with the original FABRIK:

- **Orientation Control**: Provides position solutions but does not directly control end-effector orientation
- **Obstacle Avoidance**: Does not include collision detection or obstacle avoidance mechanisms
- **Unique Solutions**: Generates a single solution per iteration without consideration of alternative valid configurations

## Symbol and Notation Legend

### Joint and Position Variables
- **pi**: Position of the i-th joint in 3D space
- **p'i**: New calculated position of the i-th joint after algorithm iteration
- **p̂i**: Projected or intermediate position of joint i during constraint application
- **pprev**: Previous joint in the kinematic chain (pi+1 in forward reaching)
- **pnext**: Next joint in the kinematic chain relative to current joint
- **t**: Target position for the end-effector
- **n**: Total number of joints in the manipulator

### Distance and Link Parameters
- **di**: Distance (length) between consecutive joints pi and pi+1
- **dprev**: Distance from the previous joint to the current joint

### Constraint Planes and Vectors
- **Φi**: Constraint plane for joint i that defines allowable movement directions
- **Φprev**: Constraint plane for the previous joint
- **n⃗i**: Normal vector to the constraint plane Φi
- **v⃗**: Direction vector representing joint actuation direction
- **v⃗prev**: Direction vector of the previous joint
- **l⃗**: Line vector passing through joints, used for quaternion rotations
- **t⃗**: Cross product vector (l⃗ × v⃗) used in constraint calculations

### Angular and Rotation Parameters
- **θ**: Rotation angle calculated for quaternion-based joint positioning
- **2θ**: Double angle used in quaternion rotation formulas

### Mathematical Components
- **K1, K2, K3**: Intermediate calculation terms used in the constraint plane equation
- **(α, β, γ)**: Components of the vector difference between joints
- **(x, y, z)**: Cartesian coordinate components for position vectors
- **(n1, n2, n3)**: Components of the normal vector n⃗
- **(v1, v2, v3)**: Components of the direction vector v⃗
- **(l1, l2, l3)**: Components of the line vector l⃗
- **(t1, t2, t3)**: Components of the cross product vector t⃗

### Geometric Operations
- **|·|**: Magnitude or length of a vector
- **·**: Dot product operation between vectors
- **×**: Cross product operation between vectors
- **‖·‖**: Norm (magnitude) of a vector

### Algorithm Functions
- **DEFINE_Φprev()**: Function to establish the constraint plane of the previous joint
- **CREATE_NEW_Pi()**: Function to generate a new joint position respecting constraints
- **DEFINE_Φi()**: Function to calculate the constraint plane for joint i
- **ROT_QUATERNIONS()**: Function to perform quaternion-based rotation
- **FIND_CONCURRENT()**: Function to locate the next joint with different actuation direction
- **GENERATE_RANDOM()**: Function to create a random vector within constraint boundaries

## Applications

FABRIK-R is particularly suitable for industrial manipulators with sequential 1-DOF joints, medical robotics applications requiring precise positioning, assistive robotics where smooth and natural movements are essential, and real-time applications requiring fast inverse kinematics solutions. The algorithm has been successfully tested on various manipulator configurations, including robots with mixed pivot and hinge joint sequences, demonstrating consistent convergence to valid solutions within the reachable workspace.
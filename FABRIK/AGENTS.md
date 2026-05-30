# AGENTS.md - Governance for FABRIK Subtree

Scope: This file governs all changes under FABRIK/.

## 1) Mission

Implement and maintain FABRIK behavior aligned with primary references.
Do not replace normative joint semantics with convenience shortcuts without explicit documentation and validation.

## 2) Normative Sources (priority order)

1. AL11 (primary): Aristidou, A., Lasenby, J. "FABRIK: a fast, iterative solver for the inverse kinematics problem".
   local: docs/02.1-FABRIK.pdf
   DOI: 10.1016/j.gmod.2011.05.003
2. ACL16 (constraints extension): Aristidou, A., Chrysanthou, Y., Lasenby, J. "Extending FABRIK with model constraints".
  local: docs/01.1.2-Extending FABRIK with model constraints.pdf
   DOI: 10.1002/cav.1630
3. CALIKO software paper (practical reference):
   DOI: 10.5334/jors.116
4. Local reference implementations:
   - FABRIK/referencias/FABRIK_chain_3D-master/fabrik_chain_3d/FABRIK.py
   - FABRIK/referencias/FABRIK_chain_3D-master/fabrik_chain_3d/Joint.py
   - FABRIK/referencias/FABRIK_Full_Body-master/fabrik_full_body/constraints.py

If full paper text is not available in-session, use local reference implementations as executable ground truth and record the limitation in commit notes or PR notes.

## 3) Constraint Semantics (must follow)

- BALL:
  - Cone limit between incoming and outgoing segment directions.
  - Not equivalent to a revolute joint angle limit.
- GLOBAL_HINGE:
  - Rotation constrained in a plane with global fixed hinge axis.
  - Optional signed-angle clamp around reference axis.
- LOCAL_HINGE:
  - Rotation constrained in a plane using hinge axis/reference defined in local frame of parent segment.
  - Requires transforming local axes using parent-frame rotation each iteration.

Rule: For serial robot revolute joints (e.g., Niryo J1-J5), target semantics are LOCAL_HINGE unless a documented exception is approved.

## 4) Allowed Deviations and Required Labeling

Temporary deviation is allowed only when convergence/regression is demonstrated.

If deviation exists (example: mapping revolute-perpendicular to BALL), all of the following are mandatory:

1. Code comment near the decision says "transitional deviation".
2. Documentation states why it exists and what test evidence supports it.
3. A tracked task exists to return to normative semantics.
4. Test output baseline before/after is recorded.

Do not describe temporary deviation as "correct interpretation".

## 5) Source-of-Truth Files in this Repo

- Core solver: FABRIK/fabrik_core/fabrik_serial_solver.py
- Tests: FABRIK/tests/test_fabrik_niryo.py
- Technical constraints doc: FABRIK/docs/FABRIK_joint_constraints.md
- Niryo configuration: config/robot-niryo.yaml
- Roadmap/history notes: FABRIK/FABRIK_README.md

## 6) Change Workflow (mandatory)

Before coding:

1. Read relevant section in normative sources.
2. Compare target behavior against local reference implementation.
3. Write expected behavior in 3-5 bullets in task notes.

During coding:

1. Keep edits minimal and localized.
2. Preserve existing behavior outside target scope.
3. Mark any non-normative workaround explicitly.

After coding:

1. Run FABRIK/tests/test_fabrik_niryo.py.
2. Report convergence and limits summary for both batteries.
3. Confirm docs and code comments still match actual behavior.

## 7) Niryo Context Snapshot (for quick orientation)

- Joint indexing used by tests/solver:
  - J0 Base, J1 Hombro, J2 Brazo, J3 Codo, J4 Antebrazo, J5 Muneca.
- High-risk confusion to avoid:
  - J3 and J4 are different physical joints and have different limits.
  - BALL local deflection and absolute revolute angle are not the same quantity.

## 8) Rejection Criteria for New Changes

Reject a change if any of the following is true:

- It changes constraint semantics without citing normative source behavior.
- It introduces global-axis hinge assumptions for joints that are local-frame hinges.
- It updates code but not the matching documentation/comments.
- It claims "paper compliant" without evidence against references.

## 9) Minimal Evidence Template for Constraint Changes

Include in change summary:

1. Normative rule applied (AL11/ACL16/CALIKO reference path or DOI).
2. Files changed.
3. Before/after convergence and limits counts.
4. Known tradeoff or remaining deviation.

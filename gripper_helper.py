def equivalent_two_side_ratio(r_one_side: float, gripper_upper: float = 1.0) -> float:
    """
    Return a two-side ratio that produces the same cmd at v=1 as one-side ratio r_one_side.
    If result is <0 it is clamped to 0.
    """
    c = gripper_upper / 2.0
    if (1 - c) == 0:
        return r_one_side  # degenerate case
    r_two = (r_one_side - c) / (1 - c)
    return max(0.0, r_two)

def equivalent_one_side_ratio(r_two_side: float, gripper_upper: float = 1.0) -> float:
    """
    Return a one-side ratio that produces the same cmd at v=1 as two-side ratio r_two_side.
    If result is <0 it is clamped to 0.
    """
    c = gripper_upper / 2.0
    r_one = r_two_side * (1 - c) + c
    return max(0.0, r_one)

# def clamp_gripper_ratio(ratio: float, gripper_upper: float = 1.0) -> float:
#     """
#     Clamp the gripper ratio to be within [0, gripper_upper].
#     """
#     return max(0.0, min(ratio, gripper_upper))

# def is_valid_gripper_ratio(ratio: float, gripper_upper: float = 1.0) -> bool:
#     """
#     Check if the gripper ratio is within [0, gripper_upper].
#     """
#     return 0.0 <= ratio <= gripper_upper

# def gripper_cmd_from_ratio(ratio: float, gripper_upper: float = 1.0) -> float:
#     """
#     Convert a gripper ratio to a command value in the range [-1, 1].
#     """
#     if not is_valid_gripper_ratio(ratio, gripper_upper):
#         raise ValueError(f"Gripper ratio {ratio} is out of bounds [0, {gripper_upper}]")
#     return (ratio / gripper_upper) * 2.0 - 1.0

# print(equivalent_two_side_ratio(1.7))  # Example usage
# print(equivalent_one_side_ratio(2.4))  # Example usage

# print(equivalent_two_side_ratio(2))  # Example usage
# print(equivalent_one_side_ratio(3))  # Example usage

def combined_scale(v, r1, r2, gripper_upper=1.0):
    c = gripper_upper / 2.0
    # step 1: one-side
    v1 = np.clip(v * r1, 0.0, 1.0)
    # step 2: two-side around center
    cmd = c + (v1 - c) * r2
    return np.clip(cmd, 0.0, 1.0)
                   
import numpy as np
# from gripper_helper import equivalent_two_side_ratio

# Dummy predicted gripper values (vla output) in [0,1]
v = np.linspace(0.0, 1.0, 21)  # 21 samples from fully open (0) to fully closed (1)

# pick a one-side ratio (example)
r_one_side = 2

# compute equivalent two-side ratio that matches cmd at v=1
r_two_side = equivalent_two_side_ratio(r_one_side)  # uses gripper_upper=1.0 by default

# center used in two-side mode (gripper_upper/2)
center = 1.0 / 2.0

# apply mappings (clip to [0,1] same as your code)
one_side_cmd = np.clip(v * r_one_side, 0.0, 1.0)
# two_side_cmd = np.clip(center*r_two_side + (v - center) * r_two_side, 0.0, 1.0)
two_side_cmd = np.clip(center*r_two_side + (v - center) * r_two_side, 0.0, 1.0)

# print a compact table
print(f"r_one_side={r_one_side:.3f}, r_two_side={r_two_side:.3f}, center={center:.3f}\n")
print("v    one_side_cmd    two_side_cmd")
for vi, o, t in zip(v, one_side_cmd, two_side_cmd):
    print(f"{vi:0.2f}  {o:0.4f}          {t:0.4f}")
# r_one_side=1.700, r_two_side=2.400, center=0.500

# v    one_side_cmd    two_side_cmd
# 0.00  0.0000          0.0000
# 0.05  0.0850          0.0000
# 0.10  0.1700          0.0000
# 0.15  0.2550          0.0000
# 0.20  0.3400          0.0000
# 0.25  0.4250          0.0000
# 0.30  0.5100          0.0200
# 0.35  0.5950          0.1400
# 0.40  0.6800          0.2600
# 0.45  0.7650          0.3800
# 0.50  0.8500          0.5000
# 0.55  0.9350          0.6200
# 0.60  1.0000          0.7400
# 0.65  1.0000          0.8600
# 0.70  1.0000          0.9800
# 0.75  1.0000          1.0000
# 0.80  1.0000          1.0000
# 0.85  1.0000          1.0000
# 0.90  1.0000          1.0000
# 0.95  1.0000          1.0000
# 1.00  1.0000          1.0000

# r_one_side=2.000, r_two_side=3.000, center=0.500

# v    one_side_cmd    two_side_cmd
# 0.00  0.0000          0.0000
# 0.05  0.1000          0.0000
# 0.10  0.2000          0.0000
# 0.15  0.3000          0.0000
# 0.20  0.4000          0.0000
# 0.25  0.5000          0.0000
# 0.30  0.6000          0.0000
# 0.35  0.7000          0.0500
# 0.40  0.8000          0.2000
# 0.45  0.9000          0.3500
# 0.50  1.0000          0.5000
# 0.55  1.0000          0.6500
# 0.60  1.0000          0.8000
# 0.65  1.0000          0.9500
# 0.70  1.0000          1.0000
# 0.75  1.0000          1.0000
# 0.80  1.0000          1.0000
# 0.85  1.0000          1.0000
# 0.90  1.0000          1.0000
# 0.95  1.0000          1.0000
# 1.00  1.0000          1.0000

# r_one_side=1.429, r_two_side=1.857, center=0.500

# v    one_side_cmd    two_side_cmd
# 0.00  0.0000          0.0000
# 0.05  0.0714          0.0000
# 0.10  0.1429          0.0000
# 0.15  0.2143          0.0000
# 0.20  0.2857          0.0000
# 0.25  0.3571          0.0357
# 0.30  0.4286          0.1286
# 0.35  0.5000          0.2214
# 0.40  0.5714          0.3143
# 0.45  0.6429          0.4071
# 0.50  0.7143          0.5000
# 0.55  0.7857          0.5929
# 0.60  0.8571          0.6857
# 0.65  0.9286          0.7786
# 0.70  1.0000          0.8714
# 0.75  1.0000          0.9643
# 0.80  1.0000          1.0000
# 0.85  1.0000          1.0000
# 0.90  1.0000          1.0000
# 0.95  1.0000          1.0000
# 1.00  1.0000          1.0000

r1 = r_one_side
r2 = r_two_side
# one_side = np.clip(v * r1, 0.0, 1.0)
one_side = v * r1
two_side_direct = center*r2 + (v - center) * r2  # if you wanted to apply two-side directly on raw v
two_side_centered = center + (v - center) * r2  # if you wanted to apply two-side directly on raw v
combined = combined_scale(v, r1, r2)

def combined_centered_scale(v, r1, r2, gripper_upper=1.0):
    c = gripper_upper / 2.0
    # scale around center (no intermediate clipping)
    v1 = c + (v - c) * r1
    cmd = c + (v1 - c) * r2
    return np.clip(cmd, 0.0, gripper_upper)

combined_centered = combined_centered_scale(v, r1, r2)

# print a compact table
print("v\tone_side\ttwo_side_direct\ttwo_side_centered\tcombined\tcombined_centered")
for vi, o, t, c, comb, comb_c in zip(v, one_side, two_side_direct, two_side_centered, combined, combined_centered):
    print(f"{vi:0.2f}\t{o:0.3f}\t\t{t:0.3f}\t\t{c:0.3f}\t\t{comb:0.3f}\t\t{comb_c:0.3f}")

print("\nAfter applying [0, 1] clipping:")
print("v\tone_side\ttwo_side_direct\ttwo_side_centered\tcombined\tcombined_centered")
# apply [0, 1] to all and print
for vi, o, t, c, comb, comb_c in zip(v, one_side, two_side_direct, two_side_centered, combined, combined_centered):
    print(f"{np.clip(vi, 0.0, 1.0):0.2f}\t{np.clip(o, 0.0, 1.0):0.3f}\t\t{np.clip(t, 0.0, 1.0):0.3f}\t\t{np.clip(c, 0.0, 1.0):0.3f}\t\t{np.clip(comb, 0.0, 1.0):0.3f}\t\t{np.clip(comb_c, 0.0, 1.0):0.3f}")
# compute the ratio to vi and print
print("\nRatios to vi:")
print("v\tone_side\ttwo_side_direct\ttwo_side_centered\tcombined\tcombined_centered")
for vi, o, t, c, comb, comb_c in zip(v, one_side, two_side_direct, two_side_centered, combined, combined_centered):
    print(f"{np.clip(vi, 0.0, 1.0):0.2f}\t{np.clip(o, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(t, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(c, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(comb, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(comb_c, 0.0, 1.0)/vi:0.3f}")
# compute the ratio to vi and print

# for vi, r_2_v, r_1_v, r_combined_v in zip(v, one_side/vi, two_side_direct/one_side, combined/vi):
#     print(f"{vi:0.2f}\t\t{r_2_v:0.3f}\t\t{r_1_v:0.3f}\t\t{r_combined_v:0.3f}")

def variable_centered_scale(v, r_mid=1.0, r_edge=2.5, gripper_upper=1.0):
    """
    Scale around center so center stays fixed and gain increases toward edges.
    r_mid: multiplier at center (v==c)
    r_edge: multiplier at edges (v==0 or 1)
    """
    c = gripper_upper / 2.0
    # weight in [0,1], 0 at center, 1 at edges
    edge_weight = 4.0 * (v - c) ** 2
    r_v = r_mid + (r_edge - r_mid) * edge_weight
    cmd = c + (v - c) * r_v
    return np.clip(cmd, 0.0, gripper_upper)

def variable_one_side_scale(v, r_mid=1.0, r_edge=2.5, gripper_upper=1.0):
    """
    Scale from zero with gain increasing toward edges (symmetric).
    """
    c = gripper_upper / 2.0
    edge_weight = 4.0 * (v - c) ** 2
    r_v = r_mid + (r_edge - r_mid) * edge_weight
    cmd = v * r_v
    return np.clip(cmd, 0.0, gripper_upper)

# quick demo
v = np.linspace(0.0, 1.0, 21)
print("v  centered")
print(np.column_stack([v, variable_centered_scale(v, r_mid=1.0, r_edge=3.0)]))
print("v  one_side")
print(np.column_stack([v, variable_one_side_scale(v, r_mid=1.0, r_edge=3.0)]))

# also compare these two with one-side and two side direct
r_one_side = 2
r_two_side = equivalent_two_side_ratio(r_one_side)
one_side = np.clip(v * r_one_side, 0.0, 1.0)
center = 1.0 / 2.0
two_side_direct = np.clip(center*r_two_side + (v - center) * r_two_side, 0.0, 1.0)
centered = variable_centered_scale(v, r_mid=1.0, r_edge=4.0)
one_side_var = variable_one_side_scale(v, r_mid=1.0, r_edge=4.0)
print("v\tone_side\ttwo_side_direct\tcentered\tone_side_var")
for vi, o, t, c, ov in zip(v, one_side, two_side_direct, centered, one_side_var):
    print(f"{vi:0.2f}\t{o:0.3f}\t\t{t:0.3f}\t\t{c:0.3f}\t\t{ov:0.3f}")
# ration ratio of centered to vi
print("\nRatios to vi:")
print("v\tone_side\ttwo_side_direct\tcentered\tone_side_var")
for vi, o, t, c, ov in zip(v, one_side, two_side_direct, centered, one_side_var):
    print(f"{vi:0.2f}\t{np.clip(o, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(t, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(c, 0.0, 1.0)/vi:0.3f}\t\t{np.clip(ov, 0.0, 1.0)/vi:0.3f}")


def scale_with_margin(v, margin=0.4):
    """
    Map v in [0,1] to [-margin, 1+margin].
    v may be scalar or numpy array.
    """
    scale = 1.0 + 2.0 * margin
    return v * scale - margin

def unscale_with_margin(cmd, margin=0.4):
    """Inverse mapping: map cmd in [-margin, 1+margin] back to [0,1]."""
    scale = 1.0 + 2.0 * margin
    return (cmd + margin) / scale

# Optional safe wrapper with final clipping to a desired range:
import numpy as np
def scale_with_margin_clipped(v, margin=0.4, clip_low=None, clip_high=None):
    cmd = scale_with_margin(v, margin)
    if clip_low is None: clip_low = -margin
    if clip_high is None: clip_high = 1.0 + margin
    return np.clip(cmd, clip_low, clip_high)

# Quick demo
import numpy as np
v = np.linspace(0.0, 1.0, 11)
print(scale_with_margin(v, margin=0.4))
# -> [-0.4  -0.22 -0.04  0.14  0.32  0.5   0.68  0.86  1.04  1.22  1.4 ]

# print the vi and scaled values in a table, also the clipped version and ratio of clipped to vi
print("v\tscaled\tclipped\tratio")
for vi, sv in zip(v, scale_with_margin(v, margin=0.4)):
    clipped = np.clip(sv, 0, 1)
    ratio = clipped / vi if vi != 0 else 0
    print(f"{vi:0.2f}\t{sv:0.3f}\t{clipped:0.3f}\t{ratio:0.3f}")

# print with radius to degree
def radius_to_degree(radius):
    return radius * 180.0 / np.pi
# What if first one-side scale for faster closing, then margin scaling?
r_one_side = 1.0/0.98
v = np.linspace(0.0, 1.0, 11)
one_side = np.clip(v * r_one_side, 0.0, 1.0)
scaled = scale_with_margin(one_side, margin=0.1)
r_two_side = equivalent_two_side_ratio(r_one_side)
two_side_direct = v * center * r_two_side + (v - center) * r_two_side  # if you wanted to apply two-side directly on raw v
clipped = np.clip(scaled, 0.0, 1.0)
print("v\tone_side\tscaled\tscaled_clipped\tratio")
for vi, o, s, td in zip(v, one_side, scaled, two_side_direct):
    clipped = np.clip(s, 0, 1)
    ratio = clipped / vi if vi != 0 else 0
    # print(f"{vi:0.2f}\t{o:0.3f}\t{s:0.3f}\t{clipped:0.3f}\t{ratio:0.3f}")
    print(f"{radius_to_degree(vi):0.2f}\t{radius_to_degree(o):0.2f}\t\t{radius_to_degree(s):0.2f}\t\t{radius_to_degree(clipped):0.2f}\t{radius_to_degree(ratio*vi):0.2f}\t{radius_to_degree(td):0.2f}")

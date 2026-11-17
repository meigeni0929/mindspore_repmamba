"""Static complexity analysis for both selective-scan backends."""

import argparse
import math


def analyze(batch, length, channels, state):
    expanded = batch * channels * length * state
    reference = {
        "python_loop_iterations": length,
        "prefix_rounds": 0,
        "dominant_state_elements": batch * channels * state,
        "transition_elements": expanded,
        "operator_pattern": "L sequential exp/state-update/reduce steps",
        "estimated_primitive_calls": 15 * length + 5,
    }
    optimized = {
        "python_loop_iterations": 0,
        "prefix_rounds": math.ceil(math.log2(max(1, length))),
        "dominant_state_elements": expanded,
        "transition_elements": expanded,
        "operator_pattern": "log2(L) shift/concat/multiply/add prefix rounds",
        "estimated_primitive_calls": 12 * math.ceil(math.log2(max(1, length))) + 20,
    }
    return reference, optimized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--state", type=int, default=16)
    args = parser.parse_args()
    reference, optimized = analyze(args.batch, args.length, args.channels, args.state)
    print("reference:", reference)
    print("optimized:", optimized)
    print("expanded-state memory ratio:",
          optimized["dominant_state_elements"] /
          max(1, reference["dominant_state_elements"]))


if __name__ == "__main__":
    main()

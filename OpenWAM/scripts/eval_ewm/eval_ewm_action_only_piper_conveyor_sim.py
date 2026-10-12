"""RGB/state-only policy evaluation in the shared Piper conveyor scene."""
from eval_ewm_piper_conveyor_sim import main

if __name__ == "__main__":
    main(policy_mode="action_only")

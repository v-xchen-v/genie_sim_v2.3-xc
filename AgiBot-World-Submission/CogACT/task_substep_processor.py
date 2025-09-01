"""
Task substep progression logic for robot task execution.

This module contains all the logic related to task instruction management,
substep progression strategies, and task configuration.
"""

import numpy as np
from config_loader import get_config

# Load configuration
config = get_config()


def get_instruction(task_name):
    """
    Get the instruction string for a given task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        str: Semicolon-separated instruction string
        
    Raises:
        ValueError: If task does not exist
    """
    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
        # lang = "Pick up the grape juice on the table with the right arm.;Place the grape juice on the shelf where the grape juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang


def get_num_substeps(task_name):
    """
    Get the number of substeps for a given task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        int: Number of substeps in the task
    """
    lang = get_instruction(task_name)
    substeps = [action.strip() for action in lang.split(";") if action.strip()]
    return len(substeps)


def get_task_progression_config():
    """
    Get task-specific configuration for substep progression.
    
    Returns:
        dict: Configuration dictionary containing min/max inference counters
    """
    return {
        # "progress_thresholds": {
        #     "iros_pack_in_the_supermarket": 0.3,
        #     "iros_restock_supermarket_items": 0.9,
        #     "iros_stamp_the_seal": 0.99,
        #     "iros_clear_the_countertop_waste": 0.4,
        #     "iros_clear_table_in_the_restaurant": 0.6,
        #     "iros_heat_the_food_in_the_microwave": 0.6,
        #     "iros_open_drawer_and_store_items": 0.97,
        #     "iros_pack_moving_objects_from_conveyor": 0.4,
        #     "iros_pickup_items_from_the_freezer": 0.4,
        #     "iros_make_a_sandwich": 0.9,
        # },
        "min_inference_counters": {
            "iros_pack_in_the_supermarket": 12,  # 1-8 steps
            "iros_restock_supermarket_items": 5,  # 1-16 steps
            "iros_stamp_the_seal": 10,
            "iros_clear_the_countertop_waste": 6,
            "iros_clear_table_in_the_restaurant": 10,
            "iros_heat_the_food_in_the_microwave": 40,
            "iros_open_drawer_and_store_items": 20,
            "iros_pack_moving_objects_from_conveyor": 12, # for steps: [0, 4, 8, 12], 6 is enough for pickup directly but not enough for failed and retry, 12 is enough for failed and retry
            "iros_pickup_items_from_the_freezer": 24,
            "iros_make_a_sandwich": 12,
        },
        "max_inference_counters": {
            "iros_pack_in_the_supermarket": 48,  # 1-8 steps
            "iros_heat_the_food_in_the_microwave": 40,  # 1-8 steps
            # "iros_restock_supermarket_items": 48,  # 1-8 steps
            # "iros_open_drawer_and_store_items": 40,  # 1-8 steps
            "iros_open_drawer_and_store_items": 20,  # 1-16 steps
        }
    }


def extract_action_task_substep_progress(action_raw):
    """
    Get the task substep progress from the action raw.
    
    Args:
        action_raw: Raw action containing progress information
        
    Returns:
        Progress information from the action
    """
    return action_raw["PROGRESS"]  # shape: []


def check_progress_based_advancement(task_substep_progress, task_name, substep_inference_counter, config, logger):
    """
    Strategy 3: Check if substep should advance based on progress threshold OR max counter.
    
    Args:
        task_substep_progress: Current task substep progress
        task_name: Name of the task
        substep_inference_counter: Current inference counter
        config: Task configuration
        logger: Logger instance
        
    Returns:
        bool: Whether substep should advance
    """
    progress_threshold = 0.95  # High threshold for reliable progress signal
    max_inference_counter = config["max_inference_counters"].get(task_name, 20)
    
    progress_list = np.array(task_substep_progress[0])
    current_progress = task_substep_progress[0][0]
    
    # Check advancement conditions
    counter_exceeded = substep_inference_counter >= max_inference_counter
    progress_exceeded = np.any(progress_list > progress_threshold)
    
    should_advance = counter_exceeded or progress_exceeded
    
    if logger is not None:
        if should_advance:
            reason = "counter exceeded" if counter_exceeded else "progress exceeded"
            logger.info(f"✅ ADVANCING (Strategy 3): {reason} - Progress ({current_progress:.3f}), Counter ({substep_inference_counter})")
        else:
            logger.info(f"❌ NOT ADVANCING (Strategy 3): Progress ({current_progress:.3f} <= {progress_threshold}), Counter ({substep_inference_counter} < {max_inference_counter})")
    
    return should_advance


def check_restrict_progress_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """
    Strategy 2: Strictly follow progress signal only - advance when progress > threshold.
    
    Args:
        task_substep_progress: Current task substep progress
        task_name: Name of the task
        substep_inference_counter: Current inference counter
        config: Task configuration
        
    Returns:
        bool: Whether substep should advance
    """
    progress_threshold = 0.95  # High threshold for reliable progress signal
    
    progress_list = np.array(task_substep_progress[0])
    current_progress = task_substep_progress[0][0]
    
    # Only advance based on progress, ignore counter
    progress_exceeded = np.any(progress_list > progress_threshold)
    should_advance = progress_exceeded
    
    if should_advance:
        print(f"✅ ADVANCING (Strategy 2): Progress exceeded - Progress ({current_progress:.3f} > {progress_threshold})")
    else:
        print(f"❌ NOT ADVANCING (Strategy 2): Progress not met - Progress ({current_progress:.3f} <= {progress_threshold})")
    
    return should_advance


def check_legacy_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """
    Strategy 1: Check if substep should advance based on legacy logic (progress AND counter thresholds).
    
    Args:
        task_substep_progress: Current task substep progress
        task_name: Name of the task
        substep_inference_counter: Current inference counter
        config: Task configuration
        
    Returns:
        bool: Whether substep should advance
    """
    progress_threshold = config["progress_thresholds"].get(task_name, 0.4)
    min_inference_counter = config["min_inference_counters"].get(task_name, 6)
    
    current_progress = task_substep_progress[0][0]
    progress_met = current_progress > progress_threshold
    counter_met = substep_inference_counter >= min_inference_counter
    
    should_advance = progress_met and counter_met
    
    if should_advance:
        print(f"✅ ADVANCING (Strategy 1): Progress ({current_progress:.3f} > {progress_threshold}) AND Counter ({substep_inference_counter} >= {min_inference_counter})")
    else:
        progress_ok = "✅" if progress_met else "❌"
        counter_ok = "✅" if counter_met else "❌"
        print(f"STAYING (Strategy 1): Progress {progress_ok} ({current_progress:.3f} > {progress_threshold}) AND Counter {counter_ok} ({substep_inference_counter} >= {min_inference_counter})")
    
    return should_advance


def check_restrict_inference_count_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """
    Strategy 4: Advance only when inference counter exceeds minimum threshold (ignore progress).
    
    Args:
        task_substep_progress: Current task substep progress
        task_name: Name of the task
        substep_inference_counter: Current inference counter
        config: Task configuration
        
    Returns:
        bool: Whether substep should advance
    """
    min_inference_counter = config["min_inference_counters"].get(task_name, 6)
    current_progress = task_substep_progress[0][0]
    
    # Only advance based on counter, ignore progress completely
    counter_exceeded = substep_inference_counter >= min_inference_counter
    should_advance = counter_exceeded
    
    if should_advance:
        print(f"✅ ADVANCING (Strategy 4): Counter exceeded - Counter ({substep_inference_counter} >= {min_inference_counter}), Progress ({current_progress:.3f})")
    else:
        print(f"❌ NOT ADVANCING (Strategy 4): Counter not met - Counter ({substep_inference_counter} < {min_inference_counter}), Progress ({current_progress:.3f})")
    
    return should_advance


def handle_substep_progression(action, task_name, curr_task_substep_index, substep_inference_counter, model_input, mode="by_progress", logger=None):
    """
    Handle substep progression logic based on task progress and inference counter.
    
    Args:
        action: Action containing progress information
        task_name: Name of the current task
        curr_task_substep_index: Current substep index
        substep_inference_counter: Current inference counter for this substep
        model_input: Model input containing task description
        mode: Strategy selection:
            - "legacy": Strategy 1 - Advance when progress > low_threshold AND counter >= min_counter
            - "restrict_progress": Strategy 2 - Advance only when progress > high_threshold (strict progress following)
            - "by_progress": Strategy 3 - Advance when progress > high_threshold OR counter >= max_counter
            - "restrict_inference_count": Strategy 4 - Advance only when counter >= min_counter (ignore progress)
        logger: Logger instance for logging
    
    Returns:
        tuple: (new_curr_task_substep_index, new_substep_inference_counter)
    """
    substep_inference_counter += 1
    task_substep_progress = extract_action_task_substep_progress(action)
    
    # Log current state
    if logger is not None:
        # Log current state
        logger.info(f"------------Task substep progress: {task_substep_progress[0][0]}------------")
        logger.info(f"Substep: {curr_task_substep_index}, Inference Counter: {substep_inference_counter}, Mode: {mode}")
        logger.info(f"Instruction: {model_input['task_description']}")

    # Get task configuration
    task_config = get_task_progression_config()
    task_config["progress_thresholds"] = config.get_task_progression_config()["progress_thresholds"]
    
    # Determine if we should advance based on the selected strategy
    if mode == "legacy":
        should_advance = check_legacy_advancement(task_substep_progress, task_name, substep_inference_counter, task_config)
    elif mode == "restrict_progress":
        should_advance = check_restrict_progress_advancement(task_substep_progress, task_name, substep_inference_counter, task_config)
    elif mode == "by_progress":
        should_advance = check_progress_based_advancement(task_substep_progress, task_name, substep_inference_counter, task_config, logger)
    elif mode == "restrict_inference_count":
        should_advance = check_restrict_inference_count_advancement(task_substep_progress, task_name, substep_inference_counter, task_config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'legacy', 'restrict_progress', 'by_progress', or 'restrict_inference_count'")

    # Update indices if advancing
    if should_advance:
        curr_task_substep_index += 1
        substep_inference_counter = 0  # Reset counter for new substep
    
    return curr_task_substep_index, substep_inference_counter

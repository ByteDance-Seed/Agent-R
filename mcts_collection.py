"""
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
from fastchat.model.model_adapter import get_conversation_template
from mcts_utils.llm_server import *
from agentenv.envs import WebshopEnvClient, SciworldEnvClient, TextCraftEnvClient
import argparse
import os

Task = os.environ["TASK"]

if Task == "webshop":
    from mcts_utils.webshop.mcts_ws import *
elif Task == "sciworld":
    from mcts_utils.sciworld.mcts_sci import *
elif Task == "textcraft":
    from mcts_utils.textcraft.mcts_tc import *

def initialize_environment_webshop(env_server_base: str, data_len: int):
    return WebshopEnvClient(
        env_server_base=env_server_base,
        data_len=data_len,
    )

def initialize_environment_sciworld(env_server_base: str, data_len: int):
    return SciworldEnvClient(
        env_server_base=env_server_base,
        data_len=data_len,
    )

def initialize_environment_textcraft(env_server_base: str, data_len: int):
    return TextCraftEnvClient(
        env_server_base=env_server_base,
        data_len=data_len,
    )

def setup_conversation(env):
    conversation = list(env.conversation_start)
    conv = get_conversation_template('gpt-4')

    conv.append_message(conv.roles[0], conversation[0]["value"])
    conv.append_message(conv.roles[1], 'Ok.')
    if os.environ["TASK"] == "webshop":
        conv.append_message(conv.roles[0], env.observe())
    else:
        conv.append_message(conv.roles[0], env.info["observation"])
    return conv

def perform_mcts_search(Task, calling, env, conv, model_name, idx):

    recent_actions = []
    mcts_search = ExtendedMCTS(calling=calling, max_len=int(os.environ["MAX_TOKEN_LENGTH"]), model_name=model_name, env=env, idx=idx)

    mcts_search.search(env, conv, recent_actions)
    dir_path = f"mcts_result/{Task}/{model_name}"
    file_path = f"{dir_path}/search_results_{idx}.json"

    # 如果目录不存在则创建
    os.makedirs(dir_path, exist_ok=True)
    mcts_search.save(f"{file_path}")
    print("MCTS Done")

def initialize_environment(Task, env_server_base, data_len = 200):
    """
    Initializes the environment based on the task type.
    """
    if Task == "webshop":
        return initialize_environment_webshop(env_server_base, data_len)
    elif Task == "sciworld":
        return initialize_environment_sciworld(env_server_base, data_len)
    elif Task == "textcraft":
        return initialize_environment_textcraft(env_server_base, data_len)
    else:
        raise ValueError(f"Unknown Task: {Task}")

def load_task_data(Task):
    """
    Loads test and training data for the given task.
    """
    test_data = read_json(f"mcts_utils/{Task}/{Task}_test.json")
    train_data = read_json(f"mcts_utils/{Task}/{Task}_train_clean.json")
    task_inds = [ind["item_id"].replace(f"{Task}_", "") for ind in test_data]
    return task_inds, train_data

def process_task(Task, task_inds, train_data, model_name, env, calling, min, max):
    """
    Processes tasks for "webshop" and "textcraft".
    """
    train_ids = [i for i in range(1000)][min:max]
    for idx in train_ids:
        if str(idx) in task_inds or str(idx) not in train_data:
            continue

        dir_path = f"mcts_result/{Task}/{model_name}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/search_results_{idx}.json"
        if os.path.exists(file_path):
            print(f"{file_path} exists. Skipping.")
            continue

        env.reset(int(idx))
        conv = setup_conversation(env)
        perform_mcts_search(Task, calling, env, conv, model_name, idx)

def process_sciworld(Task, task_inds, task_num, task_iteration, model_name, env, calling):
    """
    Processes tasks for "sciworld".
    """
    for k in range(1, task_iteration + 1):
        if k >= len(task_inds[str(task_num)]):
            break
        
        idx = task_inds[str(task_num)][k]
        dir_path = f"mcts_result/{Task}/{model_name}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/search_results_{idx}.json"
        if os.path.exists(file_path):
            continue

        print(f"idx: {idx}")
        env.reset(int(idx))
        conv = setup_conversation(env)
        perform_mcts_search(Task, calling, env, conv, model_name, idx)

def main(Task, calling, min, max, task_num, model_name, env_server_base, task_iteration):
    """
    Main function to handle tasks based on their type.
    """
    # Initialize the environment
    env = initialize_environment(Task, env_server_base)

    # Process tasks based on the task type
    if Task in ["webshop", "textcraft"]:
        task_inds, train_data = load_task_data(Task)
        process_task(Task, task_inds, train_data, model_name, env, calling, min, max)
    elif Task == "sciworld":
        game_nums, task_inds = env.get_game_nums()
        process_sciworld(Task, task_inds, task_num, task_iteration, model_name, env, calling)
    else:
        print(f"Task '{Task}' is not supported.")


def main_script():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCTS for specified tasks.")
    parser.add_argument("--env_server_base", type=str, default="http://127.0.0.1:8000", help="Base URL of the environment server.")
    parser.add_argument("--task_num", type=int, default=1, help="Task number for processing (for sciworld).")
    parser.add_argument("--task_iteration", type=int, default=5, help="Number of iterations per task (for sciworld).")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06", help="Name of the model.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps for processing.")
    parser.add_argument("--min", type=int, default=0, help="Minimum range for processing tasks.")
    parser.add_argument("--max", type=int, default=500, help="Maximum range for processing tasks.")
    args = parser.parse_args()

    # Load environment variables
    env_server_base = args.env_server_base
    task_num = args.task_num
    task_iteration = args.task_iteration
    model_name = args.model_name
    min_range = args.min
    max_range = args.max
    calling = FuncCallOffline(model_name=model_name)

    # Get the task from the environment variable
    Task = os.environ.get("TASK")
    if not Task:
        raise ValueError("The TASK environment variable is not set.")

    # Process based on the task
    if Task == "sciworld":
        task_nums = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 12,
            13, 17, 18, 19, 20, 21, 22, 25, 26, 27,
            28, 29, 0
        ]
        # Iterate over specified task numbers
        for current_task_num in task_nums[min_range:max_range]:
            main(Task, calling, min_range, max_range, current_task_num, model_name, env_server_base, task_iteration)
    else:
        # Handle non-sciworld tasks
        main(Task, calling, min_range, max_range, task_num, model_name, env_server_base, task_iteration)

if __name__ == "__main__":
    main_script()
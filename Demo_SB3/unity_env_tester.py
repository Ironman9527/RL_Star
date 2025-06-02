import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zmq
import msgpack
import subprocess # 用于启动Unity可执行文件
import atexit     # 用于在Python退出时尝试关闭Unity
import time
import argparse
import os

# --- 和Unity的 RLDataStructs.cs 中 EnvironmentStateData 对应的观测大小 ---
# AgentData: Position(3) + Rotation(4 quat) + Velocity(3) + ForwardVector(3) = 13
# TargetData: Position(3) = 3
# RelativeDirectionToTarget(2) + DistanceToTarget(1) + RemainingStepsInEpisode(1) = 4
# 总计 = 13 + 3 + 4 = 20 个浮点数 (如果展平的话)
OBS_FLAT_SIZE = 20
ACTION_SIZE = 2 # [前进/后退, 左转/右转]

def flatten_observation(structured_obs):
    """
    将从Unity接收到的结构化观测数据展平为一维Numpy数组。
    顺序必须严格匹配Unity中构建观测的方式和此处解析的方式。
    """
    try:
        agent_data = structured_obs['agentData']
        target_data = structured_obs['targetData']
        
        flat_list = []
        flat_list.extend(agent_data.get('position', [0,0,0]))
        flat_list.extend(agent_data.get('rotation', [0,0,0,1]))
        flat_list.extend(agent_data.get('velocity', [0,0,0]))
        flat_list.extend(agent_data.get('forwardVector', [0,0,1]))
        flat_list.extend(target_data.get('position', [0,0,0]))
        flat_list.extend(structured_obs.get('relativeDirectionToTarget', [0,0]))
        flat_list.append(structured_obs.get('distanceToTarget', 0.0))
        flat_list.append(float(structured_obs.get('remainingStepsInEpisode', 0)))
        
        if len(flat_list) != OBS_FLAT_SIZE:
            print(f"Warning: Flattened observation size mismatch. Expected {OBS_FLAT_SIZE}, got {len(flat_list)}")
            # Pad with zeros or truncate if necessary, or raise error
            # For now, let's ensure it has the right size, padding with zeros
            if len(flat_list) < OBS_FLAT_SIZE:
                flat_list.extend([0.0] * (OBS_FLAT_SIZE - len(flat_list)))
            else:
                flat_list = flat_list[:OBS_FLAT_SIZE]

        return np.array(flat_list, dtype=np.float32)
    except KeyError as e:
        print(f"Error flattening observation: Missing key {e}. Observation was: {structured_obs}")
        return np.zeros(OBS_FLAT_SIZE, dtype=np.float32) # 返回一个零数组以避免崩溃
    except Exception as e:
        print(f"An unexpected error occurred during flattening: {e}. Observation was: {structured_obs}")
        return np.zeros(OBS_FLAT_SIZE, dtype=np.float32)


class UnityVectorEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30} # Gym metadata

    def __init__(self, executable_path=None, port=5555, num_envs=1,
                 unity_upspeed=1, unity_qps=10, unity_max_steps=1000,
                 launch_unity=True, worker_id=0, editor_debug_mode=False):
        super().__init__()
        self.num_envs = num_envs
        self.port = port + worker_id 
        self.unity_process = None
        self.editor_debug_mode = editor_debug_mode # 如果为True，则不启动exe，仅连接

        if executable_path and launch_unity and not self.editor_debug_mode:
            unity_exe_path = os.path.abspath(executable_path)
            if not os.path.exists(unity_exe_path):
                raise FileNotFoundError(f"Unity executable not found at {unity_exe_path}")

            launch_command = [
                unity_exe_path,
                "-ip", "127.0.0.1", # Unity客户端连接到Python服务端(本机)
                "-port", str(self.port),
                "-numEnvs", str(self.num_envs),
                "-upSpeed", str(unity_upspeed),
                "-qps", str(unity_qps),
                "-maxSteps", str(unity_max_steps),
                "-testMode", "false", # 确保Unity不在其内部测试模式，由Python控制AI
                "-logfile", f"unity_env_{worker_id}.log", # 为每个Unity实例生成日志
                "-batchmode", # 可选: 如果不需要Unity窗口显示，以批处理模式运行
                "-nographics" # 可选: 如果不需要Unity窗口显示
            ]
            # 移除 -batchmode 和 -nographics 如果你需要看到Unity窗口
            # launch_command = [cmd for cmd in launch_command if cmd not in ["-batchmode", "-nographics"]]

            print(f"Python: Launching Unity with command: {' '.join(launch_command)}")
            try:
                self.unity_process = subprocess.Popen(launch_command)
                print(f"Python: Unity process launched with PID: {self.unity_process.pid}")
                atexit.register(self.close) # 注册退出时清理函数
                print("Python: Waiting for Unity to initialize (15 seconds)...")
                time.sleep(5) # 给Unity足够的时间启动和初始化网络
            except Exception as e:
                print(f"Python: Failed to launch Unity: {e}")
                raise
        elif self.editor_debug_mode:
            print(f"Python: Editor Debug Mode. Assuming Unity Editor is running and will connect to port {self.port}.")
            print("Python: Please ensure ZmqRLClient in Unity Editor is NOT in its own 'isInTestMode'.")
        elif not launch_unity:
             print(f"Python: Not launching Unity. Assuming an external instance will connect to port {self.port}.")


        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP) # Python作为服务端
        print(f"Python: Binding ZMQ REP socket to tcp://*:{self.port}")
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"Python: ZMQ Server listening on port {self.port} for {self.num_envs} Unity environments...")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs, ACTION_SIZE), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)

        self._current_unity_batch_data = None
        self._is_first_reset_call = True

    def _receive_unity_batch(self):
        """等待并接收来自Unity的批量数据"""
        try:
            # print("Python: Waiting to receive message from Unity...")
            message_bytes = self.socket.recv()
            self._current_unity_batch_data = msgpack.unpackb(message_bytes, raw=False)
            if 'envDataList' not in self._current_unity_batch_data or not isinstance(self._current_unity_batch_data['envDataList'], list):
                print(f"Python: Received malformed batch data from Unity: {self._current_unity_batch_data}")
                # 可以选择抛出异常或尝试恢复
                return False # 表示接收失败

            # 按envId排序以确保Gym接口返回的数据顺序一致
            self._current_unity_batch_data['envDataList'].sort(key=lambda x: x.get('envId', -1))
            # print(f"Python: Received batch data for {len(self._current_unity_batch_data['envDataList'])} envs.")
            return True
        except zmq.error.ContextTerminated:
            print("Python: ZMQ context terminated, likely during shutdown.")
            return False
        except Exception as e:
            print(f"Python: Error receiving or unpacking data from Unity: {e}")
            self._current_unity_batch_data = None # 清除可能损坏的数据
            return False


    def _send_actions_to_unity(self, actions_list_of_dicts):
        """将批量动作数据发送给Unity"""
        try:
            python_batch_reply = {'actionsList': actions_list_of_dicts}
            reply_bytes = msgpack.packb(python_batch_reply)
            self.socket.send(reply_bytes)
            # print(f"Python: Sent actions for {len(actions_list_of_dicts)} envs.")
        except zmq.error.ContextTerminated:
            print("Python: ZMQ context terminated, likely during shutdown.")
        except Exception as e:
            print(f"Python: Error sending actions to Unity: {e}")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # 处理Gymnasium的seed逻辑

        if self._is_first_reset_call:
            print("Python Env (reset): First reset call. Waiting for initial batch from Unity...")
            if not self._receive_unity_batch(): # 等待Unity发送第一个包
                # 如果接收失败，可能需要返回一个默认的“错误”观测值或抛出异常
                print("Python Env (reset): Failed to receive initial batch from Unity.")
                dummy_obs = np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
                return dummy_obs, [{"error": "Failed to receive initial batch"}] * self.num_envs
            
            self._is_first_reset_call = False
            print("Python Env (reset): Initial batch received.")
        else:
            # 对于非首次调用reset，通常意味着外部代码（例如RL agent的episode结束逻辑）希望重置。
            # 在我们的模型中，Unity会在isDone=true后自动重置。
            # Python端的reset()主要是为了获取当前的观测状态（可能是刚重置后的）。
            # 我们需要确保完成一次REQ-REP循环以同步状态。
            # 发送一个“无操作”的动作批次，然后接收Unity的响应。
            # print("Python Env (reset): Subsequent reset call. Sending no-op actions to sync state.")
            noop_actions_to_send = []
            for i in range(self.num_envs):
                 # 尝试从 self._current_unity_batch_data 获取 envId
                env_id_to_use = i 
                if self._current_unity_batch_data and self._current_unity_batch_data.get('envDataList') and i < len(self._current_unity_batch_data['envDataList']):
                    env_id_to_use = self._current_unity_batch_data['envDataList'][i].get('envId', i)
                
                noop_actions_to_send.append({
                    'envId': env_id_to_use,
                    'action': [0.0, 0.0], # 无操作
                    'nextResetProximityFactor': None
                })
            self._send_actions_to_unity(noop_actions_to_send)
            if not self._receive_unity_batch():
                 print("Python Env (reset): Failed to receive batch from Unity after sending no-op.")
                 dummy_obs = np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
                 return dummy_obs, [{"error": "Failed to sync after no-op"}] * self.num_envs


        obs_list = []
        info_list = [{} for _ in range(self.num_envs)] # 创建info列表
        if self._current_unity_batch_data and 'envDataList' in self._current_unity_batch_data:
            for i, env_data in enumerate(self._current_unity_batch_data['envDataList']):
                obs_list.append(flatten_observation(env_data['observation']))
                # 你可以在info中包含原始的结构化观测数据或其他调试信息
                # info_list[i]['raw_observation'] = env_data['observation'] 
        else: # 如果没有收到有效数据
             obs_list = [np.zeros(OBS_FLAT_SIZE, dtype=np.float32) for _ in range(self.num_envs)]


        return np.array(obs_list, dtype=np.float32), info_list


    def step(self, actions_for_all_envs):
        # actions_for_all_envs: numpy数组, shape (self.num_envs, ACTION_SIZE)
        if self._is_first_reset_call:
            # 如果reset()没有被首先调用以完成初始握手，step()不应该继续
            print("Python Env (step): Error - reset() must be called once before the first step to initialize connection.")
            # 返回一个表示错误或无效状态的值
            dummy_obs = np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
            dummy_rewards = np.zeros(self.num_envs, dtype=np.float32)
            dummy_dones = np.ones(self.num_envs, dtype=bool) # 标记为done以可能触发外部重置
            dummy_truncs = np.zeros(self.num_envs, dtype=bool)
            dummy_infos = [{"error": "step() called before initial reset()"}] * self.num_envs
            return dummy_obs, dummy_rewards, dummy_dones, dummy_truncs, dummy_infos

        actions_to_send_list = []
        for i in range(self.num_envs):
            # 从上一步Unity发送的数据中获取isDone状态，以决定是否发送proximityFactor
            # 确保self._current_unity_batch_data 和其内容存在
            is_done_for_current_env = False
            env_id_for_action = i # 默认使用索引作为envId

            if self._current_unity_batch_data and \
               self._current_unity_batch_data.get('envDataList') and \
               i < len(self._current_unity_batch_data['envDataList']):
                
                env_state_data = self._current_unity_batch_data['envDataList'][i]
                is_done_for_current_env = env_state_data.get('isDone', False)
                env_id_for_action = env_state_data.get('envId', i) # 使用Unity提供的envId
            
            action_for_env_np = actions_for_all_envs[i]
            action_for_env_list = action_for_env_np.tolist() # MessagePack通常需要列表

            next_prox_factor = None
            if is_done_for_current_env and np.random.rand() < 0.5: # 50%的几率在done后发送proximityFactor
                next_prox_factor = np.random.rand()
            
            actions_to_send_list.append({
                'envId': env_id_for_action,
                'action': action_for_env_list,
                'nextResetProximityFactor': next_prox_factor
            })
            
        self._send_actions_to_unity(actions_to_send_list)
        
        # 接收这些动作执行后的结果
        if not self._receive_unity_batch():
            print("Python Env (step): Failed to receive batch from Unity after sending actions.")
            # 返回错误状态或上一个状态的副本
            dummy_obs = np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
            dummy_rewards = np.zeros(self.num_envs, dtype=np.float32)
            dummy_dones = np.ones(self.num_envs, dtype=bool) 
            dummy_truncs = np.zeros(self.num_envs, dtype=bool)
            dummy_infos = [{"error": "Failed to receive state after action"}] * self.num_envs
            return dummy_obs, dummy_rewards, dummy_dones, dummy_truncs, dummy_infos
        
        obs_list = []
        reward_list = []
        done_list = []
        # Gymnasium 的 step 返回 terminated 和 truncated 两个布尔数组
        # Unity只发送一个 isDone 信号，我们将其映射到 terminated
        terminated_list = [] 
        truncated_list = [False] * self.num_envs # Unity环境目前不发送truncated信号
        info_list = [{} for _ in range(self.num_envs)]

        for i, env_data in enumerate(self._current_unity_batch_data['envDataList']):
            obs_list.append(flatten_observation(env_data['observation']))
            reward_list.append(env_data.get('reward', 0.0))
            terminated_list.append(env_data.get('isDone', True)) # 如果缺少isDone，则假设已结束
            # info_list[i]['raw_observation'] = env_data['observation'] # 可选：在info中包含原始数据

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list, dtype=bool), # isDone -> terminated
            np.array(truncated_list, dtype=bool),  # No truncation signal from Unity
            info_list
        )

    def render(self):
        # Unity端自行渲染。此方法在VecEnv中通常是可选的。
        pass

    def close(self):
        print("Python: Initiating close sequence...")
        if self.socket:
            print("Python: Closing ZMQ socket.")
            self.socket.close() # 关闭套接字
            self.socket = None
        if self.context:
            print("Python: Terminating ZMQ context.")
            self.context.term() # 终止上下文
            self.context = None
        
        if self.unity_process and self.unity_process.poll() is None: # 检查进程是否仍在运行
            print(f"Python: Attempting to terminate Unity process (PID: {self.unity_process.pid})...")
            try:
                self.unity_process.terminate() # 发送终止信号
                self.unity_process.wait(timeout=10) # 等待进程终止
                if self.unity_process.poll() is None:
                    print("Python: Unity process did not terminate gracefully, killing...")
                    self.unity_process.kill() # 强制终止
                    self.unity_process.wait(timeout=5)
                print("Python: Unity process terminated.")
            except Exception as e:
                print(f"Python: Error during Unity process termination: {e}")
        self.unity_process = None
        print("Python: Cleanup complete.")

# --- 主测试循环 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python test script for Unity ZMQ environment.")
    parser.add_argument("--executable-path", type=str, default=None, 
                        help="Path to Unity executable. If None and not --editor-debug, will fail.")
    parser.add_argument("--port", type=int, default=5555, help="Base port for ZMQ communication.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments Unity should run.")
    parser.add_argument("--unity-upspeed", type=int, default=1, help="Time.timeScale for Unity.")
    parser.add_argument("--unity-qps", type=int, default=10, help="Target QPS for Unity ZmqRLClient.")
    parser.add_argument("--unity-max-steps", type=int, default=1000, help="Max steps per episode in Unity.")
    parser.add_argument("--total-test-steps", type=int, default=2000, help="Total random steps to run in this script.")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID, offsets port if multiple Python scripts run.")
    parser.add_argument("--editor-debug", action="store_true", help="Connect to Unity Editor instead of launching an executable.")
    
    args = parser.parse_args()

    if args.editor_debug:
        print("Python: Editor debug mode selected. Will attempt to connect to Unity Editor.")
        launch_unity_process = False # 不启动exe
        if args.executable_path:
            print("Python: Warning - --executable-path is provided but --editor-debug is active. Executable path will be ignored.")
    elif args.executable_path:
        launch_unity_process = True
    else:
        print("Python: Error - Must provide --executable-path OR use --editor-debug flag.")
        exit(1)

    env = None
    try:
        env = UnityVectorEnv(
            executable_path=args.executable_path,
            port=args.port, # worker_id 会在内部处理端口偏移
            num_envs=args.num_envs,
            unity_upspeed=args.unity_upspeed,
            unity_qps=args.unity_qps,
            unity_max_steps=args.unity_max_steps,
            launch_unity=launch_unity_process,
            worker_id=args.worker_id,
            editor_debug_mode=args.editor_debug
        )

        print("Python: Initializing environment by calling reset()...")
        # 第一次调用 reset() 会等待Unity的初始状态包
        observations, infos = env.reset() 
        print(f"Python: Initial observations received for {len(observations)} environments.")
        # print(f"Python: First observation for env 0 (shape {observations[0].shape}): {observations[0][:5]}...") # 打印前5个元素

        for step_num in range(args.total_test_steps):
            # 为每个环境生成随机动作
            random_actions = env.action_space.sample() # gym的sample()会考虑num_envs
            
            # 执行step
            next_observations, rewards, terminated, truncated, infos = env.step(random_actions)
            
            if (step_num + 1) % 100 == 0: # 每100步打印一次信息
                print(f"\n--- Step: {step_num + 1} ---")
                for i in range(args.num_envs):
                    # 确保索引在obs、rewards等数组的范围内
                    if i < len(next_observations) and i < len(rewards) and i < len(terminated):
                         print(f"  Env {i}: Obs (shape): {next_observations[i].shape}, Reward: {rewards[i]:.3f}, Terminated: {terminated[i]}")
                    else:
                         print(f"  Env {i}: Data missing for this step ( ممکن است تعداد envs با پاسخ不符).")


            # 如果所有环境都完成了（例如在单环境测试中），通常RL循环会调用reset
            # 在向量化环境中，Gymnasium VecEnv 通常会自动重置done了的环境
            # 我们的包装器依赖于Unity在isDone后发送新初始状态，Python端的step方法会返回这个新状态
            # 如果你想在Python端显式重置所有已完成的环境，你需要检查 dones 数组并相应处理
            # 但对于随机测试，我们持续step即可，Unity的isDone逻辑会处理重置后的新观测

        print(f"\nPython: Completed {args.total_test_steps} random steps.")

    except KeyboardInterrupt:
        print("\nPython: Test interrupted by user.")
    except Exception as e:
        print(f"Python: An error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            print("Python: Cleaning up environment...")
            env.close()
        print("Python: Script finished.")
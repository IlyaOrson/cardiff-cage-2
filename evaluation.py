import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random

from tqdm import trange, tqdm
from rich import inspect as rin
from rich.console import Console
from pprint import pformat
from pathlib import Path

MAX_EPS = 1000
agent_name = 'Blue'
random.seed(0)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    commit_hash = "Not using git"
    # ask for a name
    name = "John Hannay"
    # ask for a team
    team = "CardiffUni"
    # ask for a name for the agent
    name_of_agent = "PPO + Greedy decoys"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = MainAgent()

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in tqdm([30, 50, 100], desc="max_steps", leave=False):
        for red_agent in tqdm([B_lineAgent, RedMeanderAgent, SleepAgent], desc="red_agent", leave=False):

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            # wrapped_cyborg.env.env.env.env.env === cyborg
            ec = cyborg.environment_controller

            # ChallengeWrapper2 > OpenAIGymWrapper > EnumActionWrapper > BlueTableWrapper > TrueTableWrapper > CyBORG
            bt = wrapped_cyborg.env.env.env

            # the state is tracked by a State object
            # ec.state
            # the observation is extracted through an Observation object
            # which is constructed by following the entries per agent in the INFO_DICT
            # ec.state.get_true_state(ec.INFO_DICT["True"])
            # ec.state.get_true_state(ec.INFO_DICT["Blue"])

            # extract ip maps (stored in CybORG.environment_controller)
            # ec.subnet_cidr_map  # subnet to ip networks
            # ec.hostname_ip_map  # hostname to ip addresses

            # get last action
            ec.get_last_action("Blue")

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            # NOTE  ec.INFO_DICT["True"/"Blue"] is in theory the reference dict used to filter the state object
            #       and produce an initial observation, although the following observations do not come from here
            # Path("info_dict_true.txt").write_text(pformat(ec.INFO_DICT["True"]))
            # Path("info_dict_blue.txt").write_text(pformat(ec.INFO_DICT["Blue"]))

            # these are invariant nodes attributes
            hosts_state = [host.get_state() for host in ec.state.hosts.values()]

            # Path("obs_prev.txt").write_text(pformat(observation))

            # NOTE this is equivalent to ec.get_last_observation("Blue").data --> ec.observation["Blue"]
            bt_obs = bt.get_observation("Blue")
            bt_tab = bt.get_table()
            # Path("bt_obs_prev.txt").write_text(pformat(bt_obs))
            # Path("bt_tab_prev.txt").write_text(pformat(bt_tab))

            ec_st_prev = ec.state
            ec_st_true_prev = ec.get_true_state(ec.INFO_DICT["True"]).data
            ec_obs_true_prev = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["True"])).data
            # NOTE this is not updated because the observations are updated directly in ec.observation["Blue"] in the ec.step(...) method
            # ec_obs_blue_prev = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["Blue"]), "Blue").data

            # console = Console(record=True, width=120)
            # rin(ec_st_prev, console=console)
            # Path("ec_st_prev.txt").write_text(console.export_text())
            # Path("ec_st_true_prev.txt").write_text(pformat(ec_st_true_prev))
            # Path("ec_obs_true_prev.txt").write_text(pformat(ec_obs_true_prev))

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in trange(MAX_EPS, desc="episodes", leave=False):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in trange(num_steps, desc="num_steps", leave=False):
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)

                    # Path("obs.txt").write_text(pformat(observation))

                    bt_obs = bt.get_observation("Blue")
                    bt_tab = bt.get_table()
                    # Path("bt_obs.txt").write_text(pformat(bt_obs))
                    # Path("bt_tab.txt").write_text(pformat(bt_tab))

                    ec_st = ec.state
                    ec_st_true = ec.get_true_state(ec.INFO_DICT["True"]).data
                    ec_obs_true = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["True"])).data

                    # console = Console(record=True, width=120)
                    # rin(ec_st, console=console)
                    # Path("ec_st.txt").write_text(console.export_text())
                    # Path("ec_st_true.txt").write_text(pformat(ec_st_true))
                    # Path("ec_obs_true.txt").write_text(pformat(ec_obs_true))

                    # difference between steps
                    # if ec_obs_true != ec_obs_true_prev:
                    #     diff_obs_true = DeepDiff(ec_obs_true, ec_obs_true_prev)
                    #     # Path("diff_obs_true.txt").write_text(diff_obs_true.pretty())

                    # update previous files
                    # Path("obs_prev.txt").write_text(pformat(observation))

                    # Path("bt_obs_prev.txt").write_text(pformat(bt_obs))
                    # Path("bt_tab_prev.txt").write_text(pformat(bt_tab))

                    # update per step trackers
                    ec_st_prev = ec_st
                    ec_st_true_prev = ec_st_true
                    ec_obs_true_prev = ec_obs_true

                    # console = Console(record=True, width=120)
                    # rin(ec_st_prev, console=console)
                    # Path("ec_st_prev.txt").write_text(console.export_text())
                    # Path("ec_st_true_prev.txt").write_text(pformat(ec_st_true_prev))
                    # Path("ec_obs_true_prev.txt").write_text(pformat(ec_obs_true_prev))

                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
"""
COMP 3106 A Assignment 3
Names:
  Mohamed Azab    (101331588)
  Yassin Koubeisy (101352113)
"""

import csv
import os
from functools import lru_cache


# ---------- 2.1 Environment ----------

ACTIONS = [1, 2, 3]

# the w part of a state
AGENT_WON = 'A'
OPPONENT_WON = 'O'
NOT_OVER = '-'

# terminal states have no real action, so we use 0 for them
NO_ACTION = 0


# state looks like "cbag/cag/cop/w"
# cached since the same states get parsed over and over during training
@lru_cache(maxsize = None)
def parse_state(state):
  bag, agent, opp, winner = state.strip().split('/')
  return int(bag), int(agent), int(opp), winner


def is_terminal(state):
  return parse_state(state)[3] != NOT_OVER


# can't take more coins than what's left in the bag
def legal_actions(state):
  if is_terminal(state):
    return [NO_ACTION]
  bag = parse_state(state)[0]
  return [a for a in ACTIONS if a <= bag]


# ---------- 2.2 Reward function ----------

def reward(state):
  _, agent, _, winner = parse_state(state)
  if winner == AGENT_WON:
    return agent
  elif winner == OPPONENT_WON:
    return -agent
  return 0


# ---------- helpers ----------

# anything not in the q table yet is still at its initial value Q(s,a) = r(s)
# terminal state value is just its reward
def get_q(q, state, action):
  if is_terminal(state):
    return reward(state)
  return q.get((state, action), reward(state))


def best_q(q, state):
  return max(get_q(q, state, a) for a in legal_actions(state))


# each trial is a list of (state, action), last row is terminal with action None
def read_trials(directory):
  trials = []
  for filename in sorted(os.listdir(directory)):
    if not filename.lower().endswith('.csv'):
      continue

    trial = []
    with open(os.path.join(directory, filename), encoding = 'utf-8-sig') as f:
      for row in csv.reader(f):
        if len(row) == 0 or row[0].strip() == '':
          continue
        state = row[0].strip()

        # skip anything that isn't a state (like a header row)
        try:
          parse_state(state)
        except ValueError:
          continue

        # use the state to tell if the game is over, not the action column
        if is_terminal(state):
          action = None
        else:
          action = int(row[1])
        trial.append((state, action))

    if trial:
      trials.append(trial)
  return trials


# ---------- 2.3 Q-learning class ----------

TOLERANCE = 1e-12     # stop when nothing changes more than this in a full pass
MAX_PASSES = 100000   # just in case it never converges


class td_qlearning:

  def __init__(self, directory, alpha = 0.10, gamma = 0.90):
    # directory is the path to a directory containing trials through state space
    # alpha is the learning rate in td-update
    # gamma is the discount factor in the Bellman equation (and hence in the td-update)
    # Return nothing
    self.alpha = alpha
    self.gamma = gamma

    # q table, key is (state, action)
    self.q = {}

    trials = read_trials(directory)

    # keep going over all the trials until the q values stop changing
    # update: Q(s,a) += alpha * (r(s) + gamma * max Q(s',a') - Q(s,a))
    for _ in range(MAX_PASSES):
      old_q = dict(self.q)

      for trial in trials:
        for j in range(len(trial) - 1):
          state, action = trial[j]
          next_state = trial[j + 1][0]
          current = get_q(self.q, state, action)
          target = reward(state) + self.gamma * best_q(self.q, next_state)
          self.q[(state, action)] = current + self.alpha * (target - current)

      # compare whole table before vs after the pass (not each single update),
      # otherwise it never stops if the same (s,a) goes to different next states
      biggest_change = 0
      for key in self.q:
        old = old_q.get(key, reward(key[0]))
        biggest_change = max(biggest_change, abs(self.q[key] - old))

      if biggest_change < TOLERANCE:
        break

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # Return the q-value for the state-action pair
    q = get_q(self.q, state.strip(), int(action))
    return q

  def policy(self, state):
    # state is a string representation of a state

    # Return the optimal action (as an integer) under the learned policy
    state = state.strip()
    a = None
    best = None
    # go from biggest to smallest so ties go to the bigger action
    for action in reversed(legal_actions(state)):
      value = get_q(self.q, state, action)
      if best is None or value > best:
        best = value
        a = action
    return a

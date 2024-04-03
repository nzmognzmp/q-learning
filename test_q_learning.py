from itertools import product
from typing import Any, cast

import gymnasium as gym

from q_learning import BaseEnvironment, QLearner


class Environment(BaseEnvironment[tuple[int, int], str]):
    def __init__(self) -> None:
        self.allowed = frozenset(
            {
                (0, 1),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 1),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 1),
                (4, 1),
            }
        )

    def reset(self) -> tuple[tuple[int, int], dict[str, Any]]:
        self.current_state = (2, 0)
        return self.current_state, {}

    def step(
        self, action: str
    ) -> tuple[tuple[int, int], float, bool, bool, dict[Any, Any]]:
        r, c = self.current_state
        if action == "↑":
            r -= 1
        elif action == "→":
            c += 1
        elif action == "↓":
            r += 1
        elif action == "←":
            c -= 1
        else:
            raise ValueError("Invalid action")
        if 0 <= r < 5 and 1 <= c < 6 and (r, c) in self.allowed:
            self.current_state = (r, c)
            return (r, c), -1.0, False, False, {}
        elif (r, c) == (2, 6):
            self.current_state = (r, c)
            return (r, c), -1.0, True, False, {}
        else:
            return self.current_state, -1.0, False, False, {}


def test_qlearn() -> None:
    actions = ["↑", "→", "↓", "←"]
    states = [(i, j) for i, j in product(range(5), range(1, 6))] + [(2, 0), (2, 6)]

    environment = Environment()

    q_learner: QLearner[tuple[int, int], str] = QLearner(
        actions=actions, states=states, environment=environment, n_iter=1000
    )
    final_reward = q_learner.learn()
    assert int(final_reward) == -6
    print(q_learner)


def test_frozen() -> None:
    env = cast(BaseEnvironment[int, int], gym.make("FrozenLake-v1"))
    q_learner = QLearner(
        actions=[0, 1, 2, 3], states=list(range(16)), environment=env, n_iter=10000
    )
    final_reward = q_learner.learn()
    print(final_reward, q_learner)


if __name__ == "__main__":
    test_qlearn()
    test_frozen()

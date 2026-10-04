# COMP 3106 A - Assignment 1
# Mohamed Azab - 101331588
# Yassin Koubsiey - 101352113

import csv
import heapq


def heuristic(position, goals):
    """
    Return the Manhattan distance from the current position
    to the closest goal tile.
    """
    row, col = position

    return min(
        abs(row - goal_row) + abs(col - goal_col) for goal_row, goal_col in goals
    )


def treasure_total(mask, treasure_values):
    """
    Return the total value of all treasures represented
    by the given bit mask.
    """
    total = 0

    for index, value in enumerate(treasure_values):
        if mask & (1 << index):
            total += value

    return total


def pathfinding(filepath):
    """
    Use A* search to find the minimum-cost path from S to any G
    while collecting treasures with a total value of at least 5.

    Returns:
        optimal_path: list of (row, column) coordinates
        optimal_path_cost: total movement cost of the path
        num_states_explored: number of states expanded by A*
    """

    # ------------------------------------------------------------
    # 1. Read the CSV grid
    # ------------------------------------------------------------
    grid = []

    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            # Remove any accidental whitespace around CSV entries.
            grid.append([cell.strip() for cell in row])

    rows = len(grid)
    cols = len(grid[0])

    # ------------------------------------------------------------
    # 2. Find the start, goal tiles, and treasures
    # ------------------------------------------------------------
    start = None
    goals = []
    treasures = {}

    # Each treasure receives an index so that the set of collected
    # treasures can be represented efficiently using a bit mask.
    treasure_index = 0

    for row in range(rows):
        for col in range(cols):
            tile = grid[row][col]

            if tile == "S":
                start = (row, col)

            elif tile == "G":
                goals.append((row, col))

            elif tile in {"1", "2", "3", "4", "5"}:
                treasures[(row, col)] = (treasure_index, int(tile))
                treasure_index += 1

    # Store treasure values by their assigned indices.
    treasure_values = [0] * len(treasures)

    for position, (index, value) in treasures.items():
        treasure_values[index] = value

    # ------------------------------------------------------------
    # 3. Create the initial A* state
    #
    # State format:
    #     (row, column, treasure_mask)
    #
    # The treasure mask is necessary because reaching the same tile
    # with different treasures collected represents a different state.
    # ------------------------------------------------------------
    start_mask = 0
    start_state = (start[0], start[1], start_mask)

    # Priority queue entries have the form:
    #     (f_cost, g_cost, state)
    #
    # where:
    #     g_cost = actual cost from the start
    #     h_cost = heuristic estimate to a goal
    #     f_cost = g_cost + h_cost
    frontier = []

    start_h_cost = heuristic(start, goals)
    heapq.heappush(frontier, (start_h_cost, 0, start_state))

    # best_cost[state] stores the cheapest known g-cost for that state.
    best_cost = {start_state: 0}

    # parent[state] stores the previous state on the best known path.
    # It is used to reconstruct the final optimal path.
    parent = {start_state: None}

    num_states_explored = 0
    goal_state = None
    optimal_path_cost = None

    # Possible movements: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------
    # 4. A* search
    # ------------------------------------------------------------
    while frontier:

        # Remove the state with the smallest f-cost.
        f_cost, g_cost, state = heapq.heappop(frontier)

        row, col, mask = state

        # A cheaper route to this same state may have been discovered
        # after this heap entry was inserted. Ignore outdated entries.
        if g_cost != best_cost[state]:
            continue

        # This state is now being explored/expanded.
        num_states_explored += 1

        # A goal tile is only a valid final state if the agent has
        # collected treasures with a total value of at least 5.
        if (row, col) in goals and treasure_total(mask, treasure_values) >= 5:
            goal_state = state
            optimal_path_cost = g_cost
            break

        # Explore all valid adjacent tiles.
        for row_change, col_change in directions:

            new_row = row + row_change
            new_col = col + col_change

            # The new position must remain inside the grid.
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # The agent cannot enter wall tiles.
            if grid[new_row][new_col] == "X":
                continue

            new_mask = mask

            # If the new tile contains treasure, collect it only if
            # this particular treasure has not already been collected.
            if (new_row, new_col) in treasures:
                index, value = treasures[(new_row, new_col)]

                if not (mask & (1 << index)):
                    new_mask = mask | (1 << index)

            new_state = (new_row, new_col, new_mask)

            # Every movement to an adjacent tile costs exactly 1.
            new_g_cost = g_cost + 1

            # Keep this route only if it is the first route to this
            # state or is cheaper than the best route found previously.
            if new_state not in best_cost or new_g_cost < best_cost[new_state]:
                best_cost[new_state] = new_g_cost
                parent[new_state] = state

                new_h_cost = heuristic((new_row, new_col), goals)

                new_f_cost = new_g_cost + new_h_cost

                heapq.heappush(frontier, (new_f_cost, new_g_cost, new_state))

    # The assignment states that a solution exists. This fallback
    # prevents an error if an invalid input with no solution is used.
    if goal_state is None:
        return [], None, num_states_explored

    # ------------------------------------------------------------
    # 5. Reconstruct the optimal path
    # ------------------------------------------------------------
    optimal_path = []

    current_state = goal_state

    # Follow parent links from the goal back to the start.
    while current_state is not None:
        row, col, mask = current_state

        # The required output contains coordinates only, not the
        # internal treasure mask.
        optimal_path.append((row, col))

        current_state = parent[current_state]

    # The path was built from goal -> start, so reverse it.
    optimal_path.reverse()

    # Required return values:
    # 1. optimal path
    # 2. cost of the optimal path
    # 3. number of states explored by A*
    return optimal_path, optimal_path_cost, num_states_explored

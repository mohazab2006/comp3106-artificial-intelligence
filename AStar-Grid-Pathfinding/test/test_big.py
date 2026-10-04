from assignment1 import pathfinding

path, cost, explored = pathfinding("big_grid.txt")

print("Path:", path)
print("Cost:", cost)
print("States explored:", explored)

print("\nExpected cost: 16")
print("Cost correct:", cost == 16)

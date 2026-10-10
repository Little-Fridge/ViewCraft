---
name: maze-solving
description: Find and verify routes through maze images by reconstructing corridor connectivity, tracking junctions and checking every route segment against walls.
---

# Visual Maze Navigation

1. **Locate start and goal.** Read their labels and movement rules. If unlabeled boundary openings are ambiguous, request the intended pair. For a cell maze, number rows downward and columns rightward from `(1,1)` at the top-left. For a freeform maze, name junctions `J1`, `J2`, etc., with original-image coordinates.
2. **Follow corridors to decisions.** Start at the entrance and follow the corridor center until a junction, dead end or goal. At each junction inspect all exits. Use `zoom` where a faint wall could change connectivity; include both sides of the suspected opening. Add an edge only when an uninterrupted passage is visible. Crossing corners do not permit diagonal movement.
3. **Keep a branch record.** Save `J3: entered W; exits N,E,W; tried N(dead); next E` and the current route stack in the action's `note`. Try unexplored exits in a fixed order, N/E/S/W for a grid. On a dead end, mark that branch closed and return to the last junction with an unexplored exit. Reaching a recorded junction closes a loop; do not restart its exploration.
4. **Handle shortest-path requests separately.** For an unweighted maze, expand cells or junctions in breadth-first layers and record each first-discovery parent. Stop when the goal is first discovered and follow parents back to the start. If edges summarize corridors of unequal length, use their lengths as costs; fewest junctions need not mean shortest distance. A code tool may search the visually verified graph when available.
5. **Validate the extracted route.** Walk it again from start to goal on the original image. Check every wall crossing and turn. Remove exploratory backtracking from the delivered path; do not connect distant waypoints with a line that cuts through a wall.

Return grid directions/cells or freeform turning points with the coordinate convention. If a decisive passage remains unreadable, identify it and give only the verified prefix; do not invent an exit.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.

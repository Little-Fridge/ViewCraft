---
name: maze-solving
description: Find and verify routes through maze images by reconstructing corridor connectivity, tracking junctions and checking every route segment against walls.
---

# Visual Maze Navigation

1. **Locate start and goal.** Read their labels and movement rules. If unlabeled boundary openings are ambiguous, request the intended pair. For a cell maze, number rows downward and columns rightward from `(1,1)` at the top-left. For a freeform maze, name junctions `J1`, `J2`, etc., with original-image coordinates.
2. **Follow corridors to decisions.** Start at the entrance and follow the corridor center until a junction, dead end or goal. At each junction inspect all exits. Use `zoom` where a faint wall could change connectivity; include both sides of the suspected opening. Add an edge only when an uninterrupted passage is visible. Crossing corners do not permit diagonal movement.
3. **Keep a branch record.** Save `position=(r,c); route=...; J3: entered W; exits N,E,W; tried N(dead); next E` in the action's `note` before requesting more evidence. Try unexplored exits in a fixed order, N/E/S/W for a grid. Append a move only through a verified opening. At a dead end, mark the branch closed and remove its moves from the route stack back to the last junction with an unexplored exit. Reaching a recorded junction closes a loop; do not restart its exploration. Use `note` markers to identify junctions in the returned image, retaining the full branch record in the top-level `note` string. Resume from that record on the next turn.
4. **Handle shortest-path requests separately.** For an unweighted cell maze, keep a frontier of cells at the same distance and a first-discovery parent for each visited cell in the working record. Expand one frontier layer at a time through verified openings. When the goal is first reached, follow parents back to the start. Compress a corridor into one edge only if its length is recorded; fewest junctions need not mean shortest distance. If the observation budget or unreadable passages prevent completing this search, return a verified route without claiming it is shortest, or identify the unresolved passage if no complete route is established.
5. **Validate the extracted route.** Walk it again from start to goal on the original image. Check every wall crossing and turn. Remove exploratory backtracking from the delivered path; do not connect distant waypoints with a line that cuts through a wall.

Return grid directions/cells or freeform turning points with the coordinate convention. If a decisive passage remains unreadable, identify it and give only the verified prefix; do not invent an exit.

Use ViewCraft's supplied `view`, `zoom`, `compose` and `note` actions for inspection and working records, then finish with `{"action":"answer","answer":"..."}`. This workflow does not require a Python interpreter, shell or external graph solver. Follow the runtime's image-number and coordinate manifest; a normalized marker location is not a grid-cell index.

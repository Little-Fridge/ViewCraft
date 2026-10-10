---
name: maze-solving
description: Find a traversable route through an image of a maze. Use for visual maze
  puzzles with visible entrances, exits and walls.
---

# 迷宫寻路

Locate the specified start and goal and establish corridor boundaries. For cell mazes, define row/column origin and infer connectivity only through visible openings. For freeform mazes, track junctions and corridor segments rather than forcing them into a square grid. A near contact or diagonal touch is not an opening unless the image's rules permit it.

Inspect uncertain junctions with surrounding wall context. Maintain a record of explored branches and dead ends so returning to a junction does not restart the search. Keep any visual notes away from narrow passages. Trace a candidate route from start to goal, checking every transition against the original walls and confirming both endpoints.

Return a direction sequence for an explicit cell grid, or an ordered list of normalized original-image waypoints for a freeform maze. Claim a shortest route only if the required alternative-route comparison has actually been completed. If a critical passage is unreadable, identify that junction rather than silently crossing it.

Finish with the runtime action=answer object. Its answer field is a string containing the requested output.

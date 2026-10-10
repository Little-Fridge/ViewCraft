---
name: maze-solving
description: Find and verify routes through maze images by reconstructing corridor connectivity, tracking junctions and checking every route segment against walls.
---

# Visual Maze Navigation

## Use
Accept a maze image, start and goal, and any movement rules. If entrances are unlabeled, identify the visible boundary openings and state the chosen pair. Ask for clarification only when multiple pairs would produce materially different routes.

## Recover connectivity
For a cell maze, establish the row/column origin, grid dimensions and wall boundaries. Add a transition only through a visible opening. Diagonal contact does not create a passage unless diagonal moves are explicitly allowed. For a freeform maze, represent junctions, endpoints and corridor segments without forcing a square grid.

## Explore with a route ledger
Track each junction's incoming branch, available branches and visited state. Use a queue for a shortest route in an unweighted graph; use costs only when the puzzle defines them. For ordinary path finding, preserve the current branch stack and mark dead ends so returning to a junction does not restart the search. A host with a code tool can perform graph search after visually validated connectivity is available.

## Resolve uncertain passages
Inspect narrow openings and near-touching walls at source resolution with neighboring junctions visible. Keep the original overview available. Draw path notes beside corridor centers without covering wall boundaries. Do not convert an uncertain opening into an edge just to complete the route.

## Validate the candidate
Trace from the specified start through every segment to the specified goal. Check direction changes, each wall crossing, and the final boundary opening against the original image. Verify that consecutive waypoints follow the corridor; a straight line between distant turns must not cut a corner. Claim a shortest route only after the required graph search or alternative-route check.

## Deliver
For a cell grid, give the origin convention and ordered direction sequence or cells. For a freeform maze, give ordered normalized original-image waypoints, including start and goal. If the route cannot be certified, identify the unresolved junction. Distinguish 'no path in the verified graph' from 'the image is too unclear to determine a path'.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.

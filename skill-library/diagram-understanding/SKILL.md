---
name: diagram-understanding
description: Read flowcharts and system diagrams to answer questions about paths,
  dependencies and branches. Use when connectors and layout carry relationships that
  prose alone does not show.
---

# 流程图理解

Identify node labels, boundaries, arrowheads, legends and condition labels. Trace an edge end to end before assigning source and destination. A line crossing another line is not a junction unless the diagram marks or otherwise clearly establishes a connection.

For dense diagrams, inspect a local branch together with its incoming and outgoing connectors, retaining the overview. Distinguish the visual location of a node from its logical order; proximity alone does not establish a dependency. Treat dashed lines, containers and colors according to the supplied legend, not a guessed convention.

Answer the requested path or relationship with named nodes and branch conditions. For loops, specify both the return edge and visible exit condition. Mark unresolved arrow directions or illegible labels rather than inventing a plausible architecture. Only convert to a full edge list or structured graph when requested.

Finish with the runtime action=answer object. Its answer field is a string containing the requested output.

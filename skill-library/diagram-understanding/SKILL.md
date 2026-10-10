---
name: diagram-understanding
description: Interpret flowcharts, architectures and dependency diagrams by tracing connectors, branch conditions and loops into verifiable paths and relationships.
---

# Diagram & Workflow Reasoning

## Use
Accept a diagram and the requested relationship, path or explanation. Determine whether the drawing represents control flow, data flow, containment or another relation from its labels and legend.

## Recover the visual grammar
Identify nodes, ports, groups, arrowheads, edge styles and condition labels. Read line styles and colors from the supplied legend. Spatial proximity does not establish a dependency. A connector crossing another is not a junction without a dot or another explicit connection cue.

## Trace complete edges
Keep the overview while inspecting dense connector regions at source detail. Follow each relevant edge from its source endpoint to its destination, including bends, crossings and off-page references. Retain enough surrounding context to associate a branch label with the correct outgoing edge. Do not assign an arrow's direction from the nearest node alone.

Build a compact adjacency ledger for the portion needed: source, destination, edge type and condition. Keep node identity distinct when several boxes share a label, using group or location to disambiguate. Treat nested containers as scope unless visible edges or a legend establish execution order.

## Reason over the path
Trace the requested scenario through satisfied branch conditions. For loops, identify the return edge, repeated operation and visible exit condition. For parallel paths, distinguish a split from a join and do not assume synchronization unless the notation specifies it. Check reachability or dependencies over the traced edges rather than inventing a conventional architecture.

## Verify and deliver
Recheck ambiguous arrowheads, crossings and the branch that determines the answer. Return the requested path as named nodes with conditions, or a dependency explanation with supporting edges. Supply an edge list or Mermaid only when requested. Mark unreadable labels and uncertain edges explicitly; a plausible completion of an obscured diagram is not an observed relationship.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.

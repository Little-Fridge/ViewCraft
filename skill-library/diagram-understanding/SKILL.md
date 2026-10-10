---
name: diagram-understanding
description: Interpret flowcharts, architectures and dependency diagrams by tracing connectors, branch conditions and loops into verifiable paths and relationships.
---

# Diagram & Workflow Reasoning

1. **Identify the notation.** Read the legend and requested start/end or scenario. Distinguish nodes, containers, ports, arrowheads and branch labels. Assign short IDs to relevant boxes; disambiguate repeated labels by group and position.
2. **Trace one connector completely.** Start at a node's edge or port and follow the line through every bend to its destination. Crop uncertain crossings with enough surrounding line to reconnect both ends. A crossing is a junction only when the notation shows a dot or explicit join. Determine direction from the arrowhead, not box proximity.
3. **Record verified edges.** Save `A -> B | label/condition | confirmed` in the action's `note`. Example shape: `check -> retry | validation failed | confirmed`. Keep uncertain destinations as `?`; do not fill them using what a familiar architecture usually does. For a branch, associate each condition with its outgoing line rather than the nearest text.
4. **Execute the requested scenario over these edges.** Start at the specified node, choose branches whose conditions are supplied and follow only verified connections. If a deciding condition is unknown, return both conditional paths. At a loop, record the repeated operation and the exit condition. Parallel lines do not by themselves prove parallel execution or a synchronization barrier.
5. **Validate the decisive connection.** Reinspect the arrowhead and condition that determine the answer, including off-page connectors. Check whether a surrounding box indicates containment rather than an extra processing step. For a dependency question, distinguish a direct edge from a multi-hop path.

Return a named path with conditions or the requested dependency explanation. List unresolved edges that could change the result. Produce Mermaid or an edge list only when requested, using the verified graph rather than adding plausible missing connections.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.

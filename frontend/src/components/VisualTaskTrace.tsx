import React, { useState, useEffect, useMemo } from 'react';
import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    Node,
    Edge,
    Position,
    MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css'; // Import base styles
import styles from './VisualTaskTrace.module.css';
import { Dialog, DialogPanel } from '@tremor/react';
// Import shared type
import { ScratchpadEntry } from '../types/agentTypes';

// Define the structure of scratchpad entries
interface VisualTaskTraceProps {
    scratchpadData: ScratchpadEntry[];
}

// Function to parse content and determine node type/label
const parseScratchpadContent = (entry: ScratchpadEntry): { type: string, label: string, details: string } => {
    const { role, content } = entry;
    // Explicitly declare type as string
    let type: string = role; // Default type is the role, but can be overridden
    let label = '';
    let details = content;

    if (role === 'assistant') {
        const thoughtMatch = content.match(/Thought: ([\s\S]*?)(?:Action:|$)/);
        const actionMatch = content.match(/Action: ([\s\S]*)/);
        
        if (thoughtMatch && actionMatch) {
            type = 'thought_action';
            label = 'Thought & Action';
            details = `Thought:\n${thoughtMatch[1].trim()}\n\nAction:\n${actionMatch[1].trim()}`;
        } else if (thoughtMatch) {
            type = 'thought';
            label = 'Thought';
            details = thoughtMatch[1].trim();
        } else {
            // Fallback if no thought/action keywords found
            label = 'Assistant Message'; 
        }
    } else if (role === 'system') {
        label = 'System Message';
    } else if (role === 'user') {
        label = 'User Query';
    } else if (role === 'tool') {
        type = 'observation'; // Map 'tool' role to 'observation' type visually
        label = 'Observation (Tool Result)';
        // Potentially parse tool output if it's structured (e.g., JSON)
        try {
            const parsed = JSON.parse(content);
            details = JSON.stringify(parsed, null, 2); // Pretty print if JSON
        } catch (e) {
            details = content; // Keep as raw string if not JSON
        }
    }

    return { type, label, details };
}

const VisualTaskTrace: React.FC<VisualTaskTraceProps> = ({ scratchpadData }) => {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    // Process scratchpad data into nodes and edges
    useEffect(() => {
        if (!scratchpadData) return;

        const initialNodes: Node[] = [];
        const initialEdges: Edge[] = [];
        let yPos = 0;
        const xPos = 100;
        const yIncrement = 100;

        scratchpadData.forEach((entry, index) => {
            const { type, label, details } = parseScratchpadContent(entry);
            const nodeId = `node-${index}`;

            initialNodes.push({
                id: nodeId,
                type: 'default', // Use default node type for now, style with className
                data: { label: label, details: details, type: type }, // Store full details and type
                position: { x: xPos, y: yPos },
                sourcePosition: Position.Bottom,
                targetPosition: Position.Top,
                className: `${styles.flowNode} ${styles[`nodeType_${type}`]}` // Apply CSS module classes
            });

            yPos += yIncrement;

            // Add edge from previous node
            if (index > 0) {
                initialEdges.push({
                    id: `edge-${index - 1}-${index}`,
                    source: `node-${index - 1}`,
                    target: nodeId,
                    markerEnd: {
                        type: MarkerType.ArrowClosed,
                    },
                    style: { strokeWidth: 2 },
                });
            }
        });

        setNodes(initialNodes);
        setEdges(initialEdges);
        console.log('Processed scratchpad into nodes/edges:', { nodes: initialNodes.length, edges: initialEdges.length });

    }, [scratchpadData]);
    
    // Updated node click handler to open dialog
    const onNodeClick = (event: React.MouseEvent, node: Node) => {
        console.log('Node clicked:', node);
        setSelectedNode(node);
        setIsDialogOpen(true);
        // alert(`Node Type: ${node.data.type}\n\nLabel: ${node.data.label}\n\nDetails:\n${node.data.details}`); // Simple alert for now - replaced by dialog
    };

    // Function to close the dialog
    const handleDialogClose = () => {
        setIsDialogOpen(false);
        setSelectedNode(null);
    };

    return (
        <div className={styles.traceContainer}>
            <h4>Task Execution Trace</h4>
             <p className={styles.info}>Visual representation of the agent's thought process. Click a node for details.</p>
            <div className={styles.reactFlowWrapper}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodeClick={onNodeClick}
                    fitView // Automatically fit the view to the nodes
                    attributionPosition="bottom-left"
                >
                    <MiniMap nodeStrokeWidth={3} zoomable pannable />
                    <Controls />
                    <Background color="#aaa" gap={16} />
                </ReactFlow>
            </div>

            {/* Tremor Dialog for Node Details */}
            <Dialog open={isDialogOpen} onClose={handleDialogClose} static={true} className="z-[100]">
                <DialogPanel className="sm:max-w-lg">
                    <h3 className="text-lg font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                         Node Details: {selectedNode?.data?.type?.replace('_', ' ').toUpperCase()}
                    </h3>
                    <p className="mt-2 leading-6 text-tremor-default text-tremor-content dark:text-dark-tremor-content">
                        Content of the selected step.
                    </p>
                    <div className="mt-4 whitespace-pre-wrap text-sm border border-tremor-border rounded p-2 bg-tremor-background-subtle dark:bg-dark-tremor-background-subtle">
                        {selectedNode?.data?.details || 'No details available.'}
                    </div>
                    <button 
                        className="mt-8 w-full inline-flex justify-center rounded-md border border-transparent bg-tremor-brand px-4 py-2 text-sm font-medium text-tremor-brand-inverted shadow-sm hover:bg-tremor-brand-emphasis focus:outline-none focus:ring-2 focus:ring-tremor-brand focus:ring-offset-2 dark:bg-dark-tremor-brand dark:text-dark-tremor-brand-inverted dark:hover:bg-dark-tremor-brand-emphasis dark:focus:ring-dark-tremor-brand dark:focus:ring-offset-dark-tremor-background"
                        onClick={handleDialogClose}
                    >
                        Close
                    </button>
                </DialogPanel>
            </Dialog>
        </div>
    );
};

export default VisualTaskTrace; 
'use client'

import React, { useCallback } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  Connection,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
} from 'reactflow'
import 'reactflow/dist/style.css'

const initialNodes: Node[] = [
  { id: 'start', type: 'input', data: { label: 'Start' }, position: { x: 250, y: 0 } },
  { id: 'loadEnv', data: { label: 'Load Environment Variables' }, position: { x: 250, y: 80 } },
  { id: 'vectorStore', data: { label: 'Create LocalVectorStoreDriver' }, position: { x: 250, y: 160 } },
  { id: 'webLoader', data: { label: 'Load Web Content' }, position: { x: 250, y: 240 } },
  { id: 'upsert', data: { label: 'Upsert Text Artifacts' }, position: { x: 250, y: 320 } },
  { id: 'agent', data: { label: 'Create Agent' }, position: { x: 250, y: 400 } },
  { id: 'ragClient', data: { label: 'RagClient' }, position: { x: 250, y: 480 } },
  { id: 'ragEngine', data: { label: 'RagEngine' }, position: { x: 250, y: 560 } },
  { id: 'retrievalStage', data: { label: 'RetrievalRagStage' }, position: { x: 100, y: 640 } },
  { id: 'vectorModule', data: { label: 'VectorStoreRetrievalRagModule' }, position: { x: 100, y: 720 } },
  { id: 'responseStage', data: { label: 'ResponseRagStage' }, position: { x: 400, y: 640 } },
  { id: 'promptModule', data: { label: 'PromptResponseRagModule' }, position: { x: 400, y: 720 } },
  { id: 'openAiDriver', data: { label: 'OpenAiChatPromptDriver' }, position: { x: 400, y: 800 } },
  { id: 'runAgent', data: { label: 'Run Agent' }, position: { x: 250, y: 880 } },
  { id: 'end', type: 'output', data: { label: 'End' }, position: { x: 250, y: 960 } },
]

const initialEdges: Edge[] = [
  { id: 'e1', source: 'start', target: 'loadEnv' },
  { id: 'e2', source: 'loadEnv', target: 'vectorStore' },
  { id: 'e3', source: 'vectorStore', target: 'webLoader' },
  { id: 'e4', source: 'webLoader', target: 'upsert' },
  { id: 'e5', source: 'upsert', target: 'agent' },
  { id: 'e6', source: 'agent', target: 'ragClient' },
  { id: 'e7', source: 'ragClient', target: 'ragEngine' },
  { id: 'e8', source: 'ragEngine', target: 'retrievalStage' },
  { id: 'e9', source: 'retrievalStage', target: 'vectorModule' },
  { id: 'e10', source: 'ragEngine', target: 'responseStage' },
  { id: 'e11', source: 'responseStage', target: 'promptModule' },
  { id: 'e12', source: 'promptModule', target: 'openAiDriver' },
  { id: 'e13', source: 'agent', target: 'runAgent' },
  { id: 'e14', source: 'runAgent', target: 'end' },
]

export default function GriptapeRAGFlow() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges])

  return (
    <div style={{ width: '100%', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  )
}
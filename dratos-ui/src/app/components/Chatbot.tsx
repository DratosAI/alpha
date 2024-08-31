'use client'

import { useState } from 'react'
import { useChat } from 'ai/react'
import { Message } from 'ai'

export default function ChatBot() {
  const [error, setError] = useState<string | null>(null)

  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: '/api/chat',
    initialMessages: [{ id: '1', role: 'assistant', content: 'How can I help you today?' }],
    onError: (error) => {
      console.error('Chat error:', error)
      setError('An error occurred while processing your request. Please try again.')
    },
    onFinish: () => {
      setError(null)
    },
  })

  return (
    <div className="fixed bottom-4 right-40 w-80 bg-white shadow-lg rounded-lg">
      <div className="h-72 overflow-y-auto p-4">
        {messages.map((m: Message) => (
          <div key={m.id} className={`${m.role === 'user' ? 'text-right' : 'text-left'} mb-2`}>
            <span className={`inline-block p-2 rounded-lg ${m.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
              {m.content}
            </span>
          </div>
        ))}
        {error && (
          <div className="text-red-500 text-center mt-2">{error}</div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <input
          className="w-full border rounded p-2"
          value={input}
          onChange={handleInputChange}
          placeholder="Ask about workflows..."
        />
      </form>
    </div>
  )
}
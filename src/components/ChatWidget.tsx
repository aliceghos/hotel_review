'use client'

import { useState, useRef, useEffect, useCallback, type JSX } from 'react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: string[]
}

// 优化2：引用数据
interface ChunkInfo {
  text: string
  chunk_idx: number
  start_char: number
  end_char: number
}

interface RefData {
  refIndex: number
  commentId: string
  comment: string
  chunks: ChunkInfo[]
}

function generateThreadId(): string {
  return `thread_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

function generateMsgId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
}

function ChatIcon() {
  return (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function NewChatIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
    </svg>
  )
}

function SpinnerIcon() {
  return (
    <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [currentTool, setCurrentTool] = useState<string | null>(null)
  // 优化2：引用数据映射 (refIndex → RefData)
  const [referencesMap, setReferencesMap] = useState<Map<number, RefData>>(new Map())
  const threadIdRef = useRef<string>(generateThreadId())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll (avoid scrollIntoView)
  useEffect(() => {
    if (messagesEndRef.current) {
      const container = messagesContainerRef.current
      if (container) {
        container.scrollTop = container.scrollHeight
      }
    }
  }, [messages])

  // Focus input when panel opens
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus()
    }
  }, [open])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return

    setInput('')
    setMessages(prev => [
      ...prev,
      { id: generateMsgId(), role: 'user', content: text },
      { id: generateMsgId(), role: 'assistant', content: '' },
    ])
    setStreaming(true)
    setCurrentTool(null)

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, thread_id: threadIdRef.current }),
      })

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ error: 'Request failed' }))
        setMessages(prev => {
          const updated = [...prev]
          const lastIdx = updated.length - 1
          if (updated[lastIdx].role === 'assistant') {
            updated[lastIdx] = { ...updated[lastIdx], content: `Error: ${errData.error}` }
          }
          return updated
        })
        setStreaming(false)
        return
      }

      const reader = resp.body?.getReader()
      if (!reader) {
        setStreaming(false)
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const dataStr = line.slice(6)
          try {
            const data = JSON.parse(dataStr)

            if (data.references) {
              // 优化2：接收参考文献数据，建立索引映射
              const map = new Map<number, RefData>()
              for (const ref of data.references as RefData[]) {
                if (ref.refIndex > 0) {
                  map.set(ref.refIndex, ref)
                }
              }
              setReferencesMap(map)
            } else if (data.token) {
              setMessages(prev => {
                const updated = [...prev]
                const lastIdx = updated.length - 1
                if (updated[lastIdx].role === 'assistant') {
                  updated[lastIdx] = {
                    ...updated[lastIdx],
                    content: updated[lastIdx].content + data.token,
                  }
                }
                return updated
              })
            } else if (data.tool_start) {
              setCurrentTool(data.tool_start)
              setMessages(prev => {
                const updated = [...prev]
                const lastIdx = updated.length - 1
                if (updated[lastIdx].role === 'assistant') {
                  const prevCalls = updated[lastIdx].toolCalls || []
                  updated[lastIdx] = {
                    ...updated[lastIdx],
                    toolCalls: [...prevCalls, data.tool_start],
                  }
                }
                return updated
              })
            } else if (data.tool_end) {
              setCurrentTool(null)
            } else if (data.done) {
              // stream complete
            } else if (data.error) {
              setMessages(prev => {
                const updated = [...prev]
                const lastIdx = updated.length - 1
                if (updated[lastIdx] && updated[lastIdx].role === 'assistant') {
                  updated[lastIdx] = {
                    ...updated[lastIdx],
                    content: updated[lastIdx].content + `\n\nError: ${data.error}`,
                  }
                }
                return updated
              })
            }
          } catch {
            // Skip unparseable lines
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev]
        const lastIdx = updated.length - 1
        if (updated[lastIdx] && updated[lastIdx].role === 'assistant') {
          updated[lastIdx] = {
            ...updated[lastIdx],
            content: `Connection error: ${err instanceof Error ? err.message : 'Failed to reach agent'}`,
          }
        }
        return updated
      })
    } finally {
      setStreaming(false)
      setCurrentTool(null)
    }
  }, [input, streaming])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewChat = () => {
    setMessages([])
    threadIdRef.current = generateThreadId()
  }

  // 优化2：解析 [[ref:N]] 和 [[ref:N,M]] 为 JSX
  const renderContent = (content: string) => {
    if (!content) return null
    const parts: JSX.Element[] = []
    const regex = /\[\[ref:([\d,]+)\]\]/g
    let lastIndex = 0
    let match

    while ((match = regex.exec(content)) !== null) {
      // 前面的纯文本
      if (match.index > lastIndex) {
        parts.push(<span key={`t-${lastIndex}`}>{content.slice(lastIndex, match.index)}</span>)
      }
      // 引用标记
      const refIds = match[1].split(',').map(Number).filter(n => !isNaN(n))
      parts.push(
        <CitationBadge key={`ref-${match.index}`} refIds={refIds} refMap={referencesMap} />
      )
      lastIndex = match.index + match[0].length
    }
    // 剩余文本
    if (lastIndex < content.length) {
      parts.push(<span key={`t-${lastIndex}`}>{content.slice(lastIndex)}</span>)
    }
    return parts.length > 0 ? parts : content
  }

  return (
    <>
      {/* Floating Button */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-amber-500 hover:bg-amber-600 text-white rounded-2xl shadow-lg flex items-center justify-center transition-all hover:scale-105 hover:shadow-xl cursor-pointer"
          style={{ boxShadow: '0 4px 14px 0 rgba(200, 164, 92, 0.35)' }}
          title="AI 问答助手"
        >
          <ChatIcon />
        </button>
      )}

      {/* Chat Panel */}
      {open && (
        <div className="fixed bottom-6 right-6 z-50 w-96 h-[600px] max-h-[calc(100vh-3rem)] bg-white rounded-2xl shadow-2xl border border-stone-200 flex flex-col overflow-hidden">
          {/* Header */}
          <div
            className="flex items-center justify-between px-4 py-3 border-b border-stone-100 text-white"
            style={{ background: 'linear-gradient(135deg, #c8a45c 0%, #b8942e 100%)' }}
          >
            <div>
              <h3 className="font-semibold text-sm">AI 酒店问答助手</h3>
              <p className="text-xs text-amber-100">基于真实住客评论</p>
            </div>
            <div className="flex gap-1">
              <button
                onClick={handleNewChat}
                className="p-1.5 hover:bg-white/20 rounded-lg transition-colors cursor-pointer"
                title="新对话"
              >
                <NewChatIcon />
              </button>
              <button
                onClick={() => setOpen(false)}
                className="p-1.5 hover:bg-white/20 rounded-lg transition-colors cursor-pointer"
              >
                <CloseIcon />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div
            ref={messagesContainerRef}
            className="flex-1 overflow-y-auto p-4 space-y-4"
            style={{ background: '#faf8f5' }}
          >
            {messages.length === 0 && (
              <div className="text-center py-8">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-amber-100 text-amber-500 mb-4">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <p className="text-stone-500 text-sm mb-3">有什么关于酒店的问题？</p>
                <div className="flex flex-wrap gap-1.5 justify-center">
                  {['早餐怎么样？', '床品舒服吗？', '隔音好不好？', '适合亲子吗？'].map(q => (
                    <button
                      key={q}
                      onClick={() => { setInput(q); setTimeout(() => inputRef.current?.focus(), 50) }}
                      className="px-3 py-1.5 bg-white border border-stone-200 rounded-full text-xs text-stone-600 hover:border-amber-300 hover:text-amber-700 cursor-pointer transition-all shadow-sm"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} message-enter`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-amber-500 text-white rounded-br-lg'
                      : 'bg-white border border-stone-200 text-stone-700 rounded-bl-lg shadow-sm'
                  }`}
                >
                  {msg.content
                    ? renderContent(msg.content)
                    : (streaming && msg.id === messages[messages.length - 1]?.id ? (
                      <span className="flex items-center gap-1.5 text-stone-400">
                        <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" />
                        <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }} />
                        <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }} />
                      </span>
                    ) : null)
                  }
                  {msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="mt-1.5 pt-1.5 border-t border-stone-100">
                      {msg.toolCalls.map((tool) => (
                        <span key={tool} className="inline-block px-2 py-0.5 bg-amber-50 text-amber-700 rounded-full text-xs mr-1 font-medium">
                          {tool}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Tool indicator */}
            {currentTool && (
              <div className="flex justify-center">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-amber-50 text-amber-700 rounded-full text-xs font-medium border border-amber-200">
                  <SpinnerIcon />
                  正在使用 {currentTool}...
                </span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-stone-100 p-3 bg-white">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入问题..."
                disabled={streaming}
                className="flex-1 px-3.5 py-2 border border-stone-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40 focus:border-amber-400 disabled:bg-stone-50 placeholder:text-stone-400"
              />
              <button
                onClick={handleSend}
                disabled={streaming || !input.trim()}
                className="px-4 py-2 bg-amber-500 text-white rounded-xl hover:bg-amber-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer shadow-sm"
              >
                <SendIcon />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

// 优化2：引用标记组件 — 可悬停查看引用原文
function CitationBadge({ refIds, refMap }: { refIds: number[]; refMap: Map<number, RefData> }) {
  const [hovered, setHovered] = useState(false)

  // 获取悬停显示的文本（优先取 chunk，其次取评论片段）
  const tooltipTexts: string[] = []
  for (const rid of refIds) {
    const ref = refMap.get(rid)
    if (ref) {
      if (ref.chunks.length > 0) {
        tooltipTexts.push(`[评论${rid}] ${ref.chunks[0].text}`)
      } else {
        tooltipTexts.push(`[评论${rid}] ${ref.comment}`)
      }
    }
  }

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 mx-0.5 rounded-md text-xs font-medium cursor-pointer transition-colors"
        style={{
          background: hovered ? '#f59e0b' : '#fef3c7',
          color: hovered ? '#fff' : '#92400e',
          border: '1px solid ' + (hovered ? '#d97706' : '#fcd34d'),
        }}
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
        [{refIds.join(',')}]
      </span>
      {hovered && tooltipTexts.length > 0 && (
        <span className="absolute bottom-full left-0 mb-2 w-72 p-2.5 bg-gray-900 text-white text-xs rounded-lg shadow-xl z-50 leading-relaxed pointer-events-none">
          {tooltipTexts.map((t, i) => (
            <p key={i} className={i > 0 ? 'mt-1.5 pt-1.5 border-t border-gray-700' : ''}>{t}</p>
          ))}
        </span>
      )}
    </span>
  )
}

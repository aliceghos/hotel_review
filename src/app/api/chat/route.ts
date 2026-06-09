const RAG_SERVICE_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

export async function POST(req: Request) {
  try {
    const { message, thread_id } = await req.json()

    if (!message) {
      return Response.json({ error: 'message is required' }, { status: 400 })
    }

    // 构建请求体：message 作为 query，history 可选
    const body: Record<string, unknown> = {
      query: message,
      options: { enable_hyde: false },
    }

    const resp = await fetch(`${RAG_SERVICE_URL}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (!resp.ok) {
      const text = await resp.text()
      return Response.json({ error: `RAG service error: ${text}` }, { status: resp.status })
    }

    // SSE 格式转换：rag-service → 前端 ChatWidget 兼容格式
    const reader = resp.body?.getReader()
    if (!reader) {
      return Response.json({ error: 'No response body' }, { status: 500 })
    }

    const encoder = new TextEncoder()
    const decoder = new TextDecoder()

    const stream = new ReadableStream({
      async start(controller) {
        let buffer = ''
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              try {
                const event = JSON.parse(line.slice(6))

                if (event.type === 'intent' && event.data?.need_retrieval) {
                  // 检索开始时通知前端
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ tool_start: '混合检索' })}\n\n`
                  ))
                } else if (event.type === 'references') {
                  // 检索+排序完成
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ tool_end: '混合检索' })}\n\n`
                  ))
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ tool_start: '智能排序' })}\n\n`
                  ))
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ tool_end: '智能排序' })}\n\n`
                  ))
                  // 优化2：转发参考文献数据供前端引用高亮
                  const refs = event.data?.comments || []
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ references: refs.map((c: Record<string, unknown>) => ({
                      refIndex: c.rank || c.final_rank || 0,
                      commentId: c._id || c.comment_id || '',
                      comment: (c.comment as string || '').slice(0, 200),
                      chunks: c.matched_chunks || [],
                    })) })}\n\n`
                  ))
                } else if (event.type === 'chunk' && event.content) {
                  // 文本 token → 兼容格式
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ token: event.content })}\n\n`
                  ))
                } else if (event.type === 'done') {
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ done: true })}\n\n`
                  ))
                } else if (event.type === 'error') {
                  controller.enqueue(encoder.encode(
                    `data: ${JSON.stringify({ error: event.message })}\n\n`
                  ))
                }
              } catch {
                // 忽略解析失败的行
              }
            }
          }
        } catch (err) {
          controller.enqueue(encoder.encode(
            `data: ${JSON.stringify({ error: String(err) })}\n\n`
          ))
        } finally {
          controller.close()
        }
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    })
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Unknown error'
    return Response.json({ error: errorMessage }, { status: 500 })
  }
}

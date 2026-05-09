# Stress test: agentic coding task with substantial context
# Gives Laguna a realistic task: refactor a NestJS provider with streaming, retries, errors.

$systemPrompt = @"
You are a senior TypeScript / NestJS engineer. You write production code that is type-safe, handles errors explicitly, and follows NestJS idioms (Injectable, DI, configuration via ConfigService). You do not add comments that explain what code does — only why.
"@

$existingCode = @"
// llm/providers/ollama.provider.ts (current naive version)
import { Injectable } from '@nestjs/common';

@Injectable()
export class OllamaProvider {
  private readonly baseUrl = process.env.OLLAMA_URL ?? 'http://localhost:11434';
  private readonly model = 'laguna-xs.2:q4_K_M';

  async chat(messages: Array<{ role: string; content: string }>) {
    const response = await fetch(`\${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        messages,
        stream: false,
        options: { temperature: 0.7, top_k: 20, num_ctx: 32768 },
      }),
    });
    return response.json();
  }
}

// llm/llm.module.ts
import { Module } from '@nestjs/common';
import { OllamaProvider } from './providers/ollama.provider';

@Module({
  providers: [OllamaProvider],
  exports: [OllamaProvider],
})
export class LlmModule {}

// llm/types.ts
export type ChatRole = 'system' | 'user' | 'assistant';
export interface ChatMessage { role: ChatRole; content: string }
export interface ChatChunk { content: string; done: boolean }
export interface ChatResult { content: string; promptTokens: number; completionTokens: number; durationMs: number }
"@

$task = @"
Rewrite OllamaProvider to a production-grade implementation. Requirements:

1. Use ConfigService for baseUrl, model name, default temperature, default num_ctx, request timeout (default 60s) and max retries (default 2).
2. Two public methods:
   - chat(messages, opts?): returns Promise<ChatResult>
   - chatStream(messages, opts?): returns AsyncGenerator<ChatChunk>
   Both accept optional { temperature, topK, numCtx, model, signal? } overrides.
3. Streaming uses Ollama's /api/chat with stream=true (NDJSON over HTTP). Parse line-by-line, yield ChatChunk for each token, set done=true on the final chunk.
4. Retries: only on network errors and 5xx, exponential backoff (250ms, 500ms, 1000ms cap), respect AbortSignal.
5. Errors: throw a typed OllamaError class with cause, status code, and a short message. Distinguish ConnectionError, TimeoutError, ServerError, BadRequestError.
6. Use a single keep-alive Agent (undici Agent) for connection reuse. Inject it via factory provider in the module so tests can swap it.
7. Use Logger from @nestjs/common, never console.log.

Output: full file contents for ollama.provider.ts, llm.module.ts, llm.errors.ts, and any tiny helper file you add. No prose explanation, just code. Keep types strict — no `any`.
"@

$messages = @(
    @{ role = "system"; content = $systemPrompt }
    @{ role = "user"; content = "$existingCode`n`n---`n`n$task" }
)

$payload = [ordered]@{
    model    = "laguna-xs.2:q4_K_M"
    messages = $messages
    stream   = $false
    options  = [ordered]@{
        temperature = 0.2
        top_k       = 20
        num_ctx     = 32768
    }
}
$bodyJson = ConvertTo-Json $payload -Depth 20 -Compress
$tmp = [System.IO.Path]::Combine($env:TEMP, "ollama-req.json")
[System.IO.File]::WriteAllText($tmp, $bodyJson, [System.Text.UTF8Encoding]::new($false))

Write-Host "=== Sending request ==="
$promptChars = ($systemPrompt + $existingCode + $task).Length
Write-Host ("Prompt chars: {0} (rough estimate ~{1} tokens)" -f $promptChars, [math]::Round($promptChars / 3.5))
Write-Host ("Body bytes: {0}" -f (Get-Item $tmp).Length)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri "http://localhost:11434/api/chat" -Method Post -InFile $tmp -ContentType "application/json" -TimeoutSec 600
$sw.Stop()

Write-Host ""
Write-Host "=== METRICS ==="
$promptEvalSec = $r.prompt_eval_duration / 1e9
$evalSec       = $r.eval_duration / 1e9
$totalSec      = $r.total_duration / 1e9
$loadSec       = $r.load_duration / 1e9

"{0,-22}: {1}" -f "prompt tokens",       $r.prompt_eval_count
"{0,-22}: {1}" -f "completion tokens",   $r.eval_count
"{0,-22}: {1:N2} s" -f "load",           $loadSec
"{0,-22}: {1:N2} s" -f "prompt eval",    $promptEvalSec
"{0,-22}: {1:N2} s" -f "generation",     $evalSec
"{0,-22}: {1:N2} s" -f "total wall time",$totalSec
"{0,-22}: {1:N1} tok/s" -f "prompt eval rate", ($r.prompt_eval_count / $promptEvalSec)
"{0,-22}: {1:N1} tok/s" -f "gen rate",         ($r.eval_count / $evalSec)

Write-Host ""
Write-Host "=== RESPONSE (first 4000 chars) ==="
$resp = $r.message.content
if ($resp.Length -gt 4000) {
    Write-Host $resp.Substring(0, 4000)
    Write-Host ""
    Write-Host "...[truncated, total $($resp.Length) chars]"
} else {
    Write-Host $resp
}

# Save full response to file for inspection
$resp | Out-File -FilePath "C:\Users\Diamond\Desktop\slovo-llm\stress-test-output.md" -Encoding utf8
Write-Host ""
Write-Host "Full response saved to stress-test-output.md"

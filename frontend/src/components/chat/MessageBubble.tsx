import { Sparkles, User } from "lucide-react";
import { CitationCard } from "./CitationCard";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Citation {
  title: string;
  type?: string;
  filePath?: string;
}

interface MessageBubbleProps {
  content: string;
  isUser: boolean;
  citations?: Citation[];
  timestamp?: string;
  isStreaming?: boolean;
}

export function MessageBubble({ content, isUser, citations, timestamp, isStreaming }: MessageBubbleProps) {
  return (
    <div className={cn(
      "flex gap-3 animate-fade-in",
      isUser ? "flex-row-reverse" : "flex-row"
    )}>
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center shadow-soft",
        isUser 
          ? "chat-gradient" 
          : "bg-card border border-border"
      )}>
        {isUser ? (
          <User className="w-4 h-4 text-chat-user-foreground" />
        ) : (
          <Sparkles className={cn("w-4 h-4 text-primary", isStreaming && "animate-pulse")} />
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex flex-col gap-2 max-w-[80%]",
        isUser ? "items-end" : "items-start"
      )}>
        {/* Streaming indicator */}
        {isStreaming && !isUser && (
          <span className="text-xs text-primary animate-pulse">En train d'écrire...</span>
        )}
        
        <div className={cn(
          "px-4 py-3 rounded-2xl shadow-soft overflow-x-auto",
          isUser 
            ? "bg-chat-user text-chat-user-foreground rounded-br-md" 
            : "bg-card text-chat-ai-foreground rounded-bl-md border border-border"
        )}>
          {isUser ? (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
          ) : (
            <div className="markdown-content">
              {content ? (
                <>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      // Headings
                      h1: ({ children }) => <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>,
                      h2: ({ children }) => <h2 className="text-lg font-bold mt-3 mb-2">{children}</h2>,
                      h3: ({ children }) => <h3 className="text-base font-semibold mt-3 mb-1">{children}</h3>,
                      // Paragraphs
                      p: ({ children }) => <p className="my-2 leading-relaxed">{children}</p>,
                      // Strong/Bold
                      strong: ({ children }) => <strong className="font-bold">{children}</strong>,
                      // Emphasis/Italic
                      em: ({ children }) => <em className="italic">{children}</em>,
                      // Lists
                      ul: ({ children }) => <ul className="list-disc list-inside my-2 space-y-1">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal list-inside my-2 space-y-1">{children}</ol>,
                      li: ({ children }) => <li className="ml-2">{children}</li>,
                      // Links - handle file downloads vs external links
                      a: ({ href, children }) => {
                        const isFileDownload = href?.startsWith('/files/');
                        const isExternal = href?.startsWith('http://') || href?.startsWith('https://');
                        
                        if (isFileDownload) {
                          // File download link - open in new tab for PDF preview or trigger download
                          return (
                            <a 
                              href={href} 
                              className="text-primary underline hover:text-primary/80 inline-flex items-center gap-1"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              📄 {children}
                            </a>
                          );
                        }
                        
                        return (
                          <a 
                            href={href} 
                            className="text-primary underline hover:text-primary/80" 
                            target={isExternal ? "_blank" : undefined}
                            rel={isExternal ? "noopener noreferrer" : undefined}
                          >
                            {children}
                          </a>
                        );
                      },
                      // Code
                      code: ({ className, children }) => {
                        const isInline = !className;
                        return isInline ? (
                          <code className="bg-secondary px-1.5 py-0.5 rounded text-sm font-mono">{children}</code>
                        ) : (
                          <code className="block bg-secondary p-3 rounded-lg text-sm font-mono overflow-x-auto my-2">{children}</code>
                        );
                      },
                      // Pre (code blocks)
                      pre: ({ children }) => <pre className="bg-secondary p-3 rounded-lg overflow-x-auto my-2">{children}</pre>,
                      // Blockquote
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-4 border-primary pl-4 my-3 italic text-muted-foreground">{children}</blockquote>
                      ),
                      // Tables
                      table: ({ children }) => (
                        <div className="overflow-x-auto my-3">
                          <table className="min-w-full border-collapse border border-border text-sm">{children}</table>
                        </div>
                      ),
                      thead: ({ children }) => <thead className="bg-secondary">{children}</thead>,
                      tbody: ({ children }) => <tbody>{children}</tbody>,
                      tr: ({ children }) => <tr className="border-b border-border hover:bg-secondary/50">{children}</tr>,
                      th: ({ children }) => <th className="border border-border px-3 py-2 text-left font-semibold">{children}</th>,
                      td: ({ children }) => <td className="border border-border px-3 py-2">{children}</td>,
                      // Horizontal rule
                      hr: () => <hr className="my-4 border-border" />,
                    }}
                  >
                    {content}
                  </ReactMarkdown>
                  {isStreaming && <span className="inline-block w-1.5 h-4 ml-0.5 bg-primary animate-pulse rounded-sm" />}
                </>
              ) : (
                <div className="flex gap-1.5 py-1">
                  <div className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Citations */}
        {citations && citations.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-1">
            {citations.map((citation, index) => (
              <CitationCard key={index} title={citation.title} type={citation.type} filePath={citation.filePath} />
            ))}
          </div>
        )}

        {/* Timestamp */}
        {timestamp && (
          <span className="text-xs text-muted-foreground px-1">{timestamp}</span>
        )}
      </div>
    </div>
  );
}

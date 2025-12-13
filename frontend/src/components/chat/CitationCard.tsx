import { FileText, Download } from "lucide-react";

interface CitationCardProps {
  title: string;
  type?: string;
  filePath?: string;
}

export function CitationCard({ title, type = "PDF", filePath }: CitationCardProps) {
  // Build the download URL
  const downloadUrl = filePath ? `/files/${encodeURIComponent(filePath).replace(/%2F/g, '/')}` : undefined;
  
  if (downloadUrl) {
    return (
      <a 
        href={downloadUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-2 px-3 py-1.5 bg-citation border border-citation-border rounded-md text-sm font-medium text-primary hover:bg-primary/5 transition-colors group cursor-pointer"
      >
        <FileText className="w-3.5 h-3.5 text-primary" />
        <span className="truncate max-w-[160px]">{title}</span>
        <Download className="w-3 h-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
      </a>
    );
  }
  
  return (
    <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-citation border border-citation-border rounded-md text-sm font-medium text-muted-foreground">
      <FileText className="w-3.5 h-3.5" />
      <span className="truncate max-w-[160px]">{title}</span>
    </span>
  );
}

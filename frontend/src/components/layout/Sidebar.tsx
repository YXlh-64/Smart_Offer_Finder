import { Plus, MessageSquare, ChevronDown, User, Settings, LogOut, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Session } from "@/lib/api";

interface SidebarProps {
  sessions: Session[];
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  activeSessionId?: string;
  isLoading?: boolean;
}

// Helper function to group sessions by date
function groupSessionsByDate(sessions: Session[]): { label: string; sessions: Session[] }[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const lastMonth = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  const groups: { label: string; sessions: Session[] }[] = [
    { label: "Aujourd'hui", sessions: [] },
    { label: "Hier", sessions: [] },
    { label: "7 derniers jours", sessions: [] },
    { label: "30 derniers jours", sessions: [] },
    { label: "Plus ancien", sessions: [] },
  ];

  sessions.forEach((session) => {
    const sessionDate = new Date(session.updated_at);
    
    if (sessionDate >= today) {
      groups[0].sessions.push(session);
    } else if (sessionDate >= yesterday) {
      groups[1].sessions.push(session);
    } else if (sessionDate >= lastWeek) {
      groups[2].sessions.push(session);
    } else if (sessionDate >= lastMonth) {
      groups[3].sessions.push(session);
    } else {
      groups[4].sessions.push(session);
    }
  });

  // Filter out empty groups
  return groups.filter((group) => group.sessions.length > 0);
}

export function Sidebar({ 
  sessions,
  onNewChat, 
  onSelectSession,
  onDeleteSession,
  activeSessionId,
  isLoading
}: SidebarProps) {
  const groupedSessions = groupSessionsByDate(sessions);

  return (
    <aside className="w-72 h-screen flex flex-col sidebar-gradient text-sidebar-foreground">
      {/* Logo */}
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl chat-gradient flex items-center justify-center shadow-soft">
            <span className="text-lg font-bold text-primary-foreground">F</span>
          </div>
          <div>
            <h1 className="font-semibold text-sidebar-accent-foreground">FORSA</h1>
            <p className="text-xs text-sidebar-foreground/60">Assistant IA</p>
          </div>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <Button 
          onClick={onNewChat}
          className="w-full justify-start gap-2 bg-sidebar-accent hover:bg-sidebar-accent/80 text-sidebar-accent-foreground border border-sidebar-border rounded-xl h-11"
          variant="ghost"
        >
          <Plus className="w-4 h-4" />
          Nouvelle conversation
        </Button>
      </div>

      {/* Conversation History */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-6 h-6 border-2 border-sidebar-foreground/30 border-t-sidebar-foreground rounded-full animate-spin" />
          </div>
        ) : groupedSessions.length === 0 ? (
          <div className="text-center py-8 text-sidebar-foreground/50 text-sm">
            Aucune conversation
          </div>
        ) : (
          groupedSessions.map((group) => (
            <div key={group.label} className="mb-4">
              <h3 className="text-xs font-medium text-sidebar-foreground/50 uppercase tracking-wider px-2 mb-2">
                {group.label}
              </h3>
              <div className="space-y-1">
                {group.sessions.map((session) => (
                  <div
                    key={session.id}
                    className={cn(
                      "group w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm text-left transition-colors",
                      activeSessionId === session.id
                        ? "bg-sidebar-primary/20 text-sidebar-primary"
                        : "text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                    )}
                  >
                    <button
                      onClick={() => onSelectSession(session.id)}
                      className="flex-1 flex items-center gap-2 min-w-0"
                    >
                      <MessageSquare className="w-4 h-4 flex-shrink-0 opacity-60" />
                      <span className="truncate">{session.title}</span>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-sidebar-border rounded transition-opacity"
                      title="Supprimer"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-sidebar-foreground/60 hover:text-red-400" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* User Profile */}
      <div className="p-3 border-t border-sidebar-border">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-sidebar-accent transition-colors">
              <div className="w-9 h-9 rounded-full bg-sidebar-accent flex items-center justify-center">
                <User className="w-4 h-4 text-sidebar-accent-foreground" />
              </div>
              <div className="flex-1 text-left">
                <p className="text-sm font-medium text-sidebar-accent-foreground">Utilisateur</p>
                <p className="text-xs text-sidebar-foreground/60">Agent</p>
              </div>
              <ChevronDown className="w-4 h-4 text-sidebar-foreground/50" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuItem>
              <User className="w-4 h-4 mr-2" />
              Mon profil
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Settings className="w-4 h-4 mr-2" />
              Paramètres
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-destructive">
              <LogOut className="w-4 h-4 mr-2" />
              Déconnexion
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </aside>
  );
}

import { Filter, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

interface FilterOption {
  label: string;
  value: string;
}

interface FilterConfig {
  label: string;
  icon?: React.ReactNode;
  options: FilterOption[];
  value: string;
  onChange: (value: string) => void;
}

interface HeaderProps {
  filters: FilterConfig[];
}

export function Header({ filters }: HeaderProps) {
  return (
    <header className="bg-card border-b border-border px-6 py-3">
      <div className="flex items-center justify-end gap-4">
        {/* Filters */}
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground mr-2">Filtres:</span>
          
          {filters.map((filter) => (
            <DropdownMenu key={filter.label}>
              <DropdownMenuTrigger asChild>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="h-9 gap-1.5 rounded-lg border-border bg-card hover:bg-secondary"
                >
                  {filter.label}
                  <ChevronDown className="w-3.5 h-3.5 opacity-50" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {filter.options.map((option) => (
                  <DropdownMenuItem
                    key={option.value}
                    onClick={() => filter.onChange(option.value)}
                    className={cn(
                      filter.value === option.value && "bg-secondary"
                    )}
                  >
                    {option.label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ))}
        </div>
      </div>
    </header>
  );
}

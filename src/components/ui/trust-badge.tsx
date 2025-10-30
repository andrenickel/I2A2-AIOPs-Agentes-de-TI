import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TrustBadgeProps {
  icon: LucideIcon;
  text: string;
  className?: string;
}

export function TrustBadge({ icon: Icon, text, className }: TrustBadgeProps) {
  return (
    <div className={cn(
      "flex items-center gap-2 px-4 py-2 bg-card/80 backdrop-blur-sm rounded-lg border border-border/50",
      className
    )}>
      <Icon className="w-5 h-5 text-primary" />
      <span className="text-sm font-medium text-foreground">{text}</span>
    </div>
  );
}

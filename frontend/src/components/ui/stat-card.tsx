import { Card } from '@/components/ui/card';
import { LucideIcon } from 'lucide-react';
import { AnimatedCounter } from './animated-counter';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: number;
  suffix?: string;
  prefix?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  className?: string;
}

export function StatCard({ 
  title, 
  value, 
  suffix, 
  prefix,
  icon: Icon, 
  trend,
  className 
}: StatCardProps) {
  return (
    <Card className={cn(
      "bg-card/95 backdrop-blur-sm p-6 border-border/50 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 group",
      className
    )}>
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-semibold text-foreground">{title}</h4>
        <div className="p-2 bg-primary/10 rounded-lg group-hover:bg-primary/20 transition-colors">
          <Icon className="w-6 h-6 text-primary" />
        </div>
      </div>
      
      <div className="text-3xl font-bold text-foreground mb-2">
        <AnimatedCounter value={value} suffix={suffix} prefix={prefix} />
      </div>
      
      {trend && (
        <p className="text-sm text-muted-foreground flex items-center gap-1">
          <span className={trend.isPositive ? 'text-success' : 'text-destructive'}>
            {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
          </span>
          <span>vs. mês anterior</span>
        </p>
      )}
    </Card>
  );
}

import { memo } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { LucideIcon, AlertTriangle, Info, CheckCircle, TrendingUp } from 'lucide-react';

interface IntelligentAlertProps {
  type: 'warning' | 'info' | 'success' | 'trend';
  title: string;
  message: string;
  action?: () => void;
  actionLabel?: string;
}

const iconMap: Record<string, LucideIcon> = {
  warning: AlertTriangle,
  info: Info,
  success: CheckCircle,
  trend: TrendingUp,
};

const variantMap: Record<string, 'default' | 'destructive'> = {
  warning: 'destructive',
  info: 'default',
  success: 'default',
  trend: 'default',
};

export const IntelligentAlert = memo(({ 
  type, 
  title, 
  message, 
  action, 
  actionLabel 
}: IntelligentAlertProps) => {
  const Icon = iconMap[type];
  
  return (
    <Alert variant={variantMap[type]} className="animate-fade-in">
      <Icon className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="flex items-center justify-between">
        <span>{message}</span>
        {action && actionLabel && (
          <button 
            onClick={action}
            className="text-sm font-medium underline hover:no-underline ml-4"
          >
            {actionLabel}
          </button>
        )}
      </AlertDescription>
    </Alert>
  );
});

IntelligentAlert.displayName = 'IntelligentAlert';

import { memo } from 'react';
import { Card } from '@/components/ui/card';
import { FileText, Upload, AlertTriangle, CheckCircle } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';

const activities = [
  {
    id: 1,
    type: 'upload',
    title: 'Novo documento processado',
    description: 'NF-e 12345 - Fornecedor ABC',
    time: 'Há 5 minutos',
    icon: Upload,
    color: 'text-success'
  },
  {
    id: 2,
    type: 'validation',
    title: 'Validação concluída',
    description: '15 documentos validados com sucesso',
    time: 'Há 15 minutos',
    icon: CheckCircle,
    color: 'text-success'
  },
  {
    id: 3,
    type: 'alert',
    title: 'Atenção necessária',
    description: 'Inconsistência detectada em 2 documentos',
    time: 'Há 1 hora',
    icon: AlertTriangle,
    color: 'text-warning'
  },
  {
    id: 4,
    type: 'document',
    title: 'Relatório gerado',
    description: 'Relatório mensal de impostos',
    time: 'Há 2 horas',
    icon: FileText,
    color: 'text-primary'
  },
];

export const RecentActivity = memo(() => {
  return (
    <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50">
      <h3 className="text-xl font-semibold mb-4">Atividade Recente</h3>
      <ScrollArea className="h-[300px]">
        <div className="space-y-4">
          {activities.map((activity) => {
            const Icon = activity.icon;
            return (
              <div 
                key={activity.id}
                className="flex items-start gap-4 p-3 rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className={`p-2 rounded-lg bg-muted ${activity.color}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm">{activity.title}</p>
                  <p className="text-sm text-muted-foreground truncate">
                    {activity.description}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {activity.time}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </Card>
  );
});

RecentActivity.displayName = 'RecentActivity';

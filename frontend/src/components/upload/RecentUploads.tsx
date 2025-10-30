import { memo } from 'react';
import { Card } from '@/components/ui/card';
import { FileText, Clock } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface RecentUpload {
  id: string;
  name: string;
  timestamp: Date;
  status: 'success' | 'error';
}

const mockRecentUploads: RecentUpload[] = [
  { id: '1', name: 'NFe_12345.xml', timestamp: new Date(Date.now() - 1000 * 60 * 5), status: 'success' },
  { id: '2', name: 'NFe_67890.xml', timestamp: new Date(Date.now() - 1000 * 60 * 15), status: 'success' },
  { id: '3', name: 'NFe_11111.xml', timestamp: new Date(Date.now() - 1000 * 60 * 30), status: 'error' },
];

export const RecentUploads = memo(() => {
  const formatTimeAgo = (date: Date) => {
    const minutes = Math.floor((Date.now() - date.getTime()) / 60000);
    if (minutes < 1) return 'Agora mesmo';
    if (minutes === 1) return 'Há 1 minuto';
    if (minutes < 60) return `Há ${minutes} minutos`;
    const hours = Math.floor(minutes / 60);
    if (hours === 1) return 'Há 1 hora';
    return `Há ${hours} horas`;
  };

  return (
    <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Uploads Recentes</h3>
        <Clock className="w-5 h-5 text-muted-foreground" />
      </div>
      
      <div className="space-y-3">
        {mockRecentUploads.map((upload) => (
          <div
            key={upload.id}
            className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <FileText className="w-4 h-4 text-primary flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{upload.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatTimeAgo(upload.timestamp)}
                </p>
              </div>
            </div>
            <Badge variant={upload.status === 'success' ? 'default' : 'destructive'}>
              {upload.status === 'success' ? 'Sucesso' : 'Erro'}
            </Badge>
          </div>
        ))}
      </div>
    </Card>
  );
});

RecentUploads.displayName = 'RecentUploads';

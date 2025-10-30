import { memo } from 'react';
import { Card } from '@/components/ui/card';
import { FileText, CheckCircle, AlertTriangle, Clock } from 'lucide-react';

interface UploadStatsProps {
  total: number;
  success: number;
  error: number;
  processing: number;
}

export const UploadStats = memo(({ total, success, error, processing }: UploadStatsProps) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <FileText className="w-5 h-5 text-primary" />
          </div>
          <div>
            <p className="text-2xl font-bold">{total}</p>
            <p className="text-xs text-muted-foreground">Total</p>
          </div>
        </div>
      </Card>

      <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-success/10 rounded-lg">
            <CheckCircle className="w-5 h-5 text-success" />
          </div>
          <div>
            <p className="text-2xl font-bold text-success">{success}</p>
            <p className="text-xs text-muted-foreground">Sucesso</p>
          </div>
        </div>
      </Card>

      <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-warning/10 rounded-lg">
            <Clock className="w-5 h-5 text-warning" />
          </div>
          <div>
            <p className="text-2xl font-bold text-warning">{processing}</p>
            <p className="text-xs text-muted-foreground">Processando</p>
          </div>
        </div>
      </Card>

      <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-destructive/10 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-destructive" />
          </div>
          <div>
            <p className="text-2xl font-bold text-destructive">{error}</p>
            <p className="text-xs text-muted-foreground">Erros</p>
          </div>
        </div>
      </Card>
    </div>
  );
});

UploadStats.displayName = 'UploadStats';

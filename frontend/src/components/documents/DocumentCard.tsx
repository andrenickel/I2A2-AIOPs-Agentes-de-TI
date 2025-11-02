import { memo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { FileText, Download, Eye, Trash2, MessageSquare, MoreVertical } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Checkbox } from '@/components/ui/checkbox';

export interface Document {
  id: string;
  name: string;
  company: string;
  cnpj: string;
  value: number;
  status: 'success' | 'warning' | 'error';
  date: Date;
  tags?: string[];
  items?: number;
  icms?: number;
}

interface DocumentCardProps {
  document: Document;
  isSelected?: boolean;
  onSelect?: (id: string) => void;
  onView?: (document: Document) => void;
  onDownload?: (document: Document) => void;
  onDelete?: (document: Document) => void;
  onAskAI?: (document: Document) => void;
}

export const DocumentCard = memo(({
  document,
  isSelected,
  onSelect,
  onView,
  onDownload,
  onDelete,
  onAskAI
}: DocumentCardProps) => {
  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'success':
        return 'default';
      case 'warning':
        return 'secondary';
      case 'error':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'success':
        return 'Processado';
      case 'warning':
        return 'Pendente';
      case 'error':
        return 'Erro';
      default:
        return status;
    }
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    }).format(date);
  };

  return (
    <Card className="p-4 border-border/50 bg-background/50 hover:shadow-lg transition-all hover:-translate-y-1 group">
      <div className="flex items-start gap-4">
        {onSelect && (
          <Checkbox
            checked={isSelected}
            onCheckedChange={() => onSelect(document.id)}
            className="mt-1"
          />
        )}

        <div className="p-2 bg-primary/10 rounded-lg">
          <FileText className="w-6 h-6 text-primary" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="flex-1 min-w-0">
              <h4 className="font-semibold text-foreground truncate mb-1">
                {document.name}
              </h4>
              <p className="text-sm text-muted-foreground truncate">
                {document.company}
              </p>
            </div>
            <Badge variant={getStatusVariant(document.status)}>
              {getStatusLabel(document.status)}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm mb-3">
            <div>
              <span className="text-muted-foreground">CNPJ: </span>
              <span className="font-medium">{document.cnpj}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Data: </span>
              <span className="font-medium">{formatDate(document.date)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Valor: </span>
              <span className="font-medium text-success">
                R$ {document.value.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}
              </span>
            </div>
            {document.items && (
              <div>
                <span className="text-muted-foreground">Itens: </span>
                <span className="font-medium">{document.items}</span>
              </div>
            )}
          </div>

          {document.tags && document.tags.length > 0 && (
            <div className="flex gap-1 mb-3 flex-wrap">
              {document.tags.map((tag, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onView?.(document)}
              className="flex-1"
            >
              <Eye className="w-4 h-4 mr-2" />
              Ver
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAskAI?.(document)}
            >
              <MessageSquare className="w-4 h-4" />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="z-50 bg-popover">
                <DropdownMenuItem onClick={() => onDownload?.(document)}>
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => onDelete?.(document)}
                  className="text-destructive"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Excluir
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </Card>
  );
});

DocumentCard.displayName = 'DocumentCard';

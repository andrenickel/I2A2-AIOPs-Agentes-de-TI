import { memo, useCallback, useState, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  LayoutGrid,
  List,
  Download,
  Trash2,
  FileText,
  TrendingUp,
  DollarSign,
  AlertTriangle,
} from 'lucide-react';
import { DocumentFilters } from './DocumentFilters';
import { DocumentCard, Document } from './DocumentCard';
import { DocumentDetailModal } from './DocumentDetailModal';
import { toast } from '@/components/ui/use-toast';
import { useNavigate } from 'react-router-dom';
import { useDocuments } from '@/hooks/useDocuments';
import { Skeleton } from '@/components/ui/skeleton';

const DocumentsPage = memo(() => {
  const { data: documents, isLoading } = useDocuments();
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({});
  const [valueRange, setValueRange] = useState({ min: '', max: '' });
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [detailModal, setDetailModal] = useState<{ open: boolean; document: Document | null }>({
    open: false,
    document: null,
  });
  
  const navigate = useNavigate();

  // Filter documents
  const filteredDocuments = useMemo(() => {
    if (!documents) return [];
    return documents.filter((doc) => {
      // Search filter
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        if (
          !doc.name.toLowerCase().includes(search) &&
          !doc.company.toLowerCase().includes(search) &&
          !doc.cnpj.includes(search)
        ) {
          return false;
        }
      }

      // Status filter
      if (statusFilter !== 'all' && doc.status !== statusFilter) {
        return false;
      }

      // Date range filter
      if (dateRange.from && doc.date < dateRange.from) {
        return false;
      }
      if (dateRange.to && doc.date > dateRange.to) {
        return false;
      }

      // Value range filter
      if (valueRange.min && doc.value < parseFloat(valueRange.min)) {
        return false;
      }
      if (valueRange.max && doc.value > parseFloat(valueRange.max)) {
        return false;
      }

      return true;
    });
  }, [documents, searchTerm, statusFilter, dateRange, valueRange]);

  // Statistics
  const stats = useMemo(() => {
    const total = filteredDocuments.length;
    const totalValue = filteredDocuments.reduce((sum, doc) => sum + doc.value, 0);
    const errors = filteredDocuments.filter((doc) => doc.status === 'error').length;
    const avgValue = total > 0 ? totalValue / total : 0;

    return {
      total,
      totalValue,
      errors,
      avgValue,
    };
  }, [filteredDocuments]);

  const handleClearFilters = useCallback(() => {
    setSearchTerm('');
    setStatusFilter('all');
    setDateRange({});
    setValueRange({ min: '', max: '' });
  }, []);

  const handleSelectDocument = useCallback((id: string) => {
    setSelectedDocuments((prev) =>
      prev.includes(id) ? prev.filter((docId) => docId !== id) : [...prev, id]
    );
  }, []);

  const handleSelectAll = useCallback(() => {
    if (selectedDocuments.length === filteredDocuments.length) {
      setSelectedDocuments([]);
    } else {
      setSelectedDocuments(filteredDocuments.map((doc) => doc.id));
    }
  }, [selectedDocuments.length, filteredDocuments]);

  const handleBulkDelete = useCallback(() => {
    toast({
      title: `${selectedDocuments.length} documentos excluídos`,
      description: 'Os documentos foram removidos com sucesso.',
    });
    setSelectedDocuments([]);
  }, [selectedDocuments.length]);

  const handleBulkExport = useCallback(() => {
    toast({
      title: 'Exportando documentos',
      description: `${selectedDocuments.length} documentos serão exportados.`,
    });
  }, [selectedDocuments.length]);

  const handleViewDocument = useCallback((document: Document) => {
    setDetailModal({ open: true, document });
  }, []);

  const handleAskAI = useCallback((document: Document) => {
    navigate('/chat', { state: { document } });
  }, [navigate]);

  const handleDownloadDocument = useCallback((document: Document) => {
    toast({
      title: 'Download iniciado',
      description: `Baixando ${document.name}`,
    });
  }, []);

  const handleDeleteDocument = useCallback((document: Document) => {
    toast({
      title: 'Documento excluído',
      description: `${document.name} foi removido.`,
      variant: 'destructive',
    });
  }, []);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-32" />
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Skeleton key={i} className="h-64" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-white">Documentos Fiscais</h2>
        
        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === 'grid' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            <LayoutGrid className="w-4 h-4" />
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            <List className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <FileText className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.total}</p>
              <p className="text-xs text-muted-foreground">Documentos</p>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-success/10 rounded-lg">
              <DollarSign className="w-5 h-5 text-success" />
            </div>
            <div>
              <p className="text-2xl font-bold text-success">
                R$ {(stats.totalValue / 1000).toFixed(1)}k
              </p>
              <p className="text-xs text-muted-foreground">Valor Total</p>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-secondary/10 rounded-lg">
              <TrendingUp className="w-5 h-5 text-secondary" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                R$ {(stats.avgValue / 1000).toFixed(1)}k
              </p>
              <p className="text-xs text-muted-foreground">Valor Médio</p>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-card/95 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-destructive/10 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-destructive" />
            </div>
            <div>
              <p className="text-2xl font-bold text-destructive">{stats.errors}</p>
              <p className="text-xs text-muted-foreground">Com Erros</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50">
        <DocumentFilters
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          statusFilter={statusFilter}
          onStatusChange={setStatusFilter}
          dateRange={dateRange}
          onDateRangeChange={setDateRange}
          valueRange={valueRange}
          onValueRangeChange={setValueRange}
          onClearFilters={handleClearFilters}
        />
      </Card>

      {/* Bulk Actions */}
      {selectedDocuments.length > 0 && (
        <Card className="bg-card/95 backdrop-blur-sm p-4 border-border/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="secondary">{selectedDocuments.length} selecionados</Badge>
              <Button variant="outline" size="sm" onClick={handleSelectAll}>
                {selectedDocuments.length === filteredDocuments.length ? 'Desmarcar' : 'Selecionar'} Todos
              </Button>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={handleBulkExport}>
                <Download className="w-4 h-4 mr-2" />
                Exportar
              </Button>
              <Button variant="destructive" size="sm" onClick={handleBulkDelete}>
                <Trash2 className="w-4 h-4 mr-2" />
                Excluir
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Documents Grid/List */}
      <div className={viewMode === 'grid' ? 'grid md:grid-cols-2 lg:grid-cols-3 gap-4' : 'space-y-4'}>
        {filteredDocuments.map((doc) => (
          <DocumentCard
            key={doc.id}
            document={doc}
            isSelected={selectedDocuments.includes(doc.id)}
            onSelect={handleSelectDocument}
            onView={handleViewDocument}
            onDownload={handleDownloadDocument}
            onDelete={handleDeleteDocument}
            onAskAI={handleAskAI}
          />
        ))}
      </div>

      {filteredDocuments.length === 0 && (
        <Card className="bg-card/95 backdrop-blur-sm p-12 text-center border-border/50">
          <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Nenhum documento encontrado</h3>
          <p className="text-muted-foreground mb-4">
            Tente ajustar os filtros ou faça upload de novos documentos
          </p>
          <Button onClick={handleClearFilters} variant="outline">
            Limpar Filtros
          </Button>
        </Card>
      )}

      {/* Detail Modal */}
      <DocumentDetailModal
        open={detailModal.open}
        onOpenChange={(open) => setDetailModal({ open, document: null })}
        document={detailModal.document}
      />
    </div>
  );
});

DocumentsPage.displayName = 'DocumentsPage';

export default DocumentsPage;

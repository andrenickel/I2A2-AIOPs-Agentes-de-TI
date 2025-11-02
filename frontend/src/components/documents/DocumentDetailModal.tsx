import { memo, useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Download, FileText, MessageSquare, Loader2 } from 'lucide-react';
import { Document } from './DocumentCard';
import { toast } from '@/components/ui/use-toast';
import { useAIAnalysis } from '@/hooks/useAIAnalysis';

const mockXMLContent = `<?xml version="1.0" encoding="UTF-8"?>
<NFe xmlns="http://www.portalfiscal.inf.br/nfe">
  <infNFe versao="4.00">
    <ide>
      <cUF>35</cUF>
      <cNF>12345678</cNF>
      <natOp>VENDA</natOp>
      <mod>55</mod>
    </ide>
  </infNFe>
</NFe>`;

const mockHistory = [
  { date: new Date(2024, 0, 15), action: 'Documento criado', user: 'Sistema' },
  { date: new Date(2024, 0, 15), action: 'Processamento iniciado', user: 'IA' },
  { date: new Date(2024, 0, 15), action: 'Validação concluída', user: 'IA' },
];

interface DocumentDetailModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  document: Document | null;
}

export const DocumentDetailModal = memo(({ open, onOpenChange, document: doc }: DocumentDetailModalProps) => {
  const [note, setNote] = useState('');
  const [notes, setNotes] = useState<{ content: string; date: Date }[]>([]);
  const { data: aiAnalysis, isLoading: isLoadingAnalysis } = useAIAnalysis(doc?.id || null);

  if (!doc) return null;

  const handleAddNote = () => {
    if (note.trim()) {
      setNotes([...notes, { content: note.trim(), date: new Date() }]);
      setNote('');
      toast({
        title: 'Nota adicionada',
        description: 'A nota foi salva com sucesso.',
      });
    }
  };

  const handleDownloadXML = () => {
    const blob = new Blob([mockXMLContent], { type: 'text/xml' });
    const url = URL.createObjectURL(blob);
    const a = window.document.createElement('a');
    a.href = url;
    a.download = doc.name;
    window.document.body.appendChild(a);
    a.click();
    window.document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: 'Download iniciado',
      description: `Arquivo ${doc.name} baixado.`,
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            {doc.name}
          </DialogTitle>
          <DialogDescription>
            {doc.company} - CNPJ: {doc.cnpj}
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="info" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="info">Informações</TabsTrigger>
            <TabsTrigger value="xml">XML</TabsTrigger>
            <TabsTrigger value="notes">Notas</TabsTrigger>
            <TabsTrigger value="history">Histórico</TabsTrigger>
            <TabsTrigger value="ai-analysis">Análise IA</TabsTrigger>
          </TabsList>

          <ScrollArea className="flex-1 mt-4">
            <TabsContent value="info" className="space-y-4 m-0">
              <Card className="p-4">
                <h3 className="font-semibold mb-3">Dados Principais</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Valor Total</p>
                    <p className="text-lg font-semibold text-success">
                      R$ {doc.value.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Status</p>
                    <Badge className="mt-1">
                      {doc.status === 'success' ? 'Processado' : doc.status === 'warning' ? 'Pendente' : 'Erro'}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Data de Emissão</p>
                    <p className="font-medium">
                      {new Intl.DateTimeFormat('pt-BR').format(doc.date)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Número de Itens</p>
                    <p className="font-medium">{doc.items || 0}</p>
                  </div>
                </div>
              </Card>

              <Card className="p-4">
                <h3 className="font-semibold mb-3">Impostos</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">ICMS</p>
                    <p className="font-medium">
                      R$ {(doc.icms || 0).toLocaleString('pt-BR', { minimumFractionDigits: 2 })}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">IPI</p>
                    <p className="font-medium">R$ 0,00</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">PIS</p>
                    <p className="font-medium">R$ 0,00</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">COFINS</p>
                    <p className="font-medium">R$ 0,00</p>
                  </div>
                </div>
              </Card>

              {doc.tags && doc.tags.length > 0 && (
                <Card className="p-4">
                  <h3 className="font-semibold mb-3">Tags</h3>
                  <div className="flex gap-2 flex-wrap">
                    {doc.tags.map((tag, index) => (
                      <Badge key={index} variant="outline">{tag}</Badge>
                    ))}
                  </div>
                </Card>
              )}

              <Button onClick={handleDownloadXML} className="w-full">
                <Download className="w-4 h-4 mr-2" />
                Download XML
              </Button>
            </TabsContent>

            <TabsContent value="xml" className="m-0">
              <Card className="p-4">
                <pre className="text-xs overflow-auto">
                  <code className="language-xml">{mockXMLContent}</code>
                </pre>
              </Card>
            </TabsContent>

            <TabsContent value="notes" className="space-y-4 m-0">
              <Card className="p-4">
                <h3 className="font-semibold mb-3">Adicionar Nota</h3>
                <Textarea
                  placeholder="Digite sua nota aqui..."
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  className="mb-3"
                  rows={3}
                />
                <Button onClick={handleAddNote} disabled={!note.trim()}>
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Adicionar Nota
                </Button>
              </Card>

              {notes.length > 0 && (
                <Card className="p-4">
                  <h3 className="font-semibold mb-3">Notas ({notes.length})</h3>
                  <div className="space-y-3">
                    {notes.map((n, index) => (
                      <div key={index} className="p-3 bg-muted rounded-lg">
                        <p className="text-sm mb-1">{n.content}</p>
                        <p className="text-xs text-muted-foreground">
                          {new Intl.DateTimeFormat('pt-BR', {
                            dateStyle: 'short',
                            timeStyle: 'short',
                          }).format(n.date)}
                        </p>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="history" className="m-0">
              <Card className="p-4">
                <h3 className="font-semibold mb-3">Histórico de Alterações</h3>
                <div className="space-y-3">
                  {mockHistory.map((item, index) => (
                    <div key={index}>
                      <div className="flex items-start gap-3">
                        <div className="w-2 h-2 rounded-full bg-primary mt-2" />
                        <div className="flex-1">
                          <p className="font-medium text-sm">{item.action}</p>
                          <p className="text-xs text-muted-foreground">
                            {item.user} · {new Intl.DateTimeFormat('pt-BR', {
                              dateStyle: 'short',
                              timeStyle: 'short',
                            }).format(item.date)}
                          </p>
                        </div>
                      </div>
                      {index < mockHistory.length - 1 && (
                        <Separator className="my-3 ml-1" orientation="vertical" />
                      )}
                    </div>
                  ))}
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="ai-analysis" className="m-0">
              <ScrollArea className="h-[500px] pr-4">
                {isLoadingAnalysis ? (
                  <div className="flex items-center justify-center h-[400px]">
                    <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {aiAnalysis?.resumo && (
                      <Card className="p-4">
                        <h3 className="font-semibold mb-2">Resumo da Análise</h3>
                        <p className="text-sm text-muted-foreground">
                          {aiAnalysis.resumo}
                        </p>
                      </Card>
                    )}

                    {aiAnalysis?.problemas && aiAnalysis.problemas.length > 0 && (
                      <Card className="p-4">
                        <h3 className="font-semibold mb-3 flex items-center gap-2">
                          <Badge variant="destructive">{aiAnalysis.problemas.length}</Badge>
                          Problemas Identificados
                        </h3>
                        <div className="space-y-3">
                          {aiAnalysis.problemas.map((problema, index) => (
                            <div key={index} className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                              <p className="text-sm font-medium mb-1">{problema.titulo}</p>
                              <p className="text-xs text-muted-foreground">{problema.descricao}</p>
                              {problema.diferenca && (
                                <div className="mt-2 text-xs">
                                  <span className="text-muted-foreground">Diferença: </span>
                                  <span className="font-mono font-semibold">{problema.diferenca}</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </Card>
                    )}

                    {aiAnalysis?.riscos && aiAnalysis.riscos.length > 0 && (
                      <Card className="p-4">
                        <h3 className="font-semibold mb-3 flex items-center gap-2">
                          <Badge variant="outline" className="bg-orange-500/10 text-orange-500 border-orange-500/20">
                            {aiAnalysis.riscos.length}
                          </Badge>
                          Riscos Identificados
                        </h3>
                        <div className="space-y-3">
                          {aiAnalysis.riscos.map((risco, index) => (
                            <div key={index} className="p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                              <p className="text-sm font-medium mb-1">{risco.titulo}</p>
                              <p className="text-xs text-muted-foreground">{risco.descricao}</p>
                            </div>
                          ))}
                        </div>
                      </Card>
                    )}

                    {aiAnalysis?.agressores && aiAnalysis.agressores.length > 0 && (
                      <Card className="p-4">
                        <h3 className="font-semibold mb-3">Maiores Agressores</h3>
                        <div className="space-y-2">
                          {aiAnalysis.agressores.map((agressor, index) => (
                            <div key={index}>
                              <div className="flex items-start gap-2">
                                <Badge variant="secondary" className="mt-0.5">{agressor.categoria}</Badge>
                                <p className="text-xs text-muted-foreground flex-1">{agressor.descricao}</p>
                              </div>
                              {index < aiAnalysis.agressores.length - 1 && <Separator className="my-2" />}
                            </div>
                          ))}
                        </div>
                      </Card>
                    )}

                    {aiAnalysis?.recomendacoes && aiAnalysis.recomendacoes.length > 0 && (
                      <Card className="p-4">
                        <h3 className="font-semibold mb-3 flex items-center gap-2">
                          <Badge variant="outline" className="bg-blue-500/10 text-blue-500 border-blue-500/20">
                            {aiAnalysis.recomendacoes.length}
                          </Badge>
                          Recomendações
                        </h3>
                        <div className="space-y-3">
                          {aiAnalysis.recomendacoes.map((recomendacao, index) => (
                            <div key={index} className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                              <p className="text-sm font-medium mb-1">{recomendacao.titulo}</p>
                              <p className="text-xs text-muted-foreground">{recomendacao.descricao}</p>
                            </div>
                          ))}
                        </div>
                      </Card>
                    )}

                    <Button className="w-full">
                      Gerar Relatório Completo
                    </Button>
                  </div>
                )}
              </ScrollArea>
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
});

DocumentDetailModal.displayName = 'DocumentDetailModal';

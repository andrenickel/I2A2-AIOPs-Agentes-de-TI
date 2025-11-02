import { memo } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Copy, Download } from 'lucide-react';
import { toast } from '@/components/ui/use-toast';

interface XMLPreviewModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  fileName: string;
  content: string;
}

export const XMLPreviewModal = memo(({ open, onOpenChange, fileName, content }: XMLPreviewModalProps) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    toast({
      title: 'Copiado!',
      description: 'Conteúdo XML copiado para a área de transferência.',
    });
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: 'Download iniciado',
      description: `Arquivo ${fileName} baixado.`,
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Preview XML</DialogTitle>
          <DialogDescription>{fileName}</DialogDescription>
        </DialogHeader>
        
        <div className="flex gap-2 mb-4">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            <Copy className="w-4 h-4 mr-2" />
            Copiar
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

        <ScrollArea className="h-[500px] w-full rounded-md border">
          <pre className="p-4 text-xs">
            <code className="language-xml">{content}</code>
          </pre>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
});

XMLPreviewModal.displayName = 'XMLPreviewModal';

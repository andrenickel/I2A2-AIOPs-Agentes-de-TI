import { memo, useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Link as LinkIcon } from 'lucide-react';
import { urlUploadSchema } from '@/lib/uploadValidation';
import { toast } from '@/components/ui/use-toast';

interface URLUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUpload: (url: string) => void;
}

export const URLUploadModal = memo(({ open, onOpenChange, onUpload }: URLUploadModalProps) => {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const validatedData = urlUploadSchema.parse({ url });
      setIsLoading(true);
      
      // Simulate upload
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      onUpload(validatedData.url);
      toast({
        title: 'Upload iniciado',
        description: 'O arquivo está sendo baixado e processado.',
      });
      
      setUrl('');
      onOpenChange(false);
    } catch (error: any) {
      toast({
        title: 'Erro na validação',
        description: error.errors?.[0]?.message || 'URL inválida',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <LinkIcon className="w-5 h-5" />
            Upload via URL
          </DialogTitle>
          <DialogDescription>
            Cole a URL do arquivo XML que deseja processar
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="url">URL do arquivo</Label>
            <Input
              id="url"
              type="url"
              placeholder="https://exemplo.com/documento.xml"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={isLoading}
            />
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isLoading}
            >
              Cancelar
            </Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? 'Baixando...' : 'Fazer Upload'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
});

URLUploadModal.displayName = 'URLUploadModal';

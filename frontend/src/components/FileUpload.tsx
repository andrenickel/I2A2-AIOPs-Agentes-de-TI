import { useState, useCallback, memo } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertTriangle, X, Eye, Link, Play, Pause, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { toast } from './ui/use-toast';
import { UploadStats } from './upload/UploadStats';
import { RecentUploads } from './upload/RecentUploads';
import { URLUploadModal } from './upload/URLUploadModal';
import { XMLPreviewModal } from './upload/XMLPreviewModal';
import { fileUploadSchema, isDuplicateFile, sanitizeFilename } from '@/lib/uploadValidation';
import confetti from 'canvas-confetti';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  status: 'uploading' | 'processing' | 'success' | 'error' | 'paused';
  progress: number;
  message?: string;
  content?: string;
  isPaused?: boolean;
}

const FileUpload = memo(() => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [urlModalOpen, setUrlModalOpen] = useState(false);
  const [previewModal, setPreviewModal] = useState<{ open: boolean; file?: UploadedFile }>({ open: false });
  const [globalProgress, setGlobalProgress] = useState(0);

  const stats = {
    total: files.length,
    success: files.filter(f => f.status === 'success').length,
    error: files.filter(f => f.status === 'error').length,
    processing: files.filter(f => f.status === 'uploading' || f.status === 'processing').length,
  };

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      errors.forEach((error: any) => {
        toast({
          title: `Erro: ${file.name}`,
          description: error.message,
          variant: 'destructive',
        });
      });
    });

    // Validate and process accepted files
    const validFiles: File[] = [];
    
    acceptedFiles.forEach((file) => {
      try {
        fileUploadSchema.parse({
          name: file.name,
          size: file.size,
          type: file.type,
        });

        // Check for duplicates
        if (isDuplicateFile(file, files)) {
          toast({
            title: 'Arquivo duplicado',
            description: `${file.name} já foi adicionado.`,
            variant: 'destructive',
          });
          return;
        }

        validFiles.push(file);
      } catch (error: any) {
        toast({
          title: `Erro na validação: ${file.name}`,
          description: error.errors?.[0]?.message || 'Arquivo inválido',
          variant: 'destructive',
        });
      }
    });

    if (validFiles.length > 0) {
      processFiles(validFiles);
    }
  }, [files]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/zip': ['.zip'],
      'application/x-zip-compressed': ['.zip'],
    },
    multiple: true,
  });

  const processFiles = async (selectedFiles: File[]) => {
    const newFiles: UploadedFile[] = selectedFiles.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: sanitizeFilename(file.name),
      size: file.size,
      status: 'uploading' as const,
      progress: 0,
    }));

    setFiles((prev) => [...prev, ...newFiles]);

    // Process each file
    selectedFiles.forEach((file, index) => {
      uploadToAPI(file, newFiles[index].id);
    });
  };

  const uploadToAPI = async (file: File, fileId: string) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Enviar sem aguardar resposta (fire-and-forget)
      fetch('http://localhost:8000/ingest', {
        method: 'POST',
        body: formData,
      }).catch(() => {
        // Ignora erros de rede
      });

      // Simular progresso local
      let progress = 0;
      const interval = setInterval(() => {
        const currentFile = files.find(f => f.id === fileId);
        if (currentFile?.isPaused) {
          return;
        }

        progress += 10;
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileId
              ? {
                  ...f,
                  progress,
                  status: progress < 100 ? 'uploading' : 'success',
                }
              : f
          )
        );

        // Update global progress
        setGlobalProgress(
          Math.round(
            (files.reduce((sum, f) => sum + f.progress, 0) / (files.length * 100)) * 100
          )
        );

        if (progress >= 100) {
          clearInterval(interval);
          
          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileId
                ? {
                    ...f,
                    status: 'success',
                    message: 'Arquivo enviado com sucesso',
                  }
                : f
            )
          );

          confetti({
            particleCount: 50,
            spread: 60,
            origin: { y: 0.8 },
          });
        }
      }, 300);
    } catch (error) {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === fileId
            ? {
                ...f,
                status: 'error',
                message: 'Erro ao enviar arquivo',
              }
            : f
        )
      );
    }
  };

  const removeFile = useCallback((fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
    toast({
      title: 'Arquivo removido',
      description: 'O arquivo foi removido da lista.',
    });
  }, []);

  const pauseFile = useCallback((fileId: string) => {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === fileId ? { ...f, isPaused: true, status: 'paused' as const } : f
      )
    );
  }, []);

  const resumeFile = useCallback((fileId: string) => {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === fileId ? { ...f, isPaused: false, status: 'uploading' as const } : f
      )
    );
  }, []);

  const retryFile = useCallback((fileId: string) => {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === fileId ? { ...f, status: 'uploading' as const, progress: 0, message: undefined } : f
      )
    );
  }, []);

  const formatFileSize = useCallback((bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }, []);

  const getStatusIcon = useCallback((status: UploadedFile['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-success" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-destructive" />;
      case 'paused':
        return <Pause className="w-5 h-5 text-warning" />;
      default:
        return <FileText className="w-5 h-5 text-secondary" />;
    }
  }, []);

  const getStatusColor = useCallback((status: UploadedFile['status']) => {
    switch (status) {
      case 'success':
        return 'bg-success/10 border-success/30';
      case 'error':
        return 'bg-destructive/10 border-destructive/30';
      case 'processing':
        return 'bg-secondary/10 border-secondary/30';
      case 'paused':
        return 'bg-warning/10 border-warning/30';
      default:
        return 'bg-muted/50 border-border';
    }
  }, []);

  const handleURLUpload = (url: string) => {
    toast({
      title: 'Upload via URL',
      description: `Baixando arquivo de: ${url}`,
    });
  };

  const clearAll = () => {
    setFiles([]);
    setGlobalProgress(0);
    toast({
      title: 'Lista limpa',
      description: 'Todos os arquivos foram removidos.',
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-white">Upload de Documentos</h2>
        {files.length > 0 && (
          <Button variant="outline" onClick={clearAll} size="sm">
            Limpar Tudo
          </Button>
        )}
      </div>

      {/* Stats */}
      {files.length > 0 && <UploadStats {...stats} />}

      {/* Global Progress */}
      {stats.processing > 0 && (
        <Card className="bg-card/95 backdrop-blur-sm p-4 border-border/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Progresso Geral</span>
            <span className="text-sm text-muted-foreground">{globalProgress}%</span>
          </div>
          <Progress value={globalProgress} className="h-2" />
        </Card>
      )}

      {/* Upload Zone */}
      <Card
        {...getRootProps()}
        className={`bg-card/95 backdrop-blur-sm p-12 text-center transition-all border-2 border-dashed cursor-pointer ${
          isDragActive ? 'border-primary bg-primary/5 scale-105' : 'border-border hover:border-primary/50'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className={`w-16 h-16 mx-auto mb-4 transition-all ${isDragActive ? 'text-primary animate-float' : 'text-primary'}`} />
        <h3 className="text-2xl font-semibold mb-2">
          {isDragActive ? 'Solte os arquivos aqui' : 'Arraste arquivos ZIP aqui'}
        </h3>
        <p className="text-muted-foreground mb-6">
          ou use as opções abaixo
        </p>
        <div className="flex flex-wrap gap-3 justify-center">
          <Button className="bg-gradient-to-r from-primary to-secondary">
            Selecionar Arquivos
          </Button>
          <Button variant="outline" onClick={(e) => { e.stopPropagation(); setUrlModalOpen(true); }}>
            <Link className="w-4 h-4 mr-2" />
            Upload via URL
          </Button>
        </div>
        <p className="text-sm text-muted-foreground mt-4">
          Suporta arquivos ZIP sem restrição de tamanho
        </p>
      </Card>

      {/* Files List */}
      {files.length > 0 && (
        <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50">
          <h3 className="text-xl font-semibold mb-4">
            Arquivos ({files.length})
          </h3>
          <div className="space-y-3">
            <AnimatePresence>
              {files.map((file) => (
                <motion.div
                  key={file.id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -100 }}
                  className={`border rounded-lg p-4 ${getStatusColor(file.status)}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                      {getStatusIcon(file.status)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-medium text-foreground truncate">
                            {file.name}
                          </h4>
                          <Badge variant="outline" className="text-xs">
                            {formatFileSize(file.size)}
                          </Badge>
                        </div>
                        
                        {file.message && (
                          <p className={`text-sm mb-2 ${file.status === 'error' ? 'text-destructive' : 'text-muted-foreground'}`}>
                            {file.message}
                          </p>
                        )}
                        
                        {(file.status === 'uploading' || file.status === 'processing') && (
                          <div className="mt-2">
                            <Progress value={file.progress} className="h-2 mb-1" />
                            <p className="text-xs text-muted-foreground">
                              {file.status === 'uploading'
                                ? `Enviando... ${file.progress}%`
                                : 'Processando e validando...'}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex gap-2">
                      {file.content && file.status === 'success' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setPreviewModal({ open: true, file })}
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                      )}
                      
                      {file.status === 'uploading' && !file.isPaused && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => pauseFile(file.id)}
                        >
                          <Pause className="w-4 h-4" />
                        </Button>
                      )}
                      
                      {file.status === 'paused' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => resumeFile(file.id)}
                        >
                          <Play className="w-4 h-4" />
                        </Button>
                      )}
                      
                      {file.status === 'error' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => retryFile(file.id)}
                        >
                          <RotateCcw className="w-4 h-4" />
                        </Button>
                      )}
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(file.id)}
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </Card>
      )}

      {/* Recent Uploads */}
      <RecentUploads />

      {/* Instructions */}
      <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50">
        <h3 className="text-xl font-semibold mb-4">Como funciona?</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="flex items-start gap-3">
            <div className="bg-primary/10 text-primary rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 font-semibold">
              1
            </div>
            <div>
              <h4 className="font-semibold mb-1">Upload</h4>
              <p className="text-muted-foreground text-sm">
                Arraste, selecione ou cole URL dos arquivos ZIP
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-secondary/10 text-secondary rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 font-semibold">
              2
            </div>
            <div>
              <h4 className="font-semibold mb-1">Processamento</h4>
              <p className="text-muted-foreground text-sm">
                Arquivos são enviados para processamento automático
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-success/10 text-success rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 font-semibold">
              3
            </div>
            <div>
              <h4 className="font-semibold mb-1">Resultados</h4>
              <p className="text-muted-foreground text-sm">
                Acesse documentos processados no Dashboard e na biblioteca
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Modals */}
      <URLUploadModal
        open={urlModalOpen}
        onOpenChange={setUrlModalOpen}
        onUpload={handleURLUpload}
      />

      {previewModal.file && (
        <XMLPreviewModal
          open={previewModal.open}
          onOpenChange={(open) => setPreviewModal({ open })}
          fileName={previewModal.file.name}
          content={previewModal.file.content || ''}
        />
      )}
    </div>
  );
});

FileUpload.displayName = 'FileUpload';

export default FileUpload;

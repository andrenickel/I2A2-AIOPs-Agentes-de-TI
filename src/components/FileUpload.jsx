import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, CheckCircle, AlertTriangle, XCircle, Eye, Download } from 'lucide-react';

const FileUpload = () => {
  const [files, setFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [processing, setProcessing] = useState(false);

  // Mock processed files data
  const [processedFiles, setProcessedFiles] = useState([]);

  useEffect(() => {
    const fetchNotas = async () => {
      try {
        const response = await fetch('https://i2a2-aiops-agentes-de-ti-n8n.bysger.easypanel.host/webhook/b6021f43-0ee5-4fdc-9d2b-ae06f45d1888');
        const data = await response.json();

        // Opcional: mapear campos para manter compatibilidade visual
        const notasFormatadas = data.map((nfe, index) => ({
          id: nfe.chave_acesso || index,
          name: `NFe ${nfe.numero}/${nfe.serie}`,
          status: 'success', // ou inferir com base nos dados
          emitente: nfe.razao_social_emitente,
          valor: `R$ ${Number(nfe.valor_nota_fiscal).toLocaleString('pt-BR', { minimumFractionDigits: 2 })}`,
          data: new Date(nfe.data_emissao).toLocaleDateString('pt-BR'),
          validations: [
            `Modelo: ${nfe.modelo}`,
            `Natureza: ${nfe.natureza_operacao}`,
            `Destino: ${nfe.destino_operacao}`
          ]
        }));

        setProcessedFiles(notasFormatadas);
      } catch (err) {
        console.error('Erro ao buscar NFes:', err);
      }
    };

    fetchNotas();
  }, []);


  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  // const handleFiles = (fileList) => {
  //   const newFiles = Array.from(fileList).map((file, index) => ({
  //     id: Date.now() + index,
  //     file,
  //     name: file.name,
  //     size: file.size,
  //     status: 'uploading'
  //   }));

  //   setFiles(prev => [...prev, ...newFiles]);

  //   // Simulate processing
  //   setProcessing(true);
  //   setTimeout(() => {
  //     setFiles(prev => prev.map(f => 
  //       newFiles.find(nf => nf.id === f.id) 
  //         ? { ...f, status: 'processing' }
  //         : f
  //     ));

  //     setTimeout(() => {
  //       setFiles(prev => prev.map(f => 
  //         newFiles.find(nf => nf.id === f.id) 
  //           ? { ...f, status: 'completed' }
  //           : f
  //       ));
  //       setProcessing(false);

  //       // Add to processed files
  //       const mockProcessed = newFiles.map((f, index) => ({
  //         id: Date.now() + index + 1000,
  //         name: f.name,
  //         status: ['success', 'warning', 'error'][index % 3],
  //         emitente: `Empresa ${String.fromCharCode(65 + index)} Ltda`,
  //         valor: `R$ ${(Math.random() * 50000 + 1000).toLocaleString('pt-BR', { minimumFractionDigits: 2 })}`,
  //         data: new Date().toISOString().split('T')[0],
  //         validations: [
  //           'CPF/CNPJ válido',
  //           index % 3 === 1 ? 'NCM inconsistente' : 'NCM correto',
  //           index % 3 === 2 ? 'Valores inconsistentes' : 'Valores consistentes'
  //         ]
  //       }));

  //       setProcessedFiles(prev => [...mockProcessed, ...prev]);
  //     }, 2000);
  //   }, 1000);
  // };

  const handleFiles = async (fileList) => {
    const newFiles = Array.from(fileList).map((file, index) => ({
      id: Date.now() + index,
      file,
      name: file.name,
      size: file.size,
      status: 'uploading'
    }));

    setFiles((prev) => [...prev, ...newFiles]);
    setProcessing(true);

    // Processa cada arquivo individualmente
    for (const f of newFiles) {
      try {
        const base64 = await readFileAsBase64(f.file);
        await sendFileToAPI(base64);

        setFiles((prev) =>
          prev.map((item) =>
            item.id === f.id ? { ...item, status: 'completed' } : item
          )
        );
      } catch (err) {
        console.error(err);
        setFiles((prev) =>
          prev.map((item) =>
            item.id === f.id ? { ...item, status: 'error' } : item
          )
        );
      }
    }

    setProcessing(false);
  };

  const readFileAsBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64String = reader.result.split(',')[1]; // Remove o prefixo data:...
        resolve(base64String);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const sendFileToAPI = async (base64) => {
    const response = await fetch('https://i2a2-aiops-agentes-de-ti-n8n.bysger.easypanel.host/webhook/bde5b7b1-459f-4dee-85ea-a429dc4cf35d', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: base64 })
    });

    if (!response.ok) {
      throw new Error('Falha ao enviar arquivo');
    }

    return await response.json();
  };


  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <FileText className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'warning':
        return 'bg-orange-50 border-orange-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-white mb-6">Upload de Documentos Fiscais</h2>

      {/* Upload Zone */}
      <div className="fiscal-card rounded-xl p-8">
        <div
          className={`upload-zone rounded-lg p-12 text-center transition-all duration-300 ${dragActive ? 'dragover' : ''
            }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-upload"
            multiple
            accept=".zip"
            onChange={handleChange}
            className="hidden"
          />
          <Upload className="w-16 h-16 fiscal-green mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Arraste arquivos aqui</h3>
          <p className="text-gray-600 mb-4">ou clique para selecionar arquivos</p>
          <label
            htmlFor="file-upload"
            className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors cursor-pointer inline-block"
          >
            Selecionar Arquivos ZIP
          </label>
          <p className="text-sm text-gray-500 mt-4">
            Formatos aceitos: .zip | Tamanho máximo: 10MB por arquivo
          </p>
        </div>
      </div>

      {/* Upload Progress */}
      {files.length > 0 && (
        <div className="fiscal-card rounded-xl p-6">
          <h3 className="text-xl font-semibold mb-4">Arquivos em Processamento</h3>
          <div className="space-y-3">
            {files.map((file) => (
              <div key={file.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="w-5 h-5 text-gray-500" />
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-gray-600">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {file.status === 'uploading' && (
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                  )}
                  {file.status === 'processing' && (
                    <div className="animate-pulse text-blue-600">Processando...</div>
                  )}
                  {file.status === 'completed' && (
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Processed Files */}
      <div className="fiscal-card rounded-xl p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-semibold">Documentos Processados</h3>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>Exportar Relatório</span>
          </button>
        </div>

        <div className="space-y-4">
          {processedFiles.map((file) => (
            <div key={file.id} className={`border rounded-lg p-4 ${getStatusColor(file.status)}`}>
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  {getStatusIcon(file.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800">{file.name}</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 text-sm text-gray-600">
                      <p><strong>Emitente:</strong> {file.emitente}</p>
                      <p><strong>Valor:</strong> {file.valor}</p>
                      <p><strong>Data:</strong> {file.data}</p>
                    </div>
                    <div className="mt-3">
                      <p className="text-sm font-medium text-gray-700 mb-1">Validações:</p>
                      <div className="flex flex-wrap gap-2">
                        {file.validations.map((validation, index) => (
                          <span
                            key={index}
                            className={`px-2 py-1 rounded-full text-xs ${validation.includes('inconsistente') || validation.includes('inválido')
                              ? 'bg-red-100 text-red-700'
                              : 'bg-green-100 text-green-700'
                              }`}
                          >
                            {validation}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                <button className="text-blue-600 hover:text-blue-800 transition-colors">
                  <Eye className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;


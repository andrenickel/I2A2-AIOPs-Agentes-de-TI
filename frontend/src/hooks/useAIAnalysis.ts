import { useQuery } from '@tanstack/react-query';

interface AIAnalysis {
  resumo?: string;
  problemas?: Array<{
    campo: string;
    descricao: string;
    diferenca?: string;
  }>;
  riscos?: Array<{
    campo: string;
    descricao: string;
  }>;
  agressores?: Array<{
    campo: string;
    descricao: string;
  }>;
  recomendacoes?: Array<{
    campo: string;
    descricao: string;
  }>;
}

const mockAnalysis: AIAnalysis = {
  resumo: 'Análise detalhada da nota fiscal identificou alguns pontos de atenção relacionados à validação de dados e conformidade tributária.',
  problemas: [
    {
      titulo: 'Divergência de Valores',
      descricao: 'Foi identificada uma possível inconsistência entre o valor declarado e os itens discriminados.',
      diferenca: 'R$ 0,00'
    }
  ],
  riscos: [
    {
      titulo: 'Risco Fiscal Baixo',
      descricao: 'A operação apresenta conformidade com a legislação vigente.'
    }
  ],
  agressores: [
    {
      categoria: 'Validação',
      descricao: 'Todos os campos obrigatórios estão preenchidos corretamente.'
    }
  ],
  recomendacoes: [
    {
      titulo: 'Arquivamento Digital',
      descricao: 'Recomenda-se manter cópia digital da nota fiscal por no mínimo 5 anos.'
    }
  ]
};

export function useAIAnalysis(documentId: string | null) {
  return useQuery<AIAnalysis>({
    queryKey: ['ai-analysis', documentId],
    queryFn: async () => {
      if (!documentId) return mockAnalysis;
      
      try {
        const response = await fetch('http://localhost:8000/analise', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ chave_acesso: documentId }),
        });
        
        if (!response.ok) {
          return mockAnalysis;
        }
        
        const data = await response.json();
        
        // Se o retorno for null ou vazio, retorna dados mockados
        if (!data || Object.keys(data).length === 0) {
          return mockAnalysis;
        }
        
        return data;
      } catch (error) {
        console.error('Error fetching AI analysis:', error);
        return mockAnalysis;
      }
    },
    enabled: !!documentId,
    retry: 1,
    staleTime: 300000, // 5 minutes
  });
}

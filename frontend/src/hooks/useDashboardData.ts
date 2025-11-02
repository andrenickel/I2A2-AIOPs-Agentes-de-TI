import { useQuery } from '@tanstack/react-query';

interface DashboardResponse {
  periodo: {
    inicio: string;
    fim: string;
    anterior: {
      inicio: string;
      fim: string;
    };
  };
  kpis: {
    total_documentos: number;
    total_documentos_delta_pct: number;
    valor_total: number;
    valor_total_delta_pct: number;
    taxa_erro_pct: number;
    processamento_automatico_pct: number;
  };
  documentos_por_mes: Array<{
    mes: string;
    total: number;
  }>;
  distribuicao_fornecedor: Array<{
    fornecedor: string;
    valor: number;
    percentual: number;
  }>;
  evolucao_valor_total: Array<{
    mes: string;
    valor: number;
  }>;
  resumo_impostos: {
    disponivel: boolean;
    mensagem: string;
    icms: number | null;
    ipi: number | null;
    pis: number | null;
    cofins: number | null;
  };
}

const mockData: DashboardResponse = {
  periodo: {
    inicio: new Date().toISOString(),
    fim: new Date().toISOString(),
    anterior: {
      inicio: new Date().toISOString(),
      fim: new Date().toISOString()
    }
  },
  kpis: {
    total_documentos: 1247,
    total_documentos_delta_pct: 12,
    valor_total: 2100000,
    valor_total_delta_pct: 8,
    taxa_erro_pct: 2.3,
    processamento_automatico_pct: 97.7
  },
  documentos_por_mes: [
    { mes: '2024-01', total: 65 },
    { mes: '2024-02', total: 78 },
    { mes: '2024-03', total: 90 },
    { mes: '2024-04', total: 81 },
    { mes: '2024-05', total: 95 },
    { mes: '2024-06', total: 102 }
  ],
  distribuicao_fornecedor: [
    { fornecedor: 'Fornecedor A', valor: 735000, percentual: 35 },
    { fornecedor: 'Fornecedor B', valor: 525000, percentual: 25 },
    { fornecedor: 'Fornecedor C', valor: 420000, percentual: 20 },
    { fornecedor: 'Fornecedor D', valor: 315000, percentual: 15 },
    { fornecedor: 'Outros', valor: 105000, percentual: 5 }
  ],
  evolucao_valor_total: [
    { mes: '2024-01', valor: 125000 },
    { mes: '2024-02', valor: 145000 },
    { mes: '2024-03', valor: 180000 },
    { mes: '2024-04', valor: 165000 },
    { mes: '2024-05', valor: 195000 },
    { mes: '2024-06', valor: 210000 }
  ],
  resumo_impostos: {
    disponivel: true,
    mensagem: '',
    icms: 45000,
    ipi: 12000,
    pis: 8000,
    cofins: 15000
  }
};

export function useDashboardData() {
  return useQuery<DashboardResponse>({
    queryKey: ['dashboard-data'],
    queryFn: async () => {
      try {
        const response = await fetch('http://localhost:8000/dashboard');
        if (!response.ok) {
          return mockData;
        }
        return response.json();
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        return mockData;
      }
    },
    refetchInterval: 30000,
    retry: 1,
    staleTime: 10000,
  });
}

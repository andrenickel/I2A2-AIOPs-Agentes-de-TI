import { useQuery } from '@tanstack/react-query';

interface HomeStatsResponse {
  valor_total: number;
  total_documentos: number;
  documentos_analisados: number;
}

export function useHomeStats() {
  return useQuery<HomeStatsResponse>({
    queryKey: ['home-stats'],
    queryFn: async () => {
      try {
        const response = await fetch('http://localhost:8000/home');
        if (!response.ok) {
          throw new Error('Failed to fetch home stats');
        }
        return response.json();
      } catch (error) {
        console.error('Error fetching home stats:', error);
        return {
          valor_total: 0,
          total_documentos: 0,
          documentos_analisados: 0
        };
      }
    },
    refetchInterval: 30000,
    retry: 1,
    staleTime: 10000,
  });
}

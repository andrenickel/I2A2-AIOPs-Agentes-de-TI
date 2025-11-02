import { useQuery } from '@tanstack/react-query';
import { Document } from '@/components/documents/DocumentCard';

const mockDocuments: Document[] = [
  {
    id: '1',
    name: 'NFe_12345678901234567890.xml',
    company: 'Empresa ABC Ltda',
    cnpj: '12.345.678/0001-90',
    value: 15750.0,
    status: 'success',
    date: new Date(2024, 0, 15),
    tags: ['Eletrônicos', 'Urgente'],
    items: 12,
    icms: 2835.0,
  },
  {
    id: '2',
    name: 'NFe_09876543210987654321.xml',
    company: 'Fornecedor XYZ S.A.',
    cnpj: '98.765.432/0001-10',
    value: 8320.5,
    status: 'warning',
    date: new Date(2024, 0, 14),
    tags: ['Alimentação'],
    items: 5,
    icms: 1497.69,
  },
  {
    id: '3',
    name: 'NFe_11111111111111111111.xml',
    company: 'Empresa DEF ME',
    cnpj: '11.111.111/0001-11',
    value: 2100.0,
    status: 'error',
    date: new Date(2024, 0, 13),
    tags: ['Serviços'],
    items: 1,
    icms: 378.0,
  },
  {
    id: '4',
    name: 'NFe_22222222222222222222.xml',
    company: 'Tech Solutions Ltda',
    cnpj: '22.222.222/0001-22',
    value: 45000.0,
    status: 'success',
    date: new Date(2024, 0, 12),
    tags: ['TI', 'Hardware'],
    items: 25,
    icms: 8100.0,
  },
  {
    id: '5',
    name: 'NFe_33333333333333333333.xml',
    company: 'Comércio ABC ME',
    cnpj: '33.333.333/0001-33',
    value: 1250.75,
    status: 'success',
    date: new Date(2024, 0, 11),
    tags: ['Material de Escritório'],
    items: 8,
    icms: 225.14,
  },
];

export function useDocuments() {
  return useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: async () => {
      try {
        const response = await fetch('https://i2a2-aiops-agentes-de-ti-n8n.bysger.easypanel.host/webhook/b6021f43-0ee5-4fdc-9d2b-ae06f45d1888');
        
        if (!response.ok) {
          return mockDocuments;
        }
        
        const data = await response.json();
        
        // Se retorno for null ou vazio, usa dados mockados
        if (!data || (Array.isArray(data) && data.length === 0)) {
          return mockDocuments;
        }
        
        // Transform API data to match Document interface
        if (Array.isArray(data)) {
          return data.map((item: any, index: number) => {
            const valor = parseFloat(item.valor_nota_fiscal || 0);
            const cnpj = item.cpf_cnpj_emitente || '';
            const formattedCnpj = cnpj.length === 14 
              ? cnpj.replace(/(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})/, '$1.$2.$3/$4-$5')
              : cnpj;
            
            // Determinar status baseado no evento mais recente
            let status: 'success' | 'warning' | 'error' = 'success';
            const evento = (item.evento_mais_recente || '').toLowerCase();
            if (evento.includes('cancelamento') || evento.includes('denegado')) {
              status = 'error';
            } else if (evento.includes('pendente') || evento === '') {
              status = 'warning';
            }
            
            return {
              id: item.chave_acesso || String(index + 1),
              name: `NFe_${item.numero || index}.xml`,
              company: item.razao_social_emitente || 'N/A',
              cnpj: formattedCnpj,
              value: valor,
              status: status,
              date: item.data_emissao ? new Date(item.data_emissao) : new Date(),
              tags: item.natureza_operacao ? [item.natureza_operacao.substring(0, 30)] : [],
              items: 1,
              icms: 0,
            };
          });
        }
        
        return mockDocuments;
      } catch (error) {
        console.error('Error fetching documents:', error);
        return mockDocuments;
      }
    },
    refetchInterval: 30000,
    retry: 1,
    staleTime: 10000,
  });
}

import { memo } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Filter, Download, Maximize2 } from 'lucide-react';

interface DashboardFiltersProps {
  selectedPeriod: string;
  onPeriodChange: (period: string) => void;
  onExport: () => void;
  onFullscreen: () => void;
}

export const DashboardFilters = memo(({ 
  selectedPeriod, 
  onPeriodChange,
  onExport,
  onFullscreen 
}: DashboardFiltersProps) => {
  return (
    <div className="flex flex-wrap gap-4 items-center justify-between">
      <div className="flex items-center gap-2">
        <Filter className="w-5 h-5 text-muted-foreground" />
        <Select value={selectedPeriod} onValueChange={onPeriodChange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Selecione o período" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="week">Esta Semana</SelectItem>
            <SelectItem value="month">Este Mês</SelectItem>
            <SelectItem value="quarter">Este Trimestre</SelectItem>
            <SelectItem value="year">Este Ano</SelectItem>
            <SelectItem value="custom">Período Personalizado</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={onExport}
        >
          <Download className="w-4 h-4 mr-2" />
          Exportar
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={onFullscreen}
        >
          <Maximize2 className="w-4 h-4 mr-2" />
          Tela Cheia
        </Button>
      </div>
    </div>
  );
});

DashboardFilters.displayName = 'DashboardFilters';

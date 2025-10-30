import { useState, memo, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, FileText, AlertTriangle, CheckCircle, Sparkles } from 'lucide-react';
import { Card } from './ui/card';
import { StatCard } from './ui/stat-card';
import { Skeleton } from './ui/skeleton';
import { toast } from './ui/use-toast';
import { DashboardFilters } from './dashboard/DashboardFilters';
import { IntelligentAlert } from './dashboard/IntelligentAlert';
import { RecentActivity } from './dashboard/RecentActivity';
import { GoalsCard } from './dashboard/GoalsCard';
import { useDashboardData } from '@/hooks/useDashboardData';

const Dashboard = memo(() => {
  const [selectedPeriod, setSelectedPeriod] = useState('month');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  
  const { data: dashboardData, isLoading, isError } = useDashboardData();

  // Process monthly data for chart
  const monthlyData = useMemo(() => {
    if (!dashboardData?.documentos_por_mes) return [];
    return dashboardData.documentos_por_mes.map(item => ({
      name: new Date(item.mes).toLocaleDateString('pt-BR', { month: 'short' }),
      documentos: item.total,
      valor: 0
    }));
  }, [dashboardData]);

  // Process supplier data for pie chart
  const supplierData = useMemo(() => {
    if (!dashboardData?.distribuicao_fornecedor) return [];
    const colors = [
      'hsl(147, 50%, 36%)',
      'hsl(207, 90%, 54%)',
      'hsl(45, 100%, 50%)',
      'hsl(0, 84%, 60%)',
      'hsl(0, 0%, 70%)'
    ];
    return dashboardData.distribuicao_fornecedor.map((item, index) => ({
      name: item.fornecedor,
      value: item.percentual,
      color: colors[index % colors.length]
    }));
  }, [dashboardData]);

  // Process evolution data for line chart
  const evolutionData = useMemo(() => {
    if (!dashboardData?.evolucao_valor_total) return [];
    return dashboardData.evolucao_valor_total.map(item => ({
      name: new Date(item.mes).toLocaleDateString('pt-BR', { day: '2-digit', month: 'short' }),
      valor: item.valor
    }));
  }, [dashboardData]);

  // Tax data - use mock if not available
  const taxData = useMemo(() => {
    if (dashboardData?.resumo_impostos?.disponivel && 
        dashboardData.resumo_impostos.icms !== null) {
      return [
        { name: 'ICMS', valor: dashboardData.resumo_impostos.icms, percentual: 18 },
        { name: 'IPI', valor: dashboardData.resumo_impostos.ipi ?? 0, percentual: 5 },
        { name: 'PIS', valor: dashboardData.resumo_impostos.pis ?? 0, percentual: 1.65 },
        { name: 'COFINS', valor: dashboardData.resumo_impostos.cofins ?? 0, percentual: 7.6 },
      ];
    }
    // Fallback to mock data
    return [
      { name: 'ICMS', valor: 45000, percentual: 18 },
      { name: 'IPI', valor: 12000, percentual: 5 },
      { name: 'PIS', valor: 8000, percentual: 1.65 },
      { name: 'COFINS', valor: 15000, percentual: 7.6 },
    ];
  }, [dashboardData]);

  // AI Predictions
  const predictions = useMemo(() => ({
    nextMonth: {
      documents: 115,
      value: 225000,
      confidence: 87
    }
  }), []);

  const handleExport = () => {
    toast({
      title: "Exportando dados...",
      description: "Seu relatório será baixado em instantes.",
    });
    
    // Simulate export
    setTimeout(() => {
      toast({
        title: "Download concluído!",
        description: "Relatório exportado com sucesso.",
      });
    }, 2000);
  };

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handlePeriodChange = (period: string) => {
    setSelectedPeriod(period);
  };

  const toggleCardExpansion = (cardId: string) => {
    setExpandedCard(expandedCard === cardId ? null : cardId);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-96" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      </div>
    );
  }

  if (!dashboardData) return null;

  return (
    <div className="space-y-6">
      {/* Header with filters */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 animate-fade-in">
        <h2 className="text-3xl font-bold text-white">Dashboard Fiscal</h2>
        <div data-tour="filters">
          <DashboardFilters
            selectedPeriod={selectedPeriod}
            onPeriodChange={handlePeriodChange}
            onExport={handleExport}
            onFullscreen={handleFullscreen}
          />
        </div>
      </div>

      {/* Intelligent Alerts */}
      <div className="space-y-3 animate-fade-in" style={{ animationDelay: '0.1s' }}>
        <IntelligentAlert
          type="warning"
          title="Atenção Necessária"
          message="12 documentos pendentes de validação. Prazo: 2 dias."
          action={() => toast({ title: "Navegando para validações..." })}
          actionLabel="Ver agora"
        />
        <IntelligentAlert
          type="trend"
          title="Previsão IA"
          message={`Baseado no histórico, você deve processar ~${predictions.nextMonth.documents} documentos no próximo mês (${predictions.nextMonth.confidence}% confiança).`}
        />
      </div>

      {/* KPI Cards with Animation */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 animate-fade-in" data-tour="kpi-cards" style={{ animationDelay: '0.2s' }}>
        <StatCard
          title="Total de Documentos"
          value={dashboardData.kpis.total_documentos}
          icon={FileText}
          trend={{ 
            value: Math.abs(dashboardData.kpis.total_documentos_delta_pct), 
            isPositive: dashboardData.kpis.total_documentos_delta_pct > 0 
          }}
        />
        
        <StatCard
          title="Valor Total"
          value={Math.round(dashboardData.kpis.valor_total / 1000000)}
          suffix="M"
          prefix="R$ "
          icon={DollarSign}
          trend={{ 
            value: Math.abs(dashboardData.kpis.valor_total_delta_pct), 
            isPositive: dashboardData.kpis.valor_total_delta_pct > 0 
          }}
        />
        
        <StatCard
          title="Taxa de Erro"
          value={dashboardData.kpis.taxa_erro_pct}
          suffix="%"
          icon={AlertTriangle}
          trend={{ value: 0, isPositive: false }}
        />
        
        <StatCard
          title="Processamento"
          value={dashboardData.kpis.processamento_automatico_pct}
          suffix="%"
          icon={CheckCircle}
          trend={{ value: 0, isPositive: true }}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" data-tour="charts">
        {/* Monthly Documents Chart */}
        <Card 
          className={`bg-card/95 backdrop-blur-sm p-6 border-border/50 hover:shadow-xl transition-all duration-300 cursor-pointer animate-fade-in ${
            expandedCard === 'documents' ? 'lg:col-span-2' : ''
          }`}
          style={{ animationDelay: '0.3s' }}
          onClick={() => toggleCardExpansion('documents')}
        >
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold">Documentos por Mês</h3>
            <span className="text-xs text-muted-foreground">Clique para expandir</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="documentos" fill="hsl(147, 50%, 36%)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Supplier Distribution */}
        <Card 
          className={`bg-card/95 backdrop-blur-sm p-6 border-border/50 hover:shadow-xl transition-all duration-300 cursor-pointer animate-fade-in ${
            expandedCard === 'suppliers' ? 'lg:col-span-2' : ''
          }`}
          style={{ animationDelay: '0.4s' }}
          onClick={() => toggleCardExpansion('suppliers')}
        >
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold">Distribuição por Fornecedor</h3>
            <span className="text-xs text-muted-foreground">Clique para expandir</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={supplierData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {supplierData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Revenue Chart with AI Prediction */}
      <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50 animate-fade-in" style={{ animationDelay: '0.5s' }}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Evolução do Valor Total</h3>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Sparkles className="w-4 h-4 text-primary" />
            <span>Com previsão IA</span>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={evolutionData}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip 
              formatter={(value) => [`R$ ${Number(value).toLocaleString('pt-BR')}`, 'Valor']} 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="valor" 
              stroke="hsl(207, 90%, 54%)" 
              strokeWidth={3}
              dot={{ fill: 'hsl(207, 90%, 54%)', r: 6 }}
              activeDot={{ r: 8 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Goals and Activity Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="animate-fade-in" data-tour="goals" style={{ animationDelay: '0.6s' }}>
          <GoalsCard />
        </div>
        <div className="animate-fade-in" style={{ animationDelay: '0.7s' }}>
          <RecentActivity />
        </div>
      </div>

      {/* Tax Summary */}
      <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50 animate-fade-in" style={{ animationDelay: '0.8s' }}>
        <h3 className="text-xl font-semibold mb-6">Resumo de Impostos</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {taxData.map((tax, index) => (
            <div 
              key={index} 
              className="bg-gradient-to-br from-muted/50 to-muted/30 rounded-lg p-4 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 border border-border/50"
            >
              <h4 className="font-semibold text-foreground mb-2">{tax.name}</h4>
              <p className="text-2xl font-bold text-success mb-1">
                R$ {tax.valor.toLocaleString()}
              </p>
              <p className="text-sm text-muted-foreground">
                {tax.percentual}% alíquota média
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
});

Dashboard.displayName = 'Dashboard';

export default Dashboard;

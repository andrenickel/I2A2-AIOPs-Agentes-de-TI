import { memo } from 'react';
import { Link } from 'react-router-dom';
import { 
  Upload, 
  FileText, 
  BarChart3, 
  Shield, 
  Clock, 
  CheckCircle2,
  TrendingUp,
  Users,
  Award,
  DollarSign
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { StatCard } from '@/components/ui/stat-card';
import { TrustBadge } from '@/components/ui/trust-badge';
import { useHomeStats } from '@/hooks/useHomeStats';

const HomePage = memo(() => {
  const { data: stats, isLoading, isError } = useHomeStats();

  return (
    <div className="space-y-16">
      {/* Hero Section with Animation */}
      <section className="text-center mb-12 animate-fade-in-up">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full mb-6 animate-pulse-slow">
          <Award className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-white">
            Líder em Automação Fiscal com IA
          </span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
          Automatize Sua Gestão Fiscal{' '}
          <span className="bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
            com IA
          </span>
        </h1>
        
        <p className="text-xl text-white/90 mb-8 max-w-3xl mx-auto leading-relaxed">
          Processe documentos XML de NF-e em segundos, com validação inteligente 
          e relatórios automáticos. Transforme dias de trabalho em minutos.
        </p>
        
        <div className="flex flex-col sm:flex-row justify-center gap-4 mb-8" data-tour="hero-cta">
          <Button
            asChild
            size="lg"
            className="bg-white text-primary hover:bg-white/90 shadow-lg hover:shadow-xl transition-all hover:scale-105"
          >
            <Link to="/upload">
              <Upload className="mr-2" />
              Começar Agora
            </Link>
          </Button>
          <Button
            asChild
            variant="outline"
            size="lg"
            className="border-2 border-white text-white bg-white/10 hover:bg-white/20 backdrop-blur-sm"
          >
            <Link to="/dashboard">
              <BarChart3 className="mr-2" />
              Ver Dashboard
            </Link>
          </Button>
        </div>

        {/* Trust Badges */}
        <div className="flex flex-wrap justify-center gap-4 animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <TrustBadge icon={Shield} text="100% Seguro" />
          <TrustBadge icon={CheckCircle2} text="Certificado SEFAZ" />
          <TrustBadge icon={Users} text="2.500+ Empresas" />
          <TrustBadge icon={Clock} text="Suporte 24/7" />
        </div>
      </section>

      {/* Features Cards */}
      <section className="grid md:grid-cols-3 gap-8 mb-12" data-tour="features">
        <Card className="bg-card/95 backdrop-blur-sm p-6 text-center border-border/50 hover:shadow-xl transition-all duration-500 hover:-translate-y-2 group animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <div className="w-16 h-16 mx-auto mb-4 bg-primary/10 rounded-2xl flex items-center justify-center group-hover:bg-primary/20 transition-colors group-hover:scale-110 duration-300">
            <Upload className="w-8 h-8 text-primary group-hover:animate-float" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Upload Simples</h3>
          <p className="text-muted-foreground leading-relaxed">
            Arraste e solte seus arquivos XML. Nossa IA processa automaticamente 
            e extrai todas as informações necessárias.
          </p>
          <div className="mt-4 pt-4 border-t border-border/50">
            <p className="text-sm text-primary font-medium">
              ⚡ Processamento em menos de 5 segundos
            </p>
          </div>
        </Card>

        <Card className="bg-card/95 backdrop-blur-sm p-6 text-center border-border/50 hover:shadow-xl transition-all duration-500 hover:-translate-y-2 group animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <div className="w-16 h-16 mx-auto mb-4 bg-secondary/10 rounded-2xl flex items-center justify-center group-hover:bg-secondary/20 transition-colors group-hover:scale-110 duration-300">
            <FileText className="w-8 h-8 text-secondary group-hover:animate-float" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Validação Inteligente</h3>
          <p className="text-muted-foreground leading-relaxed">
            IA avançada detecta inconsistências, erros e possíveis problemas 
            antes que se tornem críticos.
          </p>
          <div className="mt-4 pt-4 border-t border-border/50">
            <p className="text-sm text-secondary font-medium">
              🎯 98.5% de precisão na validação
            </p>
          </div>
        </Card>

        <Card className="bg-card/95 backdrop-blur-sm p-6 text-center border-border/50 hover:shadow-xl transition-all duration-500 hover:-translate-y-2 group animate-fade-in" style={{ animationDelay: '0.3s' }}>
          <div className="w-16 h-16 mx-auto mb-4 bg-primary/10 rounded-2xl flex items-center justify-center group-hover:bg-primary/20 transition-colors group-hover:scale-110 duration-300">
            <BarChart3 className="w-8 h-8 text-primary group-hover:animate-float" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Relatórios Automáticos</h3>
          <p className="text-muted-foreground leading-relaxed">
            Dashboards interativos e relatórios detalhados gerados automaticamente 
            com insights acionáveis.
          </p>
          <div className="mt-4 pt-4 border-t border-border/50">
            <p className="text-sm text-primary font-medium">
              📊 Visualização em tempo real
            </p>
          </div>
        </Card>
      </section>

      {/* Stats Section */}
      <section className="grid md:grid-cols-3 gap-6 animate-fade-in" data-tour="stats" style={{ animationDelay: '0.4s' }}>
        <StatCard
          title="Valor Total"
          value={isLoading || isError ? 0 : (stats?.valor_total ?? 0)}
          prefix="R$ "
          icon={DollarSign}
        />
        
        <StatCard
          title="Total de Documentos"
          value={isLoading || isError ? 0 : (stats?.total_documentos ?? 0)}
          icon={FileText}
        />
        
        <StatCard
          title="Documentos Analisados"
          value={isLoading || isError ? 0 : (stats?.documentos_analisados ?? 0)}
          icon={BarChart3}
        />
      </section>

      {/* CTA Section */}
      <section className="text-center bg-gradient-to-r from-primary to-secondary rounded-2xl p-12 animate-scale-in" style={{ animationDelay: '0.8s' }}>
        <h2 className="text-3xl font-bold text-white mb-4">
          Pronto Para Revolucionar Sua Gestão Fiscal?
        </h2>
        <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
          Junte-se a milhares de empresas que já economizam tempo e dinheiro
        </p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <Button
            asChild
            size="lg"
            className="bg-white text-primary hover:bg-white/90 shadow-lg hover:shadow-xl transition-all hover:scale-105"
          >
            <Link to="/upload">
              Começar Teste Gratuito
            </Link>
          </Button>
          <Button
            asChild
            variant="outline"
            size="lg"
            className="border-2 border-white text-white bg-white/10 hover:bg-white/20"
          >
            <Link to="/chat">
              Falar com IA
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
});

HomePage.displayName = 'HomePage';

export default HomePage;
